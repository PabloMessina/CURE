import argparse
import gc
import heapq
import json
import logging
import os
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from accelerate.utils import broadcast_object_list, gather_object
from peft import LoraConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from vlm_research_kit.data.dataset_helpers import (
    CompositeDataset,
    FormattedDataset,
    RandomSubsetDataset,
    WeightedCompositeDataset,
)
from vlm_research_kit.data.datasets.chest_imagenome_dataset import (
    CustomWeightedCompositeDataset,
    create_chest_imagenome_dataset,
)
from vlm_research_kit.data.datasets.mimiccxr_dataset import (
    MIMICCXRDataset,
)
from vlm_research_kit.data.datasets.mscxr_dataset import (
    MSCXRPhraseGroundingDataset,
)
from vlm_research_kit.data.datasets.padchest_dataset import (
    PadChestGRDataset,
    PadChestGRPhraseGroundingDataset,
)
from vlm_research_kit.metrics.biomedical.cxrfescore import CXRFEScore
from vlm_research_kit.utils.bbox_utils import calculate_bbox_union_iou
from vlm_research_kit.utils.data_utils import convert_to_serializable
from vlm_research_kit.utils.file_utils import (
    get_safe_filename,
    load_config_yaml,
    load_jsonl,
    setup_experiment_dir,
)
from vlm_research_kit.utils.logging_utils import (
    ANSI_BLUE_BOLD,
    ANSI_RESET,
    setup_logging,
)
from vlm_research_kit.utils.tracking_utils import finalize_wandb, initialize_wandb

# Setup logging as early as possible
setup_logging()
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism
# See: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_SEED = 42

class DiagnosticSFTTrainer(SFTTrainer):
    """
    A subclass of SFTTrainer that logs diagnostic information about the training
    data batches by overriding the training_step method.
    """

    def __init__(
        self,
        *args,
        diagnostics_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._diagnostics_enabled = diagnostics_config and diagnostics_config.get(
            "enabled", False
        )
        
        if self._diagnostics_enabled:
            self._diagnostics_log_every_n_batches = diagnostics_config["log_every_n_batches"]
            self._batches_count = 0 # Number of batches processed since the last diagnostics log
            self._diagnostics_output_filename = diagnostics_config.get(
                "output_filename", "training_sampling_diagnostics.jsonl"
            )
            self._diagnostics_buffer = []
            logger.info("DiagnosticSFTTrainer is enabled. Will log sampling data.")

    def _get_train_sampler(self, train_dataset: Optional[Dataset]) -> Optional[torch.utils.data.Sampler]:
        """
        Returns a sequential sampler for the training dataset.
        We do this in order to have absolute control of the shuffling of the training dataset
        through our own custom shuffling logic implemented in the dataset classes.
        """
        return torch.utils.data.SequentialSampler(train_dataset)

    def training_step(
        self, model: torch.nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: int
    ) -> torch.Tensor:
        """
        Overrides the default training_step to log diagnostic information
        from the input batch before performing the actual training step.
        """

        # First, handle our diagnostic logging
        if self._diagnostics_enabled:
            self._batches_count += 1
            batch_size = len(inputs["diagnostic_orig_index"])
            for i in range(batch_size):
                # The diagnostic info is not on a device, it's just a python list
                self._diagnostics_buffer.append(
                    {
                        "global_step": self.state.global_step,
                        "process_id": int(self.accelerator.process_index),
                        "orig_index": int(inputs["diagnostic_orig_index"][i]),
                        "actual_index": int(inputs["diagnostic_actual_index"][i]),
                        "dataset_name": inputs["diagnostic_dataset_name"][i],
                        "category": inputs["diagnostic_category"][i],
                    }
                )

            if self.accelerator.is_main_process:
                logger.info(f"batches_count: {self._batches_count}, batch_size: {batch_size},"
                            f" Local diagnostics buffer size: {len(self._diagnostics_buffer)}")

            # Periodically gather and save the data
            if (
                self._batches_count % self._diagnostics_log_every_n_batches == 0 and
                self._batches_count > 0
            ):
                self._gather_and_save_diagnostics()

                
        # Now, perform the original training step to get the loss
        loss = super().training_step(model, inputs, num_items_in_batch)

        return loss

    def _gather_and_save_diagnostics(self):
        """
        Gathers the diagnostic buffer from all processes and saves it to a file
        on the main process.
        """

        if len(self._diagnostics_buffer) == 0:
            return # No data to save

        # gather_object is a blocking operation, ensuring all processes participate
        gathered_results = gather_object(self._diagnostics_buffer)        

        if self.accelerator.is_main_process:
            output_path = os.path.join(
                self.args.output_dir, self._diagnostics_output_filename
            )
            try:
                # Append to the file
                with open(output_path, "a", encoding="utf-8") as f:
                    for record in gathered_results:
                        f.write(json.dumps(record) + "\n")
                logger.info(f"Saved {len(gathered_results)} diagnostic logs to {output_path}")
            except Exception as e:
                logger.error(
                    f"Failed to save diagnostics to {output_path}: {e}",
                    exc_info=True,
                )
        
        self._diagnostics_buffer.clear()  # Clear local buffer after gathering


class FinalDiagnosticLogCallback(TrainerCallback):
    """
    A simple callback that calls the trainer's diagnostic save function
    at the end of training to flush any remaining data in the buffer.
    """

    def __init__(self):
        self.trainer = None # Will be set by the setup_and_train function

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if hasattr(self.trainer, "_gather_and_save_diagnostics"):
            logger.info(f"Flushing final diagnostic logs from {type(self.trainer).__name__}...")
            self.trainer._gather_and_save_diagnostics()


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedGemma Model")
    parser.add_argument(
        "--train_config_path",
        type=str,
        required=True,
        help="Path to the main training YAML configuration file.",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help=(
            "Directory to save checkpoints, logs, and other outputs. "
            "If None, defaults to EXPERIMENTS_DIR / run_name."
        ),
    )
    return parser.parse_args()


def _format_bbox_string(bbox: List[float], decimal_places: int = 2) -> str:
    """Formats a single bounding box into a string representation."""
    return f"[{','.join([f'{coord:.{decimal_places}f}' for coord in bbox])}]"


def format_chest_imagenome(example: dict[str, Any], include_ground_truth: bool = True) -> dict[str, Any]:
    """Converts a Chest-ImaGenome sample to the SFTTrainer format."""
    # The image transform for Chest-ImaGenome returns a PIL image in 'pixel_values'
    image = example["pixel_values"]
    del example["pixel_values"]
    example["image"] = image
    if include_ground_truth:
        example["messages"] = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": example["prompt"]}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": example["report"]}]},
        ]
    else:
        example["messages"] = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": example["prompt"]}
            ]},
        ]
        example["gt_text"] = example["report"]
    return example


def format_grounded_report_gen(example: dict[str, Any], include_ground_truth: bool = True) -> dict[str, Any]:
    """Converts a PadChest-GR (report gen) sample to the SFTTrainer format."""
    prompt = "Generate a grounded report."
    image = example["image"]
    if include_ground_truth:
        example["messages"] = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": example["report"]}]},
        ]
    else:
        example["messages"] = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]},
        ]
        example["gt_text"] = example["report"]
    return example

def format_report_gen(example: dict[str, Any], include_ground_truth: bool = True) -> dict[str, Any]:
    """Converts a MIMIC-CXR report gen sample to the SFTTrainer format."""
    prompt = "Generate a report."
    image = example["image"]
    if include_ground_truth:
        example["messages"] = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}

            ]},
            {"role": "assistant", "content": [{"type": "text", "text": example["report"]}]},
        ]
    else:
        example["messages"] = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]},
        ]
        example["gt_text"] = example["report"]
    return example

def format_phrase_grounding(example: dict[str, Any], include_ground_truth: bool = True) -> dict[str, Any]:
    """Converts a phrase grounding sample (PadChest/MS-CXR) to SFTTrainer format."""
    # These datasets return a PIL image directly in the 'image' key
    prompt = f"Ground the phrase: {example['phrase']}"
    image = example["image"]
    # Format the ground truth bounding boxes into a single string
    report = f'{example["phrase"]}: {" ".join([_format_bbox_string(bbox) for bbox in example["gt_bboxes"]])}'
    if include_ground_truth:
        example["messages"] = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": report}]},
        ]
    else:
        example["messages"] = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]},
        ]
        example["gt_text"] = report
    return example


def create_collate_fn(
    processor: Any, has_diagnostics: bool = False,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Factory function that creates and returns a collate_fn.
    The returned collate_fn will have access to the provided processor via a closure.
    """

    def collate_fn_inner(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The actual collation logic."""
        texts = []
        images = []
        # --- Prepare to collect diagnostic info ---        
        truly_has_diagnostics = has_diagnostics and "diagnostic_orig_index" in examples[0]
        
        if truly_has_diagnostics:
            diag_orig_indices = []
            diag_actual_indices = []
            diag_dataset_names = []
            diag_categories = []

        for example in examples:
            # This `processor` is the one from the outer scope
            images.append([example["image"].convert("RGB")])
            texts.append(
                processor.apply_chat_template(
                    example["messages"],
                    add_generation_prompt=False,
                    tokenize=False,
                ).strip()
            )
            # --- Collect diagnostic info if present ---
            if truly_has_diagnostics:
                diag_orig_indices.append(example["diagnostic_orig_index"])
                diag_actual_indices.append(example["diagnostic_actual_index"])
                diag_dataset_names.append(example["diagnostic_dataset_name"])
                diag_categories.append(example["diagnostic_category"])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100
        batch["labels"] = labels
        # --- Add diagnostic lists to the final batch ---
        if truly_has_diagnostics:
            batch["diagnostic_orig_index"] = diag_orig_indices
            batch["diagnostic_actual_index"] = diag_actual_indices
            batch["diagnostic_dataset_name"] = diag_dataset_names
            batch["diagnostic_category"] = diag_categories

        return batch

    return collate_fn_inner


# A new Callback to trigger resampling before each evaluation
class ResampleCallback(TrainerCallback):
    """
    A TrainerCallback that calls the `resample()` method on a dataset
    at the beginning of each evaluation phase.
    """

    def __init__(self, dataset_to_resample):
        self.dataset = dataset_to_resample
        self.trainer = None # Will be set by the setup_and_train function

    def on_evaluate(self, args, state, control, **kwargs):
        """Event hook called before every evaluation."""
        if hasattr(self.dataset, "resample") and callable(self.dataset.resample):
            self.dataset.resample()
            # Log the new dataset size if this is the main process
            if self.trainer.accelerator.is_main_process:
                logger.info(f"Resampled validation set to {len(self.dataset)} random samples.")


class TopKCheckpointCallback(TrainerCallback):
    """
    A TrainerCallback that saves copies of the top-K checkpoints based on a
    validation metric. It automatically resumes its state by scanning the
    checkpoint directory at the beginning of training.

    Args:
        top_k (int): The number of best checkpoints to keep.
        metric_name (str): The name of the metric to use for ranking.
        greater_is_better (bool): Whether a higher value of the metric is better.
        topk_dir_name (str): The name of the subdirectory to save top-K
                             checkpoints.
    """

    def __init__(
        self,
        top_k: int = 3,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
        topk_dir_name: str = "topk-checkpoints",
    ):
        self.top_k = int(top_k)
        self.metric_name = metric_name
        self.sign = 1.0 if greater_is_better else -1.0
        self.topk_dir_name = topk_dir_name

        # Min-heap of (signed_score, step, src_ckpt_path)
        self._heap = []
        # step -> copied_dest_path
        self._copied: Dict[int, str] = {}

        # state captured at last eval to associate with the next save
        self._last_metric: Optional[float] = None
        self._last_step: Optional[int] = None
        self._has_initialized = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize state by scanning disk when training starts."""
        if not self._has_initialized:
            self._initialize_from_disk(args)
            self._has_initialized = True

    def _initialize_from_disk(self, args: TrainingArguments):
        """Scans the top-k directory to repopulate state from a previous run."""
        topk_root = os.path.join(args.output_dir, self.topk_dir_name)
        if not os.path.isdir(topk_root):
            return

        # Regex to parse: checkpoint-{step}-{metric_name}-{metric_value}
        pattern = re.compile(
            rf"checkpoint-(\d+)-{re.escape(self.metric_name)}-(-?\d+\.?\d*)"
        )

        found_checkpoints = []
        for dirname in os.listdir(topk_root):
            full_path = os.path.join(topk_root, dirname)
            if not os.path.isdir(full_path):
                continue

            match = pattern.match(dirname)
            if not match:
                continue

            step = int(match.group(1))
            metric_value = float(match.group(2))
            signed_score = self.sign * metric_value

            found_checkpoints.append((signed_score, step, full_path))
            self._copied[step] = full_path

        # Rebuild the heap from the found checkpoints
        heapq.heapify(found_checkpoints)
        self._heap = found_checkpoints

        logger.info(
            f"Resumed TopKCheckpointCallback: Found {len(self._heap)} "
            f"checkpoints in {topk_root}:"
        )
        for signed_score, step, full_path in self._heap:
            logger.info(f"  - {self.sign * signed_score:.4f} at step {step}: {full_path}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics:
            raise ValueError(f"TopKCheckpointCallback on_evaluate: no metrics or {self.metric_name} not in metrics")
        self._last_metric = float(metrics[self.metric_name])
        self._last_step = int(state.global_step)
        logger.info(f"TopKCheckpointCallback on_evaluate: last_metric={self._last_metric}, last_step={self._last_step}")

    def on_save(self, args, state, control, **kwargs):
        # Initial check to ensure we have a metric for this save step
        if self._last_metric is None:
            logger.info("TopKCheckpointCallback on_save: no last_metric, skipping save.")
            return # There was no new evaluation before this save, so we skip it.

        logger.info(f"TopKCheckpointCallback on_save: last_metric={self._last_metric}, last_step={self._last_step}")
        assert self._last_step is not None
        assert self._last_step == state.global_step, (
            f"last_step={self._last_step} != current_step={state.global_step}"
        )  # This should always be true, but we check it anyway.

        step = self._last_step
        metric_value = self._last_metric
        signed_score = self.sign * metric_value

        # Find the checkpoint directory created by the Trainer
        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(ckpt_path):
            last_ckpt = get_last_checkpoint(args.output_dir)
            if last_ckpt and last_ckpt.endswith(str(step)):
                ckpt_path = last_ckpt
            else:
                logger.warning(
                    f"Top-K Checkpoints: Expected checkpoint dir for step {step} "
                    f"not found, skipping."
                )
                self._clear_last()
                return

        # This step was already processed (e.g., from a previous run)
        if step in self._copied:
            self._clear_last()
            return

        # *** THE NEW SANITY CHECK LOGIC ***
        # If the heap is not full, or if the new score is better than the worst
        # score in the heap, then we proceed.
        if len(self._heap) < self.top_k or signed_score > self._heap[0][0]:
            # If the heap is already full, remove the worst checkpoint
            if len(self._heap) == self.top_k:
                worst_signed, worst_step, _ = heapq.heappop(self._heap)
                doomed_path = self._copied.pop(worst_step, None)
                if doomed_path and os.path.isdir(doomed_path):
                    logger.info(
                        f"Top-K Checkpoints: Removing checkpoint for step "
                        f"{worst_step} (score: {self.sign * worst_signed:.4f}) "
                        f"as it is no longer in the top {self.top_k}."
                    )
                    shutil.rmtree(doomed_path, ignore_errors=True)

            # Add the new checkpoint to the heap
            heapq.heappush(self._heap, (signed_score, step, ckpt_path))

            # Copy the new top-k checkpoint to its destination
            topk_root = os.path.join(args.output_dir, self.topk_dir_name)
            os.makedirs(topk_root, exist_ok=True)
            metric_tag = f"{self.metric_name}-{metric_value:.6f}"
            filename = f"checkpoint-{step}-{metric_tag}"
            filename = get_safe_filename(filename)
            dest_path = os.path.join(topk_root, filename)

            if not os.path.isdir(dest_path):
                shutil.copytree(ckpt_path, dest_path, dirs_exist_ok=True)

            self._copied[step] = dest_path
            logger.info(
                f"Top-K Checkpoints: Saved new top-{self.top_k} checkpoint for "
                f"step {step} with {self.metric_name}: {metric_value:.4f}"
            )

        else:
            # This checkpoint is not in the top K, so we do nothing with it
            logger.info(
                f"Top-K Checkpoints: Skipping checkpoint for step {step} "
                f"(score: {metric_value:.4f}) as it is not in the top {self.top_k}."
            )

        # Clean up state for the next evaluation
        self._clear_last()

    def _clear_last(self):
        self._last_metric = None
        self._last_step = None


def default_data_collator(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A simple data collator that batches inputs and labels.
    Converts list of dicts to dict of batched tensors/lists.
    """
    batch = {}
    for key in examples[0].keys():
        if isinstance(examples[0][key], torch.Tensor):
            batch[key] = torch.stack([ex[key] for ex in examples])
        else:
            batch[key] = [ex[key] for ex in examples]
    return batch


BBOX_REGEX = re.compile(
    r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]"
)
DESCRIPTION_REGEX = re.compile(r"Description.*?:(.*)")

def _compute_iou_and_clean_text(row: pd.Series):
    """Helper function to compute IoU and clean text for a single row."""
    gt_text: str = row["ground_truth"]
    pred_text: str = row["prediction"]

    # 1. Extract bounding boxes
    gt_bboxes_str = BBOX_REGEX.findall(gt_text)
    pred_bboxes_str = BBOX_REGEX.findall(pred_text)

    gt_bboxes = [[float(c) for c in b] for b in gt_bboxes_str]
    pred_bboxes = [[float(c) for c in b] for b in pred_bboxes_str]

    # 2. Compute IoU if applicable
    bbox_iou = np.nan
    # Only compute IoU if the ground truth is supposed to have boxes
    if gt_bboxes:
        bbox_iou = calculate_bbox_union_iou(
            gt_bboxes, pred_bboxes, bbox_format="cxcywh"
        )

    # 3. Clean text by removing bboxes
    clean_gt = BBOX_REGEX.sub("", gt_text).strip()
    clean_gt = " ".join(clean_gt.split()) # Compress extra whitespace
    description_match = DESCRIPTION_REGEX.search(clean_gt)
    if description_match:
        clean_gt = description_match.group(1).strip()
    clean_gt = (
        clean_gt.replace(": .", ".").replace(" .", ".").replace(":", ".")
    )

    clean_pred = BBOX_REGEX.sub("", pred_text).strip()
    clean_pred = " ".join(clean_pred.split()) # Compress extra whitespace
    description_match = DESCRIPTION_REGEX.search(clean_pred)
    if description_match:
        clean_pred = description_match.group(1).strip()
    clean_pred = (
        clean_pred.replace(": .", ".").replace(" .", ".").replace(":", ".")
    )

    return pd.Series(
        [bbox_iou, clean_gt, clean_pred],
        index=["bbox_iou", "clean_gt", "clean_pred"],
    )


class GranularEvaluationCallback(TrainerCallback):
    """
    Performs a custom evaluation loop using model.generate() when activated.
    It computes and logs detailed metrics and a single aggregate score for checkpointing.
    """

    def __init__(
        self,
        eval_datasets: list,
        dataset_configs: list,
        processor,
        config: dict,
    ):
        assert len(eval_datasets) == len(
            dataset_configs
        ), "eval_datasets and dataset_configs must have the same length."
        self.eval_datasets = eval_datasets
        self.dataset_configs = dataset_configs
        self.processor = processor
        self.default_samples = config["custom_eval_samples_per_dataset"]
        self.default_max_tokens = config.get("custom_eval_max_new_tokens", 200)
        self.aggregate_metric_name = config["custom_metric_for_best_model"]
        self.save_predictions_to_file = config.get("save_predictions_to_file", False)
        self.save_predictions_to_file_name = config.get("save_predictions_to_file_name", "predictions_custom_eval.jsonl")
        self.trainer = None
        self.text_metric = CXRFEScore(device='cpu')
        self.metric_weights = config["metric_weights"]
        bbox_iou_weight = self.metric_weights["bbox_iou"]
        cxrfescore_weight = self.metric_weights["cxrfescore"]
        assert 0.0 <= bbox_iou_weight <= 1.0
        assert 0.0 <= cxrfescore_weight <= 1.0
        assert bbox_iou_weight + cxrfescore_weight == 1.0
        logger.info(f"Metric weights: bbox_iou_weight={bbox_iou_weight}, cxrfescore_weight={cxrfescore_weight}")
        logger.info("CXRFEScore metric initialized on device: cpu")
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, Any],
        model,
        **kwargs,
    ):
        if self.trainer is None:
            raise RuntimeError(
                "The trainer attribute has not been set on the GranularEvaluationCallback."
            )
        process_id = self.trainer.accelerator.process_index
        logger.info(
            f"[Rank {process_id}] Starting robust granular evaluation with batched model.generate()..."
        )
        model.eval()
        all_results = []

        # # Use a set to keep track of which datasets we've already printed a debug prompt for.
        # # Initialize it at the beginning of the on_evaluate method.
        # if 'debug_datasets_printed' not in self.__dict__:
        #     self.debug_datasets_printed = set()

        # --- Temporarily switch to left-padding for generation ---
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"

        per_device_eval_batch_size = args.per_device_eval_batch_size
        num_devices = self.trainer.accelerator.num_processes

        for dataset, ds_config in zip(self.eval_datasets, self.dataset_configs):
            # --- FEATURE: Use per-dataset settings from config, with defaults ---
            max_tokens = ds_config.get("max_new_tokens", self.default_max_tokens)
            num_samples = ds_config.get("eval_samples", self.default_samples)
            dataset_name = f"{ds_config['name']}_{ds_config.get('task', '')}".rstrip('_')
            if hasattr(dataset, "create_uniform_subset"):
                dataset.create_uniform_subset(
                    target_size=num_samples,
                    ensure_multiple_of=per_device_eval_batch_size * num_devices
                )
                assert len(dataset) >= num_samples, f"Expected at least {num_samples} samples, but got {len(dataset)}."
                assert len(dataset) % (per_device_eval_batch_size * num_devices) == 0, (
                    f"Expected {len(dataset)} samples to be a multiple of {per_device_eval_batch_size * num_devices}."
                )
                logger.info(
                    f"[Rank: {process_id}] Created uniform subset for {dataset_name}. "
                    f"Target size: {num_samples}. Actual size: {len(dataset)}"
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.per_device_eval_batch_size,
                    collate_fn=default_data_collator,
                )
            else:
                raise ValueError(f"Dataset {dataset_name} does not have a create_uniform_subset method.")
            
            dataloader = self.trainer.accelerator.prepare(dataloader)

            # --- Only show tqdm on the main process ---
            iterable = dataloader
            if self.trainer.accelerator.is_main_process:
                iterable = tqdm(dataloader, desc=f"Generating on {dataset_name}")

            for batch in iterable:
                inputs = self.processor.apply_chat_template(
                    batch['messages'], # Directly use batched messages
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)

                # # --- ENHANCED DEBUG BLOCK ---
                # if self.trainer.accelerator.is_main_process and dataset_name not in self.debug_datasets_printed:
                #     print("\n" + "="*20 + f" DEBUGGING PROMPT FOR: {dataset_name} " + "="*20)
                #     # Decode and print the first prompt in this batch
                #     decoded_prompt = self.processor.decode(inputs['input_ids'][0], skip_special_tokens=False)
                #     print(decoded_prompt.replace("assistant\n", "assistant\n\n"))
                #     print("="*(58 + len(dataset_name)) + "\n")
                #     # Mark this dataset as printed so we don't print it again this evaluation cycle
                #     self.debug_datasets_printed.add(dataset_name)
                # # --- END DEBUG BLOCK ---

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

                decoded_preds = self.processor.batch_decode(outputs, skip_special_tokens=True)

                for i in range(len(decoded_preds)):
                    category = (
                        batch["location"][i] if "location" in batch else None
                        or batch["category_name"][i] if "category_name" in batch else None
                        or batch["label_group"][i] if "label_group" in batch else None
                        or "unknown"
                    )
                    prediction_text = decoded_preds[i].split("model\n")[-1].strip()
                    ground_truth_text = batch["gt_text"][i]
                    image_path = batch["image_path"][i]
                    
                    all_results.append({
                        "prompt": batch["messages"][i][0]["content"][1]["text"],
                        "prediction": prediction_text,
                        "ground_truth": ground_truth_text,
                        "category": category,
                        "dataset_name": dataset_name,
                        "process_id": process_id,
                        "image_path": image_path,
                    })

        # --- Restore original padding side ---
        self.processor.tokenizer.padding_side = original_padding_side

        # Gather results from all processes
        gathered_results = gather_object(all_results)

        # Define a placeholder container on all processes for the combined weights object.
        resampling_weights_container = [None]

        if self.trainer.accelerator.is_main_process:
            # First, compute the metrics. This now returns the detailed metrics dict.
            computed_metrics = self._compute_and_log_metrics(gathered_results, metrics)
            # Now, calculate the weights. This is the INTER-dataset weights.
            dataset_weights = self._calculate_dataset_weights_from_metrics(computed_metrics)
            # Then, calculate the category weights. This is the INTRA-dataset weights.
            category_weights = self._calculate_category_weights_from_metrics(computed_metrics)
            # Combine them into a single object for broadcasting.
            combined_weights = {
                "dataset_weights": dataset_weights,
                "category_weights": category_weights,
            }
            resampling_weights_container[0] = combined_weights
            # Save the predictions to file
            if self.save_predictions_to_file:
                self._save_predictions_to_file(gathered_results, combined_weights, args, state)

        # This is a synchronization point. Rank 0's object is sent to all others.
        broadcast_object_list(resampling_weights_container)
        
        # Now, the container on every process holds the complete weights object.
        final_resampling_weights = resampling_weights_container[0]

        self.resampling_weights = final_resampling_weights # Store the weights for later use

        # --- VERIFICATION LOGGING ---
        process_id = self.trainer.accelerator.process_index
        if final_resampling_weights is not None:
            # Using json.dumps for pretty printing the nested dictionary
            weights_str = json.dumps(final_resampling_weights, indent=2)
            logger.info(f"[Rank {process_id}] Successfully received new resampling weights:\n{weights_str}")
        else:
            logger.warning(f"[Rank {process_id}] Did not receive new resampling weights.")

        # # VERY IMPORTANT: Reset the set at the end of the on_evaluate method
        # # so that the prompts are printed again on the *next* evaluation step.
        # if self.trainer.accelerator.is_main_process:
        #     self.debug_datasets_printed = set()

    def _calculate_dataset_weights_from_metrics(self, computed_metrics: dict) -> dict:
        """
        Calculates new INTER-dataset sampling weights based on evaluation metrics.
        Implements Error-Based Weighting.
        """
        raw_weights = {}
        total_error = 0.0
        epsilon = 1e-6

        for dataset_name, data in computed_metrics.items():
            iou = data["overall"].get("bbox_iou")
            sim = data["overall"].get("cxrfescore")
            iou_present = iou is not None and not pd.isna(iou)
            sim_present = sim is not None and not pd.isna(sim)
            if iou_present and sim_present:
                performance_score = self.metric_weights["bbox_iou"] * iou + self.metric_weights["cxrfescore"] * sim
            elif iou_present:
                performance_score = iou
            elif sim_present:
                performance_score = sim
            else:
                raise ValueError(f"No metrics present for dataset {dataset_name}")
            error = 1.0 - performance_score
            if error <= 0:
                logger.warning(f"Error for dataset {dataset_name} is {error}. This is not allowed. Replacing with 0.01.")
                error = 0.01
            raw_weights[dataset_name] = error
            total_error += error
            
        normalized_weights = {}
        if total_error > epsilon:
            for name, raw_w in raw_weights.items():
                normalized_weights[name] = raw_w / total_error
        else:
            num_datasets = len(raw_weights)
            for name in raw_weights:
                normalized_weights[name] = 1.0 / num_datasets if num_datasets > 0 else 0.0
                
        return normalized_weights


    def _calculate_category_weights_from_metrics_helper(self, per_category_data: dict, epsilon: float = 1e-6) -> dict:
        """
        Calculates new INTRA-category sampling weights based on evaluation metrics.
        """
        raw_cat_weights = {}
        total_cat_error = 0.0
        
        with_iou_count = 0
        with_sim_count = 0
        for category, cat_metrics in per_category_data.items():
            iou = cat_metrics.get("bbox_iou")
            sim = cat_metrics.get("cxrfescore")
            iou_present = iou is not None and not pd.isna(iou)
            sim_present = sim is not None and not pd.isna(sim)
            with_iou_count += 1 if iou_present else 0
            with_sim_count += 1 if sim_present else 0
            if iou_present and sim_present:
                performance_score = self.metric_weights["bbox_iou"] * iou + self.metric_weights["cxrfescore"] * sim
            elif iou_present:
                performance_score = iou
            elif sim_present:
                performance_score = sim
            else:
                raise ValueError(f"No metrics present for category {category}")
            error = 1.0 - performance_score
            if error <= 0:
                logger.warning(f"Error for category {category} is {error}. This is not allowed. Replacing with 0.01.")
                error = 0.01
            raw_cat_weights[category] = error
            total_cat_error += error
        assert with_iou_count == len(per_category_data) or with_iou_count == 0, (
            f"Expected {len(per_category_data)} or 0 IoU metrics, but got {with_iou_count}"
        )
        assert with_sim_count == len(per_category_data) or with_sim_count == 0, (
            f"Expected {len(per_category_data)} or 0 CXRFEScore metrics, but got {with_sim_count}"
        )

        # Normalize weights
        normalized_cat_weights = {}
        if total_cat_error > epsilon:
            for cat, raw_w in raw_cat_weights.items():
                normalized_cat_weights[cat] = raw_w / total_cat_error
        else:
            num_categories = len(raw_cat_weights)
            for cat in raw_cat_weights:
                normalized_cat_weights[cat] = 1.0 / num_categories if num_categories > 0 else 0.0
        
        return normalized_cat_weights

    def _calculate_category_weights_from_metrics(self, computed_metrics: dict) -> dict:
        """
        Calculates new INTRA-dataset (per-category) sampling weights.
        """
        all_category_weights = {}
        epsilon = 1e-6

        for dataset_name, data in computed_metrics.items():
            per_category_data = data.get("per_category", {})
            if not per_category_data:
                continue # Skip if no per-category metrics for this dataset

            if dataset_name == "chest-imagenome": # Special case for chest-imagenome dataset
                all_category_weights[dataset_name] = {}
                for prompt_type, prompt_type_data in per_category_data.items():
                    normalized_cat_weights = self._calculate_category_weights_from_metrics_helper(prompt_type_data, epsilon)
                    all_category_weights[dataset_name][prompt_type] = normalized_cat_weights
            else: # General case for other datasets
                normalized_cat_weights = self._calculate_category_weights_from_metrics_helper(per_category_data, epsilon)
                all_category_weights[dataset_name] = normalized_cat_weights
            
        return all_category_weights

    def _compute_and_log_metrics(
        self,
        results: list,
        metrics: Dict[str, Any],
    ):
        if not results:
            raise ValueError("No evaluation results to compute metrics on.")

        logger.info("Computing metrics (IoU, CXRFEScore)...")
        df = pd.DataFrame(results)

        # Apply the helper to compute IoU and clean text for each row
        iou_and_clean_df = df.apply(_compute_iou_and_clean_text, axis=1)
        df = pd.concat([df, iou_and_clean_df], axis=1)

        # Compute text similarity in a batch for efficiency
        clean_gts = df["clean_gt"].tolist()
        clean_preds = df["clean_pred"].tolist()
        sim_results = self.text_metric.compute(hyps=clean_preds, refs=clean_gts)
        df["cxrfescore"] = sim_results["per_pair_similarity"]

        self.text_metric.save_cache() # Save the cache to disk

        # --- 3. Structure metrics into the desired dictionary format ---
        computed_metrics = defaultdict(lambda: {"overall": {}, "per_category": {}})

        # Calculate overall metrics per dataset
        overall_metrics_df = df.groupby("dataset_name")[
            ["bbox_iou", "cxrfescore"]
        ].mean()
        dataset_sizes = df.groupby("dataset_name").size().to_dict() # Size of each dataset
        for dataset_name, series in overall_metrics_df.iterrows():
            computed_metrics[dataset_name]["overall"] = series.to_dict()
            computed_metrics[dataset_name]["overall"]["iou_cxrfescore"] = ( # Aggregate metric for checkpointing
                self.metric_weights["bbox_iou"] * series["bbox_iou"] +
                self.metric_weights["cxrfescore"] * series["cxrfescore"]
            )
            computed_metrics[dataset_name]["overall"]["size"] = dataset_sizes[dataset_name]

        # Calculate per-category metrics per dataset
        
        # 1) Handle chest-imagenome dataset separately
        df_chest_imagenome = df[df["dataset_name"] == "chest-imagenome"].copy() # Copy to avoid SettingWithCopyWarning
        if not df_chest_imagenome.empty:
            # Classify the prompt into one of three types: "locate_and_describe", "describe", "locate"
            df_chest_imagenome["prompt_type"] = df_chest_imagenome["prompt"].apply(
                lambda x: (
                    "locate_and_describe" if x.lower().startswith("locate and describe") else
                    "describe" if x.lower().startswith("describe") else
                    "locate" if x.lower().startswith("locate") else None
                )
            )
            assert df_chest_imagenome["prompt_type"].notna().all(), "Some prompts were not classified into a valid type."

            # Step 1: Create a global mapping of location -> average metrics, across all prompts.
            # Some values can be NaN here if a metric isn't applicable to any prompt for a given location.
            location_metrics_all_prompts = df_chest_imagenome.groupby("category")[
                ["bbox_iou", "cxrfescore"]
            ].mean().to_dict('index')

            # Initialize the nested structure for chest-imagenome per-category metrics
            computed_metrics["chest-imagenome"]["per_category"] = {}

            prompt_types_present = df_chest_imagenome["prompt_type"].unique()

            # Step 2 & 3: Iterate through prompt types and populate the final structure
            # with the required metrics, asserting they are valid.
            if "locate" in prompt_types_present:
                computed_metrics["chest-imagenome"]["per_category"]["locate"] = {}
                locations_for_locate = df_chest_imagenome[
                    df_chest_imagenome["prompt_type"] == "locate"
                ]["category"].unique()
                assert len(locations_for_locate) > 0, "No locations found for 'locate' prompt type."
                for location in locations_for_locate:
                    iou = location_metrics_all_prompts[location]["bbox_iou"]
                    assert not pd.isna(iou) and iou is not None, f"IoU score for location '{location}' with 'locate' prompt is NaN."
                    computed_metrics["chest-imagenome"]["per_category"]["locate"][location] = {"bbox_iou": iou}
            else:
                raise ValueError("Prompt type 'locate' is not present in the dataset.")

            if "describe" in prompt_types_present:
                computed_metrics["chest-imagenome"]["per_category"]["describe"] = {}
                locations_for_describe = df_chest_imagenome[
                    df_chest_imagenome["prompt_type"] == "describe"
                ]["category"].unique()
                assert len(locations_for_describe) > 0, "No locations found for 'describe' prompt type."
                for location in locations_for_describe:
                    sim = location_metrics_all_prompts[location]["cxrfescore"]
                    assert not pd.isna(sim) and sim is not None, f"CXRFEScore for location '{location}' with 'describe' prompt is NaN."
                    computed_metrics["chest-imagenome"]["per_category"]["describe"][location] = {"cxrfescore": sim}
            else:
                raise ValueError("Prompt type 'describe' is not present in the dataset.")

            if "locate_and_describe" in prompt_types_present:
                computed_metrics["chest-imagenome"]["per_category"]["locate_and_describe"] = {}
                locations_for_both = df_chest_imagenome[
                    df_chest_imagenome["prompt_type"] == "locate_and_describe"
                ]["category"].unique()
                assert len(locations_for_both) > 0, "No locations found for 'locate_and_describe' prompt type."
                for location in locations_for_both:
                    metrics_for_loc = location_metrics_all_prompts[location]
                    iou = metrics_for_loc["bbox_iou"]
                    sim = metrics_for_loc["cxrfescore"]
                    assert not pd.isna(iou) and iou is not None, f"IoU score for location '{location}' with 'locate_and_describe' prompt is NaN."
                    assert not pd.isna(sim) and sim is not None, f"CXRFEScore for location '{location}' with 'locate_and_describe' prompt is NaN."
                    computed_metrics["chest-imagenome"]["per_category"]["locate_and_describe"][location] = {
                        "bbox_iou": iou,
                        "cxrfescore": sim,
                    }
            else:
                raise ValueError("Prompt type 'locate_and_describe' is not present in the dataset.")
        
        # 2) Handle the other datasets
        for dataset_name in df["dataset_name"].unique():
            if dataset_name != "chest-imagenome":
                df_dataset = df[df["dataset_name"] == dataset_name]
                per_category_metrics_df = df_dataset.groupby(["category"])[
                    ["bbox_iou", "cxrfescore"]
                ].mean()
                for category, series in per_category_metrics_df.iterrows():
                    computed_metrics[dataset_name]["per_category"][category] = series.to_dict()

        # --- 4. Prepare logs for WandB and Trainer ---
        wandb_logs = {}

        # Calculate and add overall metrics (macro average)
        mean_iou = 0
        iou_count = 0
        mean_sim = 0
        sim_count = 0
        for dataset_name, data in computed_metrics.items():
            size = data["overall"]["size"]
            assert size > 0, f"Size for dataset {dataset_name} is 0."
            if "bbox_iou" in data["overall"]:
                mean_iou += data["overall"]["bbox_iou"] * size
                iou_count += size
            if "cxrfescore" in data["overall"]:
                mean_sim += data["overall"]["cxrfescore"] * size
                sim_count += size
        assert iou_count > 0, "No IoU metrics found"
        assert sim_count > 0, "No CXRFEScore metrics found"
        mean_iou /= iou_count
        mean_sim /= sim_count        
        wandb_logs["eval_overall_bbox_iou"] = mean_iou
        wandb_logs["eval_overall_cxrfescore"] = mean_sim
        wandb_logs["eval_overall_iou_cxrfescore"] = (
            self.metric_weights["bbox_iou"] * mean_iou +
            self.metric_weights["cxrfescore"] * mean_sim
        )

        # Add per-dataset overall metrics
        for dataset_name, data in computed_metrics.items():
            if "bbox_iou" in data["overall"]:
                wandb_logs[f"eval_{dataset_name}/bbox_iou"] = data["overall"]["bbox_iou"]
            if "cxrfescore" in data["overall"]:
                wandb_logs[f"eval_{dataset_name}/cxrfescore"] = data["overall"]["cxrfescore"]
            if "iou_cxrfescore" in data["overall"]:
                wandb_logs[f"eval_{dataset_name}/iou_cxrfescore"] = data["overall"]["iou_cxrfescore"]

        # Ensure the aggregate metric for checkpointing is present
        if self.aggregate_metric_name not in wandb_logs:
            raise ValueError(
                f"The specified aggregate_metric_name '{self.aggregate_metric_name}' was not computed. "
                f"Available metrics: {list(wandb_logs.keys())}. "
                "TopKCheckpointCallback may not work as expected. "
                "Please check the configuration and the metrics computed."
            )
        else:
            logger.info(
                f"==> Aggregate Metric '{self.aggregate_metric_name}': {wandb_logs[self.aggregate_metric_name]:.4f}"
            )

        wandb_logs = convert_to_serializable(wandb_logs) # Convert to serializable
        
        metrics.update(wandb_logs)

        # Log to WandB
        self.trainer.log(wandb_logs)

        return computed_metrics

    def _save_predictions_to_file(self, results: list, combined_weights: dict, args: TrainingArguments, state: TrainerState):
        # This block is executed only on the main process, which is safe for file writing.
        try:
            output_file_path = os.path.join(args.output_dir, self.save_predictions_to_file_name)
            
            # Create a dictionary that includes the current training step and a timestamp
            data_to_save = {
                "global_step": state.global_step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "evaluation_results": results,
                "combined_weights": combined_weights,
            }
            
            # Open the file in append mode and write the JSON object as a single line
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data_to_save) + "\n")
            
            logger.info(f"Saved {len(results)} evaluation results to {output_file_path}")

        except Exception as e:
            logger.error(f"Failed to save evaluation results to file: {e}", exc_info=True)


class StopAfterFirstEvalCallback(TrainerCallback):
    """A callback that signals training to stop after the first evaluation."""
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after an evaluation phase.
        """
        logger.info("First evaluation complete. Signaling training to stop.")
        # The magic line: this tells the trainer's main loop to break.
        control.should_training_stop = True


def sanity_check_resampling_weights(resampling_weights: dict):
    """
    Sanity checks the resampling weights to ensure they are valid.
    Args:
        resampling_weights: The resampling weights to check.
    Returns:
        None
    """
    for key, value in resampling_weights.items():
        if isinstance(value, dict):
            sanity_check_resampling_weights(value)
        elif isinstance(value, (int, float)):
            if value <= 0:
                logger.warning(f"Resampling weight for {key} is {value}. This is not allowed. Replacing with 0.01.")
                resampling_weights[key] = 0.01
            if value > 1:
                logger.warning(f"Resampling weight for {key} is {value}. This is not allowed. Replacing with 1.0.")
                resampling_weights[key] = 1.0
        else:
            raise ValueError(f"Invalid resampling weights: {value}")

def apply_resampling_weights(
    final_train_dataset: WeightedCompositeDataset,
    resampling_weights: dict,
    is_main_process: bool,
    update_intra_dataset_weights: bool = True,
    update_inter_dataset_weights: bool = True,
):
    """
    Applies the resampling weights to the training dataset.
    Args:
        final_train_dataset: The final training dataset.
        resampling_weights: The resampling weights to apply.
        is_main_process: Whether the current process is the main process.
        update_intra_dataset_weights: Whether to update the intra-dataset weights.
        update_inter_dataset_weights: Whether to update the inter-dataset weights.
    Returns:
        None
    """
    if is_main_process:
        logger.info(f"Applying resampling weights: {resampling_weights}")
    
    # Sanity check the resampling weights
    sanity_check_resampling_weights(resampling_weights)
    
    # Update the category weights within each dataset.
    if update_intra_dataset_weights:
        if is_main_process:
            logger.info("Updating intra-dataset weights.")
        for dataset_name_, weights in resampling_weights["category_weights"].items():
            dataset = final_train_dataset.name_to_dataset[dataset_name_]
            assert isinstance(dataset, FormattedDataset)
            dataset = dataset.base_dataset
            if isinstance(dataset, CustomWeightedCompositeDataset):
                dataset.update_sampling_weights(category_weights=weights, dataset_weights=[1] * len(dataset.datasets))
            elif hasattr(dataset, "update_sampling_weights"):
                dataset.update_sampling_weights(category_weights=weights)
            else:
                logger.warning(
                    f"Dataset {dataset_name_} does not have `update_sampling_weights`. Skipping."
                )
    # Update the dataset weights within the final composite dataset.
    if update_inter_dataset_weights:
        if is_main_process:
            logger.info("Updating inter-dataset weights.")
        flattened_weights = [
            resampling_weights["dataset_weights"][x]
            for x in final_train_dataset.dataset_names
        ]
        final_train_dataset.update_weights(flattened_weights)

    if is_main_process:
        logger.info("Resampling weights applied successfully.")


def setup_and_train(
    base_model_id: str,
    output_dir: Path,
    train_dataset,
    is_main_process: bool,
    eval_dataset: Optional[Any] = None,
    eval_datasets: Optional[List[Any]] = None,
    eval_dataset_configs: Optional[List[Dict]] = None,
    use_flash_attention: bool = False,
    resume_from_checkpoint_path: Optional[str] = None,
    pretrained_lora_adapter_path: Optional[str] = None,
    lora_config: Optional[dict] = None,
    # Training Duration Control
    num_train_epochs: Optional[int] = 1,
    max_train_steps: int = -1,
    # Evaluation Control
    eval_strategy: str = "steps",
    eval_steps: int = 25,
    # Saving Control
    save_strategy: str = "epoch",
    save_steps: int = 500,
    save_top_k: int = 2, # Number of checkpoints to save.
    # Additional Training Arguments
    learning_rate: float = 2e-4,
    training_args_dict: Optional[dict] = None,
    evaluation_config: Optional[dict] = None,
    diagnostics_config: Optional[dict] = None,
):
    """
    Sets up and runs the SFT training process for a SINGLE cycle, optionally returning
    the new resampling weights.
    """

    # --- Validate evaluation configuration ---
    eval_mode = evaluation_config.get("mode", "default")
    if eval_mode == "custom":
        assert eval_datasets is not None, "eval_datasets must be provided when using custom evaluation."
        assert eval_dataset_configs is not None, "eval_dataset_configs must be provided when using custom evaluation."
        assert len(eval_datasets) == len(eval_dataset_configs), "eval_datasets and eval_dataset_configs must have the same length."
    elif eval_mode == "default":
        assert eval_dataset is not None, "eval_dataset must be provided when using default evaluation."
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")

    # --- 1. Create Processor & Collator ---
    processor = AutoProcessor.from_pretrained(base_model_id)
    processor.tokenizer.padding_side = "right"
    has_diagnostics = diagnostics_config is not None and diagnostics_config.get("enabled", False)
    data_collator = create_collate_fn(processor, has_diagnostics)

    # --- 2. Load Quantization Config and Base Model ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # Get the current device from the environment variables set by torchrun/accelerate
    # This will be 'cuda:0' for rank 0, 'cuda:1' for rank 1, etc.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": f"cuda:{local_rank}"}
    attn_implementation = "flash_attention_2" if use_flash_attention else "eager"
    logger.info(f"Using {attn_implementation.replace('_', ' ').title()}")

    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    # --- 3. Define PEFT Config ---
    if lora_config:
        lora_alpha = lora_config.get("lora_alpha", 16)
        lora_dropout = lora_config.get("lora_dropout", 0.05)
        r = lora_config.get("r", 16)
        bias = lora_config.get("bias", "none")
        target_modules = lora_config.get("target_modules", "all-linear")
        modules_to_save = lora_config.get("modules_to_save", ["lm_head", "embed_tokens"])
    else: # Default lora config
        lora_alpha = 16
        lora_dropout = 0.05
        r = 16
        bias = "none"
        target_modules = "all-linear"
        modules_to_save = ["lm_head", "embed_tokens"]
    # peft_config = LoraConfig(
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     r=16,
    #     bias="none",
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    #     modules_to_save=["lm_head", "embed_tokens"],
    # )
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=r,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )
    if is_main_process:
        logger.info(f"Peft Config: {peft_config}")
        logger.info(f"Lora Alpha: {lora_alpha}")
        logger.info(f"Lora Dropout: {lora_dropout}")
        logger.info(f"R: {r}")
        logger.info(f"Bias: {bias}")
        logger.info(f"Target Modules: {target_modules}")
        logger.info(f"Modules to Save: {modules_to_save}")

    # --- 4. Define Training Arguments ---
    default_args = {
        "output_dir": str(output_dir),
        "num_train_epochs": num_train_epochs,
        "max_steps": max_train_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "logging_steps": 25,
        "learning_rate": learning_rate,
        "bf16": True,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "linear",
        "push_to_hub": False,
        "report_to": "wandb",
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "ddp_find_unused_parameters": False, # Added to avoid warnings
        "seed": BASE_SEED,
    }
    if training_args_dict:
        default_args.update(training_args_dict)

    args = SFTConfig(**default_args)

    # --- 5. Initialize Callbacks ---
    callbacks = []

    # --- Add the final flush callback if diagnostics are enabled ---
    if diagnostics_config and diagnostics_config.get("enabled", False):
        callbacks.append(FinalDiagnosticLogCallback())

    if eval_mode == "custom":
        logger.info("Setting up callbacks for CUSTOM evaluation mode for ALL processes.")
        # This callback MUST run on all processes for the distributed logic to work.
        eval_callback = GranularEvaluationCallback(
            eval_datasets=eval_datasets,
            dataset_configs=eval_dataset_configs,
            processor=processor,
            config=evaluation_config,
        )
        callbacks.append(eval_callback)
        callbacks.append(StopAfterFirstEvalCallback())
    elif isinstance(eval_dataset, RandomSubsetDataset):
        callbacks.append(ResampleCallback(eval_dataset))

    # This callback manages file I/O and should ONLY run on the main process.
    if is_main_process:
        logger.info("Setting up file-based callbacks for the main process ONLY.")
        if eval_mode == "custom":
            callbacks.append(
                TopKCheckpointCallback(
                    top_k=save_top_k,
                    metric_name=evaluation_config["custom_metric_for_best_model"],
                    greater_is_better=evaluation_config["custom_greater_is_better"],
                )
            )
        else: # Default mode
            callbacks.append(
                TopKCheckpointCallback(
                    top_k=save_top_k,
                    metric_name=training_args_dict.get("metric_for_best_model", "eval_loss"),
                    greater_is_better=training_args_dict.get("greater_is_better", False),
                )
            )

    # --- 6. Initialize Trainer ---
    # We ALWAYS pass the base model and peft_config. The trainer will create
    # the PeftModel with the correct structure.
    trainer = DiagnosticSFTTrainer( # We use the DiagnosticSFTTrainer to log diagnostic information
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config, # Let the trainer do the wrapping
        processing_class=processor,
        data_collator=data_collator,
        callbacks=callbacks,
        diagnostics_config=diagnostics_config, # Pass the diagnostics config to the trainer
    )

    # --- Set the trainer reference on our custom callback ---
    # This gives the callback access to the trainer's methods and accelerator
    for callback in trainer.callback_handler.callbacks:
        if hasattr(callback, 'trainer'):
            callback.trainer = trainer
            logger.info(f"Successfully set trainer reference on {type(callback).__name__}.")
    # ----------------------------------------------------------------

    # --- 6.1 Debug: Inspect model and trainable parameters after trainer init ---
    # logger.info("Model structure after SFTTrainer initialization:")
    # print(trainer.model)
    logger.info(f"[Rank: {local_rank}] Trainable parameters after SFTTrainer initialization:")
    trainer.model.print_trainable_parameters()

    # --- 7. Manually load adapter weights if a path is provided ---
    # This happens AFTER the trainer has prepared the model but BEFORE training starts.
    if pretrained_lora_adapter_path:
        logger.info(
            f"Loading pretrained adapter weights from: {pretrained_lora_adapter_path}"
        )
        trainer.model.load_adapter(pretrained_lora_adapter_path, adapter_name="default")
        logger.info("Successfully loaded adapter weights into the model.")

    # --- 8. Start Training ---
    # This call will now run for AT MOST `eval_steps` (or until another condition
    # is met) and then return because of the StopAfterFirstEvalCallback.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint_path)

    # After the train call, evaluation has run. Get the new weights from the callback.
    new_weights = None
    if eval_mode == "custom":
        assert hasattr(
            eval_callback, "resampling_weights"
        ), "Eval callback must have resampling_weights attribute."
        new_weights = eval_callback.resampling_weights
        assert new_weights is not None, "Resampling weights cannot be None after evaluation."

    # The trainer state now holds the latest step count
    final_step = trainer.state.global_step

    # --- NEW: AGGRESSIVE MANUAL CLEANUP ATTEMPT ---
    logger.info("Performing aggressive manual cleanup of Trainer and Model...")
    
    # Break the circular reference
    for callback in trainer.callback_handler.callbacks:
        if hasattr(callback, 'trainer'):
            del callback.trainer

    # Delete the biggest objects in a sensible order
    del trainer
    del model
    del processor
    del callbacks
    del peft_config
    del args
    del quantization_config
    del data_collator
    # -----------------------------------------------

    return new_weights, final_step


def main(debug: bool = False):
    args = parse_args()

    # Determine if this is the main process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    # --- 1. Load Configuration ---
    logger.info(f"Loading training configuration from: {args.train_config_path}")
    config = load_config_yaml(args.train_config_path)    
    if is_main_process:
        yaml_config_string = yaml.dump(config, indent=4)
        logger.info(
            f"Training Config:\n{ANSI_BLUE_BOLD}{yaml_config_string}{ANSI_RESET}"
        )
    # Get diagnostics config
    # Pop the diagnostics config so it's not passed to SFTConfig
    diagnostics_config = config["training"].pop("diagnostics", {"enabled": False})
    # Get lora config
    lora_config = config["model"].get("lora_config", None)
    if lora_config:
        assert 'lora_alpha' in lora_config, "lora_alpha must be specified in lora_config."
        assert 'lora_dropout' in lora_config, "lora_dropout must be specified in lora_config."
        assert 'r' in lora_config, "r must be specified in lora_config."
        assert 'bias' in lora_config, "bias must be specified in lora_config."
        assert 'target_modules' in lora_config, "target_modules must be specified in lora_config."
        assert 'modules_to_save' in lora_config, "modules_to_save must be specified in lora_config."
        assert isinstance(lora_config['lora_alpha'], int), "lora_alpha must be an integer."
        assert isinstance(lora_config['lora_dropout'], float), "lora_dropout must be a float."
        assert isinstance(lora_config['r'], int), "r must be an integer."
        assert isinstance(lora_config['bias'], str), "bias must be a string."
        assert isinstance(lora_config['target_modules'], (str, list)), "target_modules must be a string or a list."
        assert isinstance(lora_config['modules_to_save'], list), "modules_to_save must be a list."
    if is_main_process:
        logger.info(f"Lora Config: {lora_config}")

    # --- 2. Setup Experiment Directory ---
    experiment_dir = setup_experiment_dir(args.experiment_dir, config.get("run_name"))
    if is_main_process:
        logger.info(
            f"Outputs will be saved to: {ANSI_BLUE_BOLD}{experiment_dir}{ANSI_RESET}"
        )

    # --- 3. Initialize Tracking (Wandb) ---
    # Only initialize wandb on the main process
    wandb_run = None
    if is_main_process:
        wandb_run = initialize_wandb(experiment_dir=experiment_dir, config=config)

     # --- 4. Create Datasets ---
    logger.info("Instantiating datasets based on configuration...")
    disable_tqdm = not is_main_process

    # --- Training Datasets ---
    train_datasets_to_compose = []
    train_weights = []
    train_dataset_names = []
    concatenate_train_datasets_without_weighted_sampling = config["datasets"].get(
        "concatenate_train_datasets_without_weighted_sampling", False)
    
    for ds_config in config["datasets"]["train"]:
        name = ds_config["name"]
        task = ds_config.get("task") # Task might not be present for Chest-ImaGenome
        weight = ds_config.get("weight")
        args = ds_config["args"]
        # Pass the flag to the dataset constructor
        args["diagnostic_mode"] = diagnostics_config.get("enabled", False)
        
        formatter = None
        pytorch_dataset = None

        if name == "chest-imagenome":
            pytorch_dataset = create_chest_imagenome_dataset(**args, disable_tqdm=disable_tqdm)
            formatter = format_chest_imagenome
        elif name == "padchest-gr" and task == "grounded-report-generation":
            pytorch_dataset = PadChestGRDataset(**args)
            formatter = format_grounded_report_gen
        elif name == "padchest-gr" and task == "phrase-grounding":
            pytorch_dataset = PadChestGRPhraseGroundingDataset(**args)
            formatter = format_phrase_grounding
        elif name == "ms-cxr" and task == "phrase-grounding":
            pytorch_dataset = MSCXRPhraseGroundingDataset(**args)
            formatter = format_phrase_grounding
        elif name == "mimic-cxr" and task == "report-generation":
            pytorch_dataset = MIMICCXRDataset(**args)
            formatter = format_report_gen
        else:
            raise ValueError(f"Unknown dataset configuration: name={name}, task={task}")

        train_datasets_to_compose.append(FormattedDataset(pytorch_dataset, formatter))
        train_weights.append(weight)
        train_dataset_names.append(f"{name}_{task}" if task else name)
        if is_main_process:
            logger.info(f"Loaded training dataset '{name}_{task}' with {len(pytorch_dataset)} samples and weight {weight}.")

    if None in train_weights:
        assert all(weight is None for weight in train_weights), "All weights must be None or a float."
        train_weights = [1.0] * len(train_datasets_to_compose) # Default weight is 1.0
    else:
        assert all(isinstance(weight, float) for weight in train_weights), "All weights must be floats."
        assert all(weight > 0 for weight in train_weights), "All weights must be positive."

    # Combine training datasets using weights
    if not train_datasets_to_compose:
        raise ValueError("No training datasets were created.")
        
    if concatenate_train_datasets_without_weighted_sampling:
        logger.info("Concatenating training datasets without weighted sampling.")
        final_train_dataset = CompositeDataset(
            datasets=train_datasets_to_compose,
            dataset_names=train_dataset_names,
        )
    else:
        logger.info("Creating WeightedCompositeDataset for training.")
        final_train_dataset = WeightedCompositeDataset(
            datasets=train_datasets_to_compose,
            weights=train_weights,
            dataset_names=train_dataset_names,
        )

    # --- Validation Datasets ---
    val_datasets_to_compose = []
    val_dataset_configs = config["datasets"]["val"]
    for ds_config in val_dataset_configs:
        name = ds_config["name"]
        task = ds_config.get("task")
        args = ds_config["args"]

        formatter = None
        pytorch_dataset = None

        if name == "chest-imagenome":
            pytorch_dataset = create_chest_imagenome_dataset(**args, disable_tqdm=disable_tqdm)
            formatter = format_chest_imagenome
        elif name == "padchest-gr" and task == "grounded-report-generation":
            pytorch_dataset = PadChestGRDataset(**args)
            formatter = format_grounded_report_gen
        elif name == "padchest-gr" and task == "phrase-grounding":
            pytorch_dataset = PadChestGRPhraseGroundingDataset(**args)
            formatter = format_phrase_grounding
        elif name == "ms-cxr" and task == "phrase-grounding":
            pytorch_dataset = MSCXRPhraseGroundingDataset(**args)
            formatter = format_phrase_grounding
        elif name == "mimic-cxr" and task == "report-generation":
            pytorch_dataset = MIMICCXRDataset(**args)
            formatter = format_report_gen
        else:
            raise ValueError(f"Unknown validation dataset configuration: name={name}, task={task}")

        formatter = partial(formatter, include_ground_truth=False) # Don't include ground truth for validation datasets
        
        val_datasets_to_compose.append(FormattedDataset(pytorch_dataset, formatter))
        
        logger.info(f"[Rank: {local_rank}] Loaded validation dataset '{name}/{task}' with {len(pytorch_dataset)} samples.")
    
    if not val_datasets_to_compose:
        raise ValueError("No validation datasets were created.")

    # --- 5. Setup and Run Training ---
    training_config = config["training"]
    model_config = config["model"]

    save_top_k = training_config.pop("save_top_k", 2) # Number of checkpoints to save. Used by TopKCheckpointCallback.
    if not isinstance(save_top_k, int):
        raise ValueError("save_top_k must be an integer.")
    if save_top_k < 1:
        raise ValueError("save_top_k must be at least 1.")
    if is_main_process:
        logger.info(f"Save Top K: {save_top_k}")

    # --- Evaluation Mode Branching Logic ---
    evaluation_config = training_config.pop("evaluation", {"mode": "default"})
    eval_mode = evaluation_config.get("mode", "default")
    skip_curriculum_reweighting = evaluation_config.get("skip_curriculum_reweighting", False)
    curriculum_reweighting_every_n_steps = evaluation_config.get("curriculum_reweighting_every_n_steps", 1)
    update_intra_dataset_weights = evaluation_config.get("update_intra_dataset_weights", True)
    update_inter_dataset_weights = evaluation_config.get("update_inter_dataset_weights", True)

    if concatenate_train_datasets_without_weighted_sampling:
        assert skip_curriculum_reweighting, "skip_curriculum_reweighting must be True when concatenate_train_datasets_without_weighted_sampling is True."
        assert not update_intra_dataset_weights, "update_intra_dataset_weights must be False when concatenate_train_datasets_without_weighted_sampling is True."
        assert not update_inter_dataset_weights, "update_inter_dataset_weights must be False when concatenate_train_datasets_without_weighted_sampling is True."
    
    if is_main_process:
        logger.info(f"Evaluation Mode: {eval_mode}")
        logger.info(f"Evaluation Config: {evaluation_config}")
        logger.info(f"Skip Curriculum Reweighting: {skip_curriculum_reweighting}")
        logger.info(f"Curriculum Reweighting Every N Steps: {curriculum_reweighting_every_n_steps}")
        logger.info(f"Update Intra-Dataset Weights: {update_intra_dataset_weights}")
        logger.info(f"Update Inter-Dataset Weights: {update_inter_dataset_weights}")

    # These will be the final arguments passed to setup_and_train
    eval_dataset_for_trainer = None
    eval_datasets_for_trainer = None
    eval_dataset_configs_for_trainer = None

    if eval_mode == "custom":
        if is_main_process:
            logger.info("Evaluation Mode: CUSTOM. Using list of validation datasets.")
        # For custom mode, we pass the raw list of datasets and their names
        eval_datasets_for_trainer = val_datasets_to_compose
        eval_dataset_configs_for_trainer = val_dataset_configs

         # --- THE FIX: Create a dummy dataset to satisfy the Trainer's constructor ---
        # We create a tiny subset (just one sample) from the first validation set.
        # We need to pass this to the SFTTrainer, otherwise it will raise an error and we won't be
        # able to run our custom evaluation callback.
        from torch.utils.data import Subset
        eval_dataset_for_trainer = Subset(val_datasets_to_compose[0], range(1))
        if is_main_process:
            logger.info("Created a dummy evaluation dataset of size 1 to satisfy SFTTrainer constructor.")
        # --------------------------------------------------------------------------

    else: # This is the "default" mode
        if is_main_process:
            logger.info("Evaluation Mode: DEFAULT. Using combined validation dataset.")
        # Preserve the original behavior for default mode
        combined_val_dataset = ConcatDataset(val_datasets_to_compose)

        if evaluation_config.get("use_random_subset", False):
            subset_size = evaluation_config.get("subset_size")
            if not subset_size:
                raise ValueError("`subset_size` must be specified when `use_random_subset` is true.")
            per_device_eval_batch_size = training_config["per_device_eval_batch_size"]
            # This is the robust way to get the number of devices before Trainer is initialized.
            # It reads the environment variable set by `torchrun` or `accelerate launch`.
            # It safely defaults to 1 for non-distributed runs.
            num_devices = int(os.environ.get("WORLD_SIZE", "1"))
            if is_main_process:
                logger.info("Using a random subset of validation samples for evaluation.")
                logger.info(f"Using {num_devices} devices for evaluation.")
                logger.info(f"Using {per_device_eval_batch_size} samples per device for evaluation.")
                logger.info(f"Using {subset_size} samples for evaluation.")
                logger.info(f"Ensuring the subset size is a multiple of {per_device_eval_batch_size * num_devices}.")
            final_eval_dataset = RandomSubsetDataset(combined_val_dataset, subset_size,
                                                     ensure_multiple_of=per_device_eval_batch_size * num_devices)
            if is_main_process:
                logger.info(f"Using a random subset of {len(final_eval_dataset)} samples for validation.")
        else:
            final_eval_dataset = combined_val_dataset
            if is_main_process:
                logger.info(f"Using the full validation set with {len(final_eval_dataset)} samples.")
        
        eval_dataset_for_trainer = final_eval_dataset
        eval_dataset_configs_for_trainer = None

    if debug: # --- Debug Mode: Skip Training ---
        return {
            "train_dataset": final_train_dataset,
            "eval_dataset": eval_dataset_for_trainer,
            "eval_datasets": eval_datasets_for_trainer,
            "eval_dataset_configs": eval_dataset_configs_for_trainer,
        }

    # Get the run name from the same place initialize_wandb does.
    wandb_run_name = config.get("tracking", {}).get("wandb", {}).get("run_name")
    if wandb_run_name:
        training_config["run_name"] = wandb_run_name

    # Pop the argument so it's not passed to SFTConfig
    resume_path = training_config.pop("resume_from_checkpoint_path", None)
    pretrained_adapter_path = training_config.pop("pretrained_lora_adapter_path", None)

    logger.info(f"resume_path = {resume_path}")
    logger.info(f"pretrained_adapter_path = {pretrained_adapter_path}")
    
    # --- Get the last checkpoint from the experiment directory ---
    last_checkpoint = get_last_checkpoint(experiment_dir)
    if last_checkpoint is not None:
        if resume_path is not None:
            logger.warning(f"resume_path is set to {resume_path}, but a checkpoint was found at {last_checkpoint}. Ignoring resume_path.")
        else:
            logger.info(f"Found existing checkpoint at {last_checkpoint}. Resuming.")
        resume_path = last_checkpoint

    # Critical: If we are resuming, we are NOT starting from a pretrained adapter.
    if resume_path is not None:
        if pretrained_adapter_path is not None:
            logger.warning(
                "Both resume_path and pretrained_lora_adapter_path are set. "
                "Prioritizing resume_path and ignoring pretrained adapter."
            )
        pretrained_adapter_path = None
        logger.info(f"Resuming training from: {resume_path}")

    # --- Get the flash attention flag from the config ---
    use_flash = model_config.get("use_flash_attention", False)
    
    # --- Main Training Loop ---
    if eval_mode == "custom":
        logger.info(
            "CUSTOM evaluation mode is on. Starting curriculum learning loop."
        )

        total_max_steps = training_config.get("max_steps", -1)
        if total_max_steps <= 0:
            raise ValueError(
                "`max_steps` must be a positive integer for curriculum learning."
            )

        current_step = 0
        
        if resume_path:
            # Determine the starting step if resuming
            state = TrainerState.load_from_json(
                os.path.join(resume_path, "trainer_state.json")
            )
            current_step = state.global_step
            if is_main_process:
                logger.info(f"Resuming. Starting step set to {current_step}.")

            # Load dataset sampling weights if present and apply them to the training dataset if curriculum reweighting is not skipped
            if evaluation_config.get("save_predictions_to_file", False):
                predictions_file_path = os.path.join(experiment_dir, evaluation_config.get(
                    "save_predictions_to_file_name", "predictions_custom_eval.jsonl"))
                if os.path.exists(predictions_file_path):
                    predictions = load_jsonl(predictions_file_path)
                    if is_main_process:
                        logger.info(f"Found {len(predictions)} prediction entries in {predictions_file_path}.")
                    reweighting_entry = None
                    for entry in predictions:
                        if is_main_process:
                            logger.info(
                                f"--- global_step: {entry['global_step']}, "
                                f"timestamp: {entry['timestamp']}, "
                                f"evaluation_results: {len(entry['evaluation_results'])} samples, "
                                f"combined_weights: keys={list(entry['combined_weights'].keys())}"
                            )
                        entry_global_step = entry['global_step']
                        assert entry_global_step <= current_step, f"entry_global_step {entry_global_step} is greater than current_step {current_step}"
                        if (not skip_curriculum_reweighting and
                            entry_global_step % curriculum_reweighting_every_n_steps == 0):
                            if is_main_process:
                                logger.info(" *** Found a past evaluation point where reweighting should be applied. ***")
                            reweighting_entry = entry
                            # Don't break here because we need to find the latest reweighting entry that applies to the current step.
                    if reweighting_entry is not None:
                        if is_main_process:
                            logger.info(f"Applying reweighting from entry with global_step {reweighting_entry['global_step']}.")
                        apply_resampling_weights(final_train_dataset, reweighting_entry['combined_weights'], is_main_process,
                                                 update_intra_dataset_weights=update_intra_dataset_weights,
                                                 update_inter_dataset_weights=update_inter_dataset_weights)
                    else:
                        if is_main_process:
                            logger.warning("No past evaluation point found where reweighting should be applied. "
                                           "Uniform sampling will be used by default.")
                else:
                    if is_main_process:
                        logger.warning("No past evaluation point found where reweighting should be applied. "
                                       "Uniform sampling will be used by default.")
            else:
                if is_main_process:
                    logger.warning("The experiment is not configured to save predictions to file for custom evaluation. "
                                   "This means we won't be able to recover past reweighting weights and apply them to the training dataset, "
                                   "e.g., in case the process was interrupted and is now resumed from a checkpoint.")                        

        while current_step < total_max_steps:
            if is_main_process:
                logger.info(
                    f"\n--- Starting Training Cycle. Current step: {current_step} / {total_max_steps} ---"
                )
                logger.info(f"Current training dataset names: {final_train_dataset.dataset_names}")
                if not concatenate_train_datasets_without_weighted_sampling:
                    logger.info(f"Current training dataset weights: {final_train_dataset.weights}")
                else:
                    logger.info("No training dataset weights are available when concatenate_train_datasets_without_weighted_sampling is True.")
                logger.info(f"Resume path: {resume_path}")
                logger.info(f"Pretrained adapter path: {pretrained_adapter_path}")            
            
            # Set random seed for reproducibility based on the current step
            current_seed = BASE_SEED + current_step
            torch.manual_seed(current_seed)
            torch.cuda.manual_seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)
            random.seed(current_seed)
            np.random.seed(current_seed)
            transformers.set_seed(current_seed)

            # Pass the dynamic seed to the training arguments for the next cycle
            training_config["seed"] = current_seed

            if not concatenate_train_datasets_without_weighted_sampling:
                if is_main_process:
                    logger.info("Shuffling each dataset in the training dataset.")
                # Shuffle each dataset in the training dataset
                for dataset in final_train_dataset.datasets:
                    dataset.shuffle_indices()
            else: # CompositeDataset
                if is_main_process:
                    logger.info("Shuffling the composite training dataset.")
                final_train_dataset.shuffle_indices()

            # Call the setup and training function for one cycle
            resampling_weights, new_step = setup_and_train(
                base_model_id=model_config["base_model_id"],
                output_dir=experiment_dir,
                train_dataset=final_train_dataset,
                is_main_process=is_main_process,
                use_flash_attention=use_flash,
                lora_config=lora_config,
                training_args_dict=training_config,
                evaluation_config=evaluation_config,
                eval_dataset=eval_dataset_for_trainer,
                eval_datasets=eval_datasets_for_trainer,
                eval_dataset_configs=eval_dataset_configs_for_trainer,
                resume_from_checkpoint_path=resume_path,
                pretrained_lora_adapter_path=pretrained_adapter_path,
                diagnostics_config=diagnostics_config,
                save_top_k=save_top_k,
            )

            current_step = new_step

            # Apply the new weights to the datasets on all processes if curriculum reweighting is not skipped
            if not skip_curriculum_reweighting:
                if current_step % curriculum_reweighting_every_n_steps == 0:
                    logger.info(
                        f"[Rank {local_rank}] Applying new dataset weights for the next training cycle."
                    )
                    apply_resampling_weights(final_train_dataset, resampling_weights, is_main_process,
                                             update_intra_dataset_weights=update_intra_dataset_weights,
                                             update_inter_dataset_weights=update_inter_dataset_weights)

            # --- CRITICAL: Clean up GPU and CPU memory ---
            if is_main_process:
                logger.info("Cleaning up GPU and CPU memory before next cycle...")
            gc.collect()
            torch.cuda.empty_cache()
            # ------------------------------------

            # For the next iteration, we will always resume from the latest checkpoint
            resume_path = get_last_checkpoint(experiment_dir)
            if resume_path is None:
                raise RuntimeError(f"Could not find a checkpoint to resume from after step {current_step}")
            # We never use the pretrained adapter again after the first cycle
            pretrained_adapter_path = None

    else:  # Default training mode
        
        # Set random seed for reproducibility based on the base seed
        torch.manual_seed(BASE_SEED)
        torch.cuda.manual_seed(BASE_SEED)
        torch.cuda.manual_seed_all(BASE_SEED)
        random.seed(BASE_SEED)
        np.random.seed(BASE_SEED)
        transformers.set_seed(BASE_SEED)

        # Pass the seed to the training arguments
        training_config["seed"] = BASE_SEED

        # Shuffle the training dataset
        if not concatenate_train_datasets_without_weighted_sampling:
            if is_main_process:
                logger.info("Shuffling each dataset in the training dataset.")
            for dataset in final_train_dataset.datasets:
                dataset.shuffle_indices()
        else: # CompositeDataset
            if is_main_process:
                logger.info("Shuffling the composite training dataset.")
            final_train_dataset.shuffle_indices()

        setup_and_train(
            base_model_id=model_config["base_model_id"],
            output_dir=experiment_dir,
            train_dataset=final_train_dataset,
            is_main_process=is_main_process,
            use_flash_attention=use_flash,
            lora_config=lora_config,
            training_args_dict=training_config,
            evaluation_config=evaluation_config,
            eval_dataset=eval_dataset_for_trainer,
            resume_from_checkpoint_path=resume_path,
            pretrained_lora_adapter_path=pretrained_adapter_path,
            diagnostics_config=diagnostics_config,
            save_top_k=save_top_k,
        )

    # --- 6. Finalize ---
    logger.info("Training finished.")
    # Only finalize wandb on the main process
    if is_main_process:
        finalize_wandb(wandb_run)


if __name__ == "__main__":
    main()
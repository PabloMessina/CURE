import argparse
import json
import logging
import os
import gc
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from vlm_research_kit.settings import EXPERIMENTS_DIR
from vlm_research_kit.data.datasets.mscxr_dataset import (
    MSCXRPhraseGroundingDataset,
)
from vlm_research_kit.data.datasets.padchest_dataset import (
    PadChestGRPhraseGroundingDataset,
)
from vlm_research_kit.data.datasets.vindrcxr_dataset import (
    VinDrCXR_PhraseGroundingDataset,
)
from vlm_research_kit.utils.file_utils import get_safe_filename
from vlm_research_kit.utils.logging_utils import setup_logging
from vlm_research_kit.utils.model_utils import load_medgemma_model, load_maira2_model
from vlm_research_kit.utils.evaluation_utils import prepare_evaluation_batch

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Data Preparation Functions ---


def load_evaluation_dataset(
    dataset_name: str,
    split: str,
    image_transforms_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dataset, str]:
    """
    Loads the specified phrase grounding dataset for evaluation.

    Args:
        dataset_name: The name of the dataset to load ('padchest-gr', 'mscxr', or 'vindrcxr').
        split: The split of the dataset to load ('train', 'val', 'test', 'all').
        image_transforms_kwargs: Optional dictionary of keyword arguments to build an image transformation function.

    Returns:
        A tuple containing the initialized PyTorch Dataset and the unique ID key
        for that dataset used for resuming progress.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_name == "padchest-gr":
        # For PadChest, a unique sample is identified by the study and the phrase
        unique_id_key = ("study_id", "phrase")
        dataset = PadChestGRPhraseGroundingDataset(
            split=split,
            return_image_path=True,
            image_transforms_kwargs=image_transforms_kwargs,
            gt_bbox_format="xyxy",
        )
    elif dataset_name == "mscxr":
        # For MS-CXR, a unique sample is identified by the image and the phrase
        unique_id_key = ("dicom_id", "phrase")
        dataset = MSCXRPhraseGroundingDataset(
            split=split,
            return_image_path=True,
            image_transforms_kwargs=image_transforms_kwargs,
            gt_bbox_format="xyxy",
        )
    elif dataset_name == "vindrcxr":
        # For VinDr-CXR, a unique sample is identified by the image and the phrase
        unique_id_key = ("image_id", "phrase")
        dataset = VinDrCXR_PhraseGroundingDataset(
            split=split,
            return_image_path=True,
            image_transforms_kwargs=image_transforms_kwargs,
            gt_bbox_format="xyxy",
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    logger.info(
        f"Loaded {len(dataset)} samples from the '{dataset_name}' {split} split."
    )
    return dataset, unique_id_key


# --- Model-Specific Functions (Copied and adapted from previous script) ---


def run_medgemma_inference(
    full_dataset: Dataset,
    evaluation_batch_indices: List[int],
    model: torch.nn.Module,
    processor: Any,
    results_jsonl_path: Path,
    max_new_tokens: int,
):
    """Runs phrase grounding inference on a batch using MedGemma."""
    if not evaluation_batch_indices:
        logger.info("Evaluation batch is empty. Nothing to process.")
        return

    logger.info(
        f"Starting MedGemma inference on {len(evaluation_batch_indices)} entries..."
    )
    error_log_path = results_jsonl_path.parent / "errors.jsonl"

    try:
        with open(results_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for iter_count, idx in enumerate(tqdm(evaluation_batch_indices, desc="Grounding with MedGemma")):
                entry = full_dataset[idx]
                
                try:
                    # The dataset provides a PIL image directly
                    pil_image = entry["image"]
                    prompt_text = f"Ground the phrase: {entry['phrase']}"
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil_image},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True,
                    ).to(model.device)

                    with torch.inference_mode():
                        generation = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )

                    decoded_response = processor.decode(
                        generation[0], skip_special_tokens=True
                    )
                    grounding_prediction = decoded_response.split("model\n")[-1].strip()

                    result_data = entry.copy()
                    # Convert image tensor/PIL back to path for JSON serialization
                    if "image" in result_data:
                        del result_data["image"]
                    result_data["prompt"] = prompt_text
                    result_data["grounding_prediction"] = grounding_prediction
                    f_jsonl.write(json.dumps(result_data) + "\n")

                except Exception as e:
                    logger.error(
                        f"Error processing entry for {entry.get('image_path', 'N/A')} (Index {idx}): {e}",
                        exc_info=True,
                    )
                    with open(error_log_path, "a", encoding="utf-8") as f_err:
                        error_entry = entry.copy()
                        if "image" in error_entry:
                            del error_entry["image"]
                        f_err.write(
                            json.dumps(
                                {
                                    "entry": error_entry,
                                    "error": str(e),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                            + "\n"
                        )
                
                # --- Flush to disk every 20 iterations ---
                if (iter_count + 1) % 20 == 0:
                    f_jsonl.flush()
                    os.fsync(f_jsonl.fileno()) # Ensure the file is written to disk

                # --- Call garbage collector to free memory ---
                if (iter_count + 1) % 50 == 0: # Only collect garbage every 50 samples to avoid overhead
                    gc.collect()
                
    except KeyboardInterrupt:
        logger.warning("Inference loop interrupted by user.")
    finally:
        logger.info("Inference loop completed or was interrupted.")


def run_maira2_inference(
    full_dataset: Dataset,
    evaluation_batch_indices: List[int],
    model: torch.nn.Module,
    processor: Any,
    results_jsonl_path: Path,
    max_new_tokens: int,
):
    """Runs phrase grounding inference on a batch using MAIRA-2."""
    if not evaluation_batch_indices:
        logger.info("Evaluation batch is empty. Nothing to process.")
        return

    logger.info(f"Starting MAIRA-2 inference on {len(evaluation_batch_indices)} entries...")
    error_log_path = results_jsonl_path.parent / "errors.jsonl"
    device = model.device

    try:
        with open(results_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for iter_count, idx in enumerate(tqdm(evaluation_batch_indices, desc="Grounding with MAIRA-2")):
                entry = full_dataset[idx]
                
                try:
                    pil_image = entry["image"]
                    location_phrase = entry["phrase"]

                    processed_inputs = (
                        processor.format_and_preprocess_phrase_grounding_input(
                            frontal_image=pil_image,
                            phrase=location_phrase,
                            return_tensors="pt",
                        )
                    )
                    processed_inputs = {
                        k: v.to(device) for k, v in processed_inputs.items()
                    }

                    with torch.no_grad():
                        output_decoding = model.generate(
                            **processed_inputs,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                        )

                    prompt_length = processed_inputs["input_ids"].shape[-1]
                    raw_prediction = processor.decode(
                        output_decoding[0][prompt_length:],
                        skip_special_tokens=True,
                    )

                    grounding_prediction = "PARSING_ERROR"
                    try:
                        grounding_prediction = (
                            processor.convert_output_to_plaintext_or_grounded_sequence(
                                raw_prediction
                            )
                        )
                    except AssertionError:
                        logger.warning(
                            f"AssertionError while parsing prediction for {entry.get('image_path', 'N/A')}. "
                            f"Saving raw output. Raw: '{raw_prediction}'"
                        )

                    result_data = entry.copy()
                    if "image" in result_data:
                        del result_data["image"]
                    result_data["raw_prediction"] = raw_prediction
                    result_data["grounding_prediction"] = grounding_prediction
                    f_jsonl.write(json.dumps(result_data) + "\n")

                except Exception as e:
                    logger.error(
                        f"Error processing entry for {entry.get('image_path', 'N/A')} (Index {idx}): {e}",
                        exc_info=True,
                    )
                    with open(error_log_path, "a", encoding="utf-8") as f_err:
                        error_entry = entry.copy()
                        if "image" in error_entry:
                            del error_entry["image"]
                        f_err.write(
                            json.dumps(
                                {
                                    "entry": error_entry,
                                    "error": str(e),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                            + "\n"
                        )
                
                # --- Flush to disk every 20 iterations ---
                if (iter_count + 1) % 20 == 0:
                    f_jsonl.flush()
                    os.fsync(f_jsonl.fileno()) # Ensure the file is written to disk

                # --- Call garbage collector to free memory ---
                if (iter_count + 1) % 50 == 0: # Only collect garbage every 50 samples to avoid overhead
                    gc.collect()

    except KeyboardInterrupt:
        logger.warning("Inference loop interrupted by user.")
    finally:
        logger.info("Inference loop completed or was interrupted.")


# --- Helper Functions ---


def generate_output_directory(
    model_name: str,
    dataset_name: str,
    split: str,
    base_experiments_dir: str,
    adapter_path: Optional[str] = None,
    revision: Optional[str] = None,
) -> Path:
    """Generates a consistent output directory path for an evaluation run."""
    if adapter_path:
        adapter_path_obj = Path(adapter_path)
        checkpoint_name = adapter_path_obj.name
        training_experiment_dir = adapter_path_obj.parent
        output_dir = (
            training_experiment_dir
            / "evaluations"
            / f"{checkpoint_name}-on-{dataset_name}-{split}"
        )
    else:
        if "maira-2" in model_name and revision:
            run_tag = revision
        else:
            run_tag = "base"

        safe_model_name = get_safe_filename(model_name)
        dir_name = f"{safe_model_name}-{run_tag}-on-{dataset_name}-{split}"
        output_dir = Path(base_experiments_dir) / "evaluations" / dir_name

    return output_dir


# --- Main Execution ---


def main(args: argparse.Namespace):
    """Main function to run the evaluation script."""
    args_str = json.dumps(vars(args), indent=4)
    logger.info(f"--- Evaluation Run Configuration ---\n{args_str}")

    # 1. Setup output directory and paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
        logger.info(f"Using manually specified output directory: {output_dir}")
    else:
        output_dir = generate_output_directory(
            model_name=args.model_name,
            dataset_name=args.dataset,
            split=args.split,
            base_experiments_dir=EXPERIMENTS_DIR,
            adapter_path=args.medgemma_adapter_path,
            revision=args.maira2_revision,
        )
        logger.info(f"Automatically generated output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.image_transforms_kwargs:
        image_transforms_kwargs_str = json.dumps(args.image_transforms_kwargs, indent=4)
        logger.info(f"Using image transforms kwargs: {image_transforms_kwargs_str}")
        # Extract image size and whether image augmentations are used
        image_size = args.image_transforms_kwargs["image_size"]
        logger.info(f"Image size: {image_size}")
        assert type(image_size) is list and len(image_size) == 2, "Image size must be a list of two integers"
        if (not args.image_transforms_kwargs.get("use_model_specific_transforms", False) or
            args.image_transforms_kwargs["model_name"] == "pil_with_augmentations"):
            results_jsonl_path = output_dir / f"predictions_{image_size[0]}x{image_size[1]}_augmented.jsonl"
        else:
            results_jsonl_path = output_dir / f"predictions_{image_size[0]}x{image_size[1]}.jsonl"
    else:
        results_jsonl_path = output_dir / "predictions.jsonl"
    
    logger.info(f"Results will be saved to: {results_jsonl_path}")

    # 2. Load data and prepare the batch for this run
    full_dataset, unique_id_keys = load_evaluation_dataset(
        args.dataset, args.split, image_transforms_kwargs=args.image_transforms_kwargs)
    evaluation_batch_indices = prepare_evaluation_batch(
        full_dataset=full_dataset,
        results_jsonl_path=results_jsonl_path,
        unique_id_keys=unique_id_keys,
        limit=args.limit,
        return_indices=True,
        skip_image_loading=True,
    )

    if not evaluation_batch_indices:
        logger.info("No new entries to process. Evaluation may be complete.")
        return

    # 3. Load model and run inference based on selection
    if args.model_name in ["medgemma", "medgemma-4b-it"]:
        model, processor = load_medgemma_model(
            base_model_id=args.medgemma_base_model_id,
            adapter_path=args.medgemma_adapter_path,
        )
        run_medgemma_inference(
            full_dataset=full_dataset,
            evaluation_batch_indices=evaluation_batch_indices,
            model=model,
            processor=processor,
            results_jsonl_path=results_jsonl_path,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.model_name == "maira-2":
        model, processor = load_maira2_model(
            model_id=args.maira2_model_id, revision=args.maira2_revision
        )
        run_maira2_inference(
            full_dataset=full_dataset,
            evaluation_batch_indices=evaluation_batch_indices,
            model=model,
            processor=processor,
            results_jsonl_path=results_jsonl_path,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    logger.info("Evaluation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Phrase Grounding Evaluation on PadChest-GR or MS-CXR."
    )

    # --- General Arguments ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results. If not provided, a directory will be automatically generated.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["medgemma", "medgemma-4b-it", "maira-2"],
        help="Name of the model to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["padchest-gr", "mscxr", "vindrcxr"],
        help="Name of the dataset to evaluate on.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "validation", "test", "all"],
        help="Split of the dataset to evaluate on.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the evaluation to a specific number of samples (for debugging).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Maximum number of new tokens for the model to generate.",
    )
    parser.add_argument(
        "--image_transforms_kwargs",
        type=str,
        default=None,
        help="Optional dictionary of keyword arguments to build an image transformation function.",
    ) # Example in a terminal: --image_transforms_kwargs '{"use_model_specific_transforms": true, "model_name": "pil_image_only", "image_size": [448, 448], "is_train": false}'

    # --- MedGemma Specific Arguments ---
    parser.add_argument(
        "--medgemma_base_model_id",
        type=str,
        default="google/medgemma-4b-it",
        help="Base model ID for MedGemma.",
    )
    parser.add_argument(
        "--medgemma_adapter_path",
        type=str,
        default=None,
        help="Optional path to the LoRA adapter for a fine-tuned MedGemma model.",
    )

    # --- MAIRA-2 Specific Arguments ---
    parser.add_argument(
        "--maira2_model_id",
        type=str,
        default="microsoft/maira-2",
        help="Model ID for MAIRA-2 from Hugging Face.",
    )
    parser.add_argument(
        "--maira2_revision",
        type=str,
        default=None,
        help="Git revision for the MAIRA-2 model. Defaults to 'main' (latest).",
    )

    # --- Parse and Run ---
    args = parser.parse_args()

    if args.image_transforms_kwargs is not None: # Convert string to dict if necessary
        args.image_transforms_kwargs = json.loads(args.image_transforms_kwargs)

    main(args)
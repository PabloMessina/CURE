import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torchmetrics import Metric

from vlm_research_kit.metrics.metrics_factory import compute_report_generation_metrics
from vlm_research_kit.utils.logging_utils import ANSI_BLACK_BOLD, ANSI_RESET
from vlm_research_kit.utils.text_generation_utils import generate_and_decode_reports
from vlm_research_kit.utils.data_utils import convert_to_serializable
from vlm_research_kit.utils.torch_utils import turn_off_determinism, turn_on_determinism

# Define MetricObject type alias (or import if defined centrally)
MetricObject = Union[Metric, nn.Module, Callable]

logger = logging.getLogger(__name__)

def run_report_generation_evaluation_loop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataloader: DataLoader,
    metrics: Dict[str, MetricObject], # Pre-initialized metrics
    device: Union[str, torch.device],
    device_type: str,
    generation_kwargs: Dict[str, Any],
    model_identifier: Optional[str] = "default", # Optional identifier
    return_predictions: bool = False,
    return_references: bool = False,
    return_image_paths: bool = False,
    compute_per_sample_metrics: bool = False,
    verbose_metric_computation: bool = False,
    use_amp: bool = False,
    print_examples: bool = False,
    cache_generations_csv_path: Optional[Path] = None,
    use_determinism: bool = False,
    handle_grounded_reports: bool = False,
) -> Tuple[Dict[str, Any], Optional[List[str]], Optional[List[str]]]:
    """
    Runs a standard evaluation loop for report generation tasks.

    Iterates through the dataloader, generates predictions, updates/computes
    pre-initialized metrics, and optionally returns all predictions/references.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        dataloader: DataLoader for the evaluation data.
        metrics: Dictionary mapping metric names to their instantiated objects.
                 These objects are expected to be updated/called.
        device: The compute device.
        device_type: The type of device (e.g., "cuda", "cpu").
        generation_kwargs: Dictionary of arguments for text generation.
        model_identifier: Optional string identifier for the model.
        return_predictions: If True, return all generated reports.
        return_references: If True, return all ground truth reports.
        return_image_paths: If True, return image paths (if applicable).
        compute_per_sample_metrics: If True, compute per-sample metrics.
        verbose_metric_computation: If True, print detailed metric computation logs.
        use_amp: If True, use automatic mixed precision for inference.
        print_examples: If True, print examples of predictions and references.
        cache_generations_csv_path: Optional path to cache generated reports.
        use_determinism: If True, enable deterministic behavior for reproducibility.
        handle_grounded_reports: If True, handle grounded reports (if applicable).

    Returns:
        A dictionary containing:
        - overall_metrics: Dictionary of aggregated metrics.
        - per_sample_metrics: Dictionary of per-sample metrics (if compute_per_sample_metrics=True).
        - all_predictions: List of all generated reports (if return_predictions=True).
        - all_references: List of all ground truth reports (if return_references=True).
        - all_image_paths: List of image paths (if return_image_paths=True).
    """
    model.eval()
    logger.info(f"Starting evaluation loop on device {device}...")

    if use_determinism:
        turn_on_determinism() # For reproducibility

    if cache_generations_csv_path is not None and cache_generations_csv_path.exists():
        logger.info(f"Loading cached generations from {cache_generations_csv_path}...")
        df = pd.read_csv(cache_generations_csv_path)
        df = df.fillna('') # Fill NaN values with empty strings
        all_predictions = df["predictions"].tolist()
        all_references = df["references"].tolist()
        assert all(isinstance(pred, str) for pred in all_predictions), "Predictions must be strings."
        assert all(isinstance(ref, str) for ref in all_references), "References must be strings."
        if return_image_paths:
            all_image_paths = df["image_paths"].tolist()
            if all_image_paths[0][0] == '[': # Check if it's a list
                all_image_paths = [ast.literal_eval(path) for path in all_image_paths]
            assert all(isinstance(path, str) or isinstance(path, list) for path in all_image_paths), "Image paths must be strings or lists of strings."
        logger.info("Loaded cached generations successfully.")
        logger.info("Skipping evaluation loop since cached generations are available.")
    else:

        # Accumulators for metrics needing all data / raw outputs if requested
        all_predictions: List[str] = []
        all_references: List[str] = []
        if return_image_paths:
            all_image_paths: List[str] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", ncols=60):
                # --- Data Preparation ---
                pixel_values = batch["pixel_values"].to(device)
                references = batch["reports"] # List[str], keep on CPU

                # Safely get prompts from the batch. It will be None if not present.
                prompts = batch.get("prompts")  # <-- New line

                if return_image_paths:
                    image_paths = batch["image_paths"] # List[str|List[str]], keep on CPU

                # --- Generation ---
                # Mixed Precision Context
                with autocast(device_type=device_type, enabled=use_amp):
                    generated_reports = generate_and_decode_reports(
                        model_identifier=model_identifier,
                        model=model,
                        prompts=prompts,  # <-- Pass prompts to the function
                        pixel_values=pixel_values,
                        generation_kwargs=generation_kwargs,
                        tokenizer=tokenizer,
                        run_model_eval=False, # Already in eval mode
                    )

                # --- Accumulate Outputs ---
                all_predictions.extend(generated_reports)
                all_references.extend(references)
                if return_image_paths:
                    all_image_paths.extend(image_paths)

        if cache_generations_csv_path is not None:
            logger.info(f"Caching generations to {cache_generations_csv_path}...")
            df = pd.DataFrame({
                "predictions": all_predictions,
                "references": all_references,
            })
            if return_image_paths:
                df["image_paths"] = all_image_paths
            df.to_csv(cache_generations_csv_path, index=False)
            logger.info("Cached generations successfully.")

    # --- Remove empty references ---
    empty_reference_idxs = [i for i, ref in enumerate(all_references) if ref == ""]
    if empty_reference_idxs:
        logger.warning(f"Found {len(empty_reference_idxs)} empty references. Removing them from predictions.")
        all_predictions = [p for i, p in enumerate(all_predictions) if i not in empty_reference_idxs]
        all_references = [r for i, r in enumerate(all_references) if i not in empty_reference_idxs]
        if return_image_paths:
            all_image_paths = [p for i, p in enumerate(all_image_paths) if i not in empty_reference_idxs]
     
    # --- Print Examples ---
    if print_examples:
        logger.info("Example predictions and references:")
        # Randomly select two examples to print
        sample_idxs = random.sample(range(len(all_predictions)), min(2, len(all_predictions)))
        for i in sample_idxs:
            logger.info(f"{ANSI_BLACK_BOLD}Prediction {i}{ANSI_RESET}: {all_predictions[i]}")
            logger.info(f"{ANSI_BLACK_BOLD}Reference {i}{ANSI_RESET}: {all_references[i]}")

    # --- Compute Final Metrics ---
    logger.info("Evaluation loop finished. Computing final metrics...")
    out = compute_report_generation_metrics(metrics=metrics,
                                            predictions=all_predictions,
                                            references=all_references,
                                            return_per_sample_metrics=compute_per_sample_metrics,
                                            verbose=verbose_metric_computation,
                                            handle_grounded_reports=handle_grounded_reports)

    if compute_per_sample_metrics:
        overall_metrics, per_sample_metrics = out
    else:
        overall_metrics = out

    # Convert final metrics dict to serializable types (e.g., for YAML saving)
    # This handles tensors, numpy arrays, etc. It also handles the CheXbert dict.
    overall_metrics = convert_to_serializable(overall_metrics)
    if compute_per_sample_metrics:
        per_sample_metrics = convert_to_serializable(per_sample_metrics)

    logger.info("Metrics computed successfully.")

    if use_determinism:
        turn_off_determinism()

    # --- Prepare Output ---
    output = {"overall_metrics": overall_metrics}
    if compute_per_sample_metrics:
        output["per_sample_metrics"] = per_sample_metrics
    if return_predictions:
        output["predictions"] = all_predictions
    if return_references:
        output["references"] = all_references
    if return_image_paths:
        output["image_paths"] = all_image_paths
    return output

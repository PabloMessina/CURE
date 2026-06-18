import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from vlm_research_kit.utils.file_utils import get_safe_filename, load_jsonl

logger = logging.getLogger(__name__)


def generate_output_directory(
    model_name: str,
    dataset_name: str,
    split: str,
    base_experiments_dir: str,
    adapter_path: Optional[str] = None,
    revision: Optional[str] = None,
    task_suffix: Optional[str] = None,
) -> Path:
    """
    Generates a consistent output directory path for an evaluation run.

    Args:
        model_name: The name of the model being evaluated.
        dataset_name: The name of the dataset.
        split: The dataset split (e.g., 'test').
        base_experiments_dir: The root directory for all experiments.
        adapter_path: Path to a LoRA adapter, if used.
        revision: Git revision for the model, if applicable.
        task_suffix: An optional suffix to append to the directory name
                     to distinguish between tasks (e.g., 'reportgen').

    Returns:
        A Path object for the generated output directory.
    """
    suffix = f"-{task_suffix}" if task_suffix else ""

    if adapter_path:
        adapter_path_obj = Path(adapter_path)
        checkpoint_name = adapter_path_obj.name
        training_experiment_dir = adapter_path_obj.parent
        dir_name = (
            f"{checkpoint_name}-on-{dataset_name}-{split}{suffix}"
        )
        output_dir = training_experiment_dir / "evaluations" / dir_name
    else:
        if "maira-2" in model_name and revision:
            run_tag = revision
        else:
            run_tag = "base"

        safe_model_name = get_safe_filename(model_name)
        dir_name = f"{safe_model_name}-{run_tag}-on-{dataset_name}-{split}{suffix}"
        output_dir = Path(base_experiments_dir) / "evaluations" / dir_name

    return output_dir


def prepare_evaluation_batch(
    full_dataset: Dataset,
    results_jsonl_path: Path,
    unique_id_keys: Union[str, Tuple[str, ...]],
    limit: Optional[int] = None,
    return_indices: bool = False,
    skip_image_loading: bool = False,
) -> Union[List[Dict[str, Any]], List[int]]:
    """
    Prepares the batch of items to be processed, skipping already completed ones.

    Args:
        full_dataset: The complete dataset for the evaluation split.
        results_jsonl_path: Path to the file where results are stored.
        unique_id_keys: A key or tuple of keys that uniquely identify a sample
                        (e.g., 'image_path' or ('study_id', 'phrase')).
        limit: Optional integer to limit the number of samples to process.
        return_indices: Whether to return the indices of the samples to be processed.
        skip_image_loading: Whether to skip loading the images.

    Returns:
        A list of dataset entries to be processed in this run.
    """
    processed_ids = set()
    if results_jsonl_path.is_file():
        logger.info(f"Resume file found at: {results_jsonl_path}. Loading...")
        processed_entries = load_jsonl(results_jsonl_path)

        if isinstance(unique_id_keys, str):
            processed_ids = {entry[unique_id_keys] for entry in processed_entries}
        else: # It's a tuple of keys
            processed_ids = {
                tuple(entry[key] for key in unique_id_keys)
                for entry in processed_entries
            }
        logger.info(f"Loaded {len(processed_ids):,} already processed entries.")
    else:
        logger.info("No resume file found. Starting a fresh evaluation.")

    unprocessed_entries = []
    for i in range(len(full_dataset)):
        if skip_image_loading:
            entry = full_dataset.__getitem__(i, skip_image_loading=True) # type: ignore
        else:
            entry = full_dataset[i]
        if isinstance(unique_id_keys, str):
            entry_id = entry[unique_id_keys]
        else:
            entry_id = tuple(entry[key] for key in unique_id_keys)

        if entry_id not in processed_ids:
            if return_indices:
                unprocessed_entries.append(i)
            else:
                unprocessed_entries.append(entry)

    logger.info(
        f"Found {len(unprocessed_entries):,} unprocessed entries for this run."
    )

    if limit is not None and limit > 0:
        logger.info(f"Limiting evaluation to the first {limit} unprocessed entries.")
        return unprocessed_entries[:limit]

    return unprocessed_entries


def compute_bootstrapped_instance_metrics(
    metrics: np.ndarray, 
    categories: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000, 
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes bootstrap statistics for precomputed instance-level metrics.
    Optionally stratified by category to compute macro-averages and per-category stats.
    
    Args:
        metrics: 1D NumPy array of precomputed metric values.
        categories: Optional 1D NumPy array of category labels (strings or ints).
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        
    Returns:
        A dictionary containing "micro", "macro" (if categories provided), 
        and "categories" stats (means and standard deviations).
    """
    rng = np.random.default_rng(seed)
    n_samples = len(metrics)
    
    # --- Standard approach (No Categories) ---
    if categories is None:
        indices = rng.integers(0, n_samples, size=(n_bootstrap, n_samples))
        boot_means = np.mean(metrics[indices], axis=1)
        return {
            "micro": {"mean": float(np.mean(boot_means)), "std": float(np.std(boot_means))}
        }
        
    # --- Stratified approach (With Categories) ---
    unique_cats = np.unique(categories)
    n_cats = len(unique_cats)
    
    # Map categories to integers for fast np.bincount array operations
    cat_to_int = {cat: idx for idx, cat in enumerate(unique_cats)}
    categories_int = np.array([cat_to_int[cat] for cat in categories])
    
    # Find indices for each category to guarantee inclusion
    cat_to_indices = [np.where(categories_int == c_idx)[0] for c_idx in range(n_cats)]
    
    micro_scores = np.empty(n_bootstrap)
    macro_scores = np.empty(n_bootstrap)
    cat_scores = {cat: np.empty(n_bootstrap) for cat in unique_cats}
    
    for i in range(n_bootstrap):
        # 1. Guarantee at least one instance per category
        guaranteed_indices = [rng.choice(idxs) for idxs in cat_to_indices]
        
        # 2. Fill the rest randomly
        remaining_needed = n_samples - n_cats
        random_indices = rng.integers(0, n_samples, size=remaining_needed)
        
        # 3. Combine to form the bootstrap sample
        sample_indices = np.concatenate([guaranteed_indices, random_indices])
        sample_metrics = metrics[sample_indices]
        sample_cats_int = categories_int[sample_indices]
        
        # Micro Average
        micro_scores[i] = np.mean(sample_metrics)
        
        # Per-Category Average (Vectorized summing and counting)
        sums = np.bincount(sample_cats_int, weights=sample_metrics, minlength=n_cats)
        counts = np.bincount(sample_cats_int, minlength=n_cats)
        
        # Division is safe because we guaranteed at least 1 count per class
        cat_means = sums / counts 
        
        # Macro Average
        macro_scores[i] = np.mean(cat_means)
        
        # Store individual category means for this iteration
        for cat_idx, cat_name in enumerate(unique_cats):
            cat_scores[cat_name][i] = cat_means[cat_idx]

    # Aggregate final statistics
    return {
        "micro": {"mean": float(np.mean(micro_scores)), "std": float(np.std(micro_scores))},
        "macro": {"mean": float(np.mean(macro_scores)), "std": float(np.std(macro_scores))},
        "categories": {
            cat: {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
            for cat, scores in cat_scores.items()
        }
    }


def compute_bootstrapped_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ensure_all_classes: bool = True,
    drop_empty_classes: bool = False,
    seed: int = 42
) -> Tuple[float, float, List[float]]:
    """
    Computes bootstrap statistics for multi-label tasks.
    
    Args:
        drop_empty_classes: If True (for Macro), slices out classes with zero 
                            ground-truth positives so they are completely ignored.
                            If False (for Micro), keeps them so False Positives penalize the score.
    """
    rng = np.random.default_rng(seed)
    
    # --- 1. Handle Empty Classes based on the flag ---
    if drop_empty_classes:
        active_classes = np.where(y_true.sum(axis=0) > 0)[0]
        if len(active_classes) == 0:
            raise ValueError("Cannot bootstrap: No positive examples exist in the ground truth.")
        y_true_working = y_true[:, active_classes]
        y_pred_working = y_pred[:, active_classes]
    else:
        y_true_working = y_true
        y_pred_working = y_pred

    n_samples = y_true_working.shape[0]
    
    # --- 2. Pre-compute valid indices pools ---
    if ensure_all_classes:
        n_classes = y_true_working.shape[1]
        pos_indices_per_class = []
        for c in range(n_classes):
            idx = np.where(y_true_working[:, c] > 0)[0]
            # Permissive check in case we didn't drop empty classes
            if len(idx) > 0:
                pos_indices_per_class.append(idx)
    else:
        any_pos_idx = np.where(y_true_working.sum(axis=1) > 0)[0]
        if len(any_pos_idx) == 0:
            raise ValueError("Cannot bootstrap: No positive examples exist in the ground truth.")

    bootstrapped_scores = []
    
    for _ in range(n_bootstrap):
        guaranteed_indices = []
        
        # 3. Seed the sample with required instances
        if ensure_all_classes:
            for c_idx in pos_indices_per_class:
                guaranteed_indices.append(rng.choice(c_idx))
        else:
            guaranteed_indices.append(rng.choice(any_pos_idx))
            
        guaranteed_indices = list(set(guaranteed_indices)) # Remove duplicates
        guaranteed_indices = np.array(guaranteed_indices)
        n_guaranteed = len(guaranteed_indices)
        
        # 4. Fill the remainder of the sample
        remaining_needed = n_samples - n_guaranteed
        random_indices = rng.integers(0, n_samples, size=remaining_needed)
        
        # 5. Combine and compute
        sample_indices = np.concatenate([guaranteed_indices, random_indices])
        
        # The metric function now receives either the full arrays (Micro) or the sliced active arrays (Macro)
        score = metric_fn(y_true_working[sample_indices], y_pred_working[sample_indices])
        bootstrapped_scores.append(score)
        
    return float(np.mean(bootstrapped_scores)), float(np.std(bootstrapped_scores)), bootstrapped_scores
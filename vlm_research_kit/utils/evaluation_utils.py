import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
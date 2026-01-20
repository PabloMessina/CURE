import argparse
import json
import logging
import math
import os
import random
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from vlm_research_kit.data.datasets.chest_imagenome_dataset import (
    SUPPORTED_DESCRIBE_LOCATIONS,
    SUPPORTED_LOCATE_AND_DESCRIBE_LOCATIONS,
    SUPPORTED_LOCATE_LOCATIONS,
)
from vlm_research_kit.data.datasets.mimiccxr_dataset import (
    get_dicom_id_to_image_path_map,
)
from vlm_research_kit.data.transforms_factory import create_image_transforms
from vlm_research_kit.settings import (
    CHEST_IMAGENOME_LOCATION_REPORT_SNIPPETS_PATH,
    EXPERIMENTS_DIR,
    GEMINI_2_5_FLASH_LITE_ANNOTATED_MINI_REPORTS_JSONL_PATH,
    GEMINI_2_5_FLASH_LITE_MIMIC_CXR_ANATOMY_SPECIFIC_REPORTS_JSONL_PATH,
    MIMIC_CXR_SPLIT_CSV_PATH,
)
from vlm_research_kit.utils.file_utils import (
    get_safe_filename,
    load_jsonl,
    load_pickle,
    save_jsonl,
)
from vlm_research_kit.utils.logging_utils import setup_logging
from vlm_research_kit.utils.model_utils import (
    load_maira2_model,
    load_medgemma_model,
)

# --- Constants ---
DEFAULT_ANATOMIES = [
    "right lung",
    "left lung",
    "left costophrenic angle",
    "right costophrenic angle",
    "spine",
    "right clavicle",
    "left clavicle",
    "mediastinum",
    "cardiac silhouette",
]
CORE_SAMPLE_FILENAME = "chest_imagenome_core_evaluation_sample.jsonl"

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Data Preparation Functions ---

def clean_report_text(report: str) -> str:
    """Removes extra whitespace from a report string."""
    return " ".join(report.strip().split())


def load_dependencies() -> Dict[str, Any]:
    """Loads all necessary data files and returns them in a dictionary."""
    logger.info("Loading data dependencies...")
    location_report_snippets = load_pickle(CHEST_IMAGENOME_LOCATION_REPORT_SNIPPETS_PATH)
    split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV_PATH)
    gemini_annotations = load_jsonl(GEMINI_2_5_FLASH_LITE_MIMIC_CXR_ANATOMY_SPECIFIC_REPORTS_JSONL_PATH)
    mini_report_annotations_raw = load_jsonl(GEMINI_2_5_FLASH_LITE_ANNOTATED_MINI_REPORTS_JSONL_PATH)

    query2response = {
        x["metadata"]["query"]: x["parsed_response"] for x in gemini_annotations
    }
    mini_report2annotations = {
        x["metadata"]["query"]: x["parsed_response"]
        for x in mini_report_annotations_raw
    }
    dicom_id_to_image_path_map = get_dicom_id_to_image_path_map()

    logger.info("Data dependencies loaded successfully.")
    return {
        "location_report_snippets": location_report_snippets,
        "split_df": split_df,
        "query2response": query2response,
        "mini_report2annotations": mini_report2annotations,
        "dicom_id_to_image_path_map": dicom_id_to_image_path_map,
    }


def build_full_test_set(
    location_report_snippets: List[Dict],
    split_df: pd.DataFrame,
    dicom_id_to_image_path_map: Dict,
    query2response: Dict,
    mini_report2annotations: Dict,
) -> List[Dict[str, Any]]:
    """
    Builds a comprehensive list of all possible evaluation entries from the test set.
    """
    logger.info("Building the full test set from source data...")
    test_dicom_ids = set(
        split_df[split_df["split"] == "test"]["dicom_id"].tolist()
    )
    logger.info(f"Found {len(test_dicom_ids)} unique DICOM IDs in the test set.")

    location_entries = []
    skipped_entries = 0
    for entry in tqdm(location_report_snippets, desc="Processing report snippets"):
        dicom_id = entry["dicom_id"]
        if dicom_id not in test_dicom_ids:
            continue

        locations = (
            set(entry["location2bbox"].keys())
            | set(entry["location2report_snippet"].keys())
            | set(DEFAULT_ANATOMIES)
        )
        report_text = clean_report_text(entry["original_report"])
        image_path = dicom_id_to_image_path_map.get(dicom_id)
        if not image_path:
            continue

        for location in locations:
            bbox = entry["location2bbox"].get(location, None)
            query = f'Report: "{report_text}" | Location: "{location}"'
            mini_report, mini_report_reasoning, mini_report_annotations = None, None, None

            if query in query2response:
                mini_report_resp = query2response[query]["mini-report"]
                if mini_report_resp != "N/A":
                    mini_report = mini_report_resp
                    mini_report_reasoning = query2response[query]["reasoning"]
                    mini_report_annotations = mini_report2annotations.get(mini_report)

            if mini_report is not None or bbox is not None:
                location_entries.append(
                    {
                        "dicom_id": dicom_id,
                        "image_path": image_path,
                        "location": location,
                        "bbox": bbox,
                        "mini_report": mini_report,
                        "mini_report_reasoning": mini_report_reasoning,
                        "mini_report_annotations": mini_report_annotations,
                    }
                )
            else:
                skipped_entries += 1

    logger.info(
        f"Built a total of {len(location_entries):,} possible evaluation entries."
    )
    if skipped_entries > 0:
        logger.warning(f"Skipped {skipped_entries} entries due to missing mini-report and bbox.")
    
    return location_entries


def _generate_and_cache_core_sample(
    unprocessed_entries: List[Dict],
    sample_size: int,
    cache_path: Path,
    random_state: int = 42,
) -> List[Dict]:
    """
    Generates a stratified, balanced sample and saves it to a cache file.
    This function contains the core sampling logic.
    """
    logger.info(
        f"Generating a new core sample of size {sample_size} and caching to {cache_path}"
    )
    df = pd.DataFrame(unprocessed_entries)
    df["has_mini_report"] = df["mini_report"].notna()

    def get_abnormality(annotation):
        return (
            annotation.get("mentions_abnormalities")
            if isinstance(annotation, dict)
            else None
        )

    def get_device(annotation):
        return (
            annotation.get("mentions_devices")
            if isinstance(annotation, dict)
            else None
        )

    df["abnormality"] = df["mini_report_annotations"].apply(get_abnormality)
    df["devices"] = df["mini_report_annotations"].apply(get_device)
    df_with_report = df[df["has_mini_report"]].copy()
    df_without_report = df[~df["has_mini_report"]].copy()

    # Stratified sampling logic (70% with report, 30% without)
    n_with_report = min(int(sample_size * 0.70), len(df_with_report))
    n_without_report = min(sample_size - n_with_report, len(df_without_report))

    random.seed(random_state) # Set the random seed for reproducibility

    sampled_with_report_list = []
    if not df_with_report.empty and n_with_report > 0:
        groupby = df_with_report.groupby(["location", "abnormality", "devices"], dropna=False)
        n_groups = len(groupby)
        n_samples_per_stratum = int(math.ceil(n_with_report / n_groups))

        for key, group in groupby:
            # Determine how many to take
            n = min(len(group), n_samples_per_stratum)
            if n == 0:
                continue # Skip if no samples are available for this group
            # Sample and convert to list of dicts
            sample_dicts = group.sample(n=n, random_state=random_state).to_dict("records")
            # Add to the list
            sampled_with_report_list.extend(sample_dicts)
            
        if len(sampled_with_report_list) > n_with_report: # If we have more than we need, sample without replacement
            sampled_with_report_list = random.sample(sampled_with_report_list, n_with_report)
        elif len(sampled_with_report_list) < n_with_report: # If we have less than we need, sample from the remaining pool
            remaining_needed = n_with_report - len(sampled_with_report_list)
            used_dicom_locations = set([(entry["dicom_id"], entry["location"]) for entry in sampled_with_report_list])
            assert len(used_dicom_locations) == len(sampled_with_report_list) # This should be true since we only have one entry for each dicom ID and location
            remaining_pool = [entry for entry in df_with_report.to_dict("records") if (entry["dicom_id"], entry["location"]) not in used_dicom_locations]
            n_to_sample_from_pool = min(remaining_needed, len(remaining_pool))
            sampled_with_report_list.extend(random.sample(remaining_pool, n_to_sample_from_pool))
        else:
            pass # We have the correct number of samples

    sampled_without_report_list = []
    if not df_without_report.empty and n_without_report > 0:
        groupby = df_without_report.groupby(["location"], dropna=False)
        n_groups = len(groupby)
        n_samples_per_stratum = int(math.ceil(n_without_report / n_groups))

        for key, group in groupby:
            # Determine how many to take
            n = min(len(group), n_samples_per_stratum)
            if n == 0:
                continue # Skip if no samples are available for this group
            # Sample and convert to list of dicts
            sample_dicts = group.sample(n=n, random_state=random_state).to_dict("records")
            # Add to the list
            sampled_without_report_list.extend(sample_dicts)
            
        if len(sampled_without_report_list) > n_without_report: # If we have more than we need, sample without replacement
            sampled_without_report_list = random.sample(sampled_without_report_list, n_without_report)
        elif len(sampled_without_report_list) < n_without_report: # If we have less than we need, sample from the remaining pool
            remaining_needed = n_without_report - len(sampled_without_report_list)
            used_dicom_locations = set([(entry["dicom_id"], entry["location"]) for entry in sampled_without_report_list])
            assert len(used_dicom_locations) == len(sampled_without_report_list) # This should be true since we only have one entry for each dicom ID and location
            remaining_pool = [entry for entry in df_without_report.to_dict("records") if (entry["dicom_id"], entry["location"]) not in used_dicom_locations]
            n_to_sample_from_pool = min(remaining_needed, len(remaining_pool))
            sampled_without_report_list.extend(random.sample(remaining_pool, n_to_sample_from_pool))
        else:
            pass # We have the correct number of samples

    core_sample = sampled_with_report_list + sampled_without_report_list
    logger.info(f"Generated core sample with {len(core_sample)} entries.")
    save_jsonl(core_sample, cache_path)
    return core_sample


def prepare_evaluation_sample(
    all_entries: List[Dict[str, Any]],
    results_jsonl_path: Path,
    core_sample_path: Path,
    target_sample_size: int,
    core_sample_size: int = 1000,
    force_regenerate_core_sample: bool = False,
    random_state: int = 42,
    skip_sampling_beyond_core_sample: bool = False,
) -> List[Dict[str, Any]]:
    """
    Prepares a reproducible and extensible sample for evaluation.

    This function ensures a common "core" sample is used across runs for fair
    comparison, and then adds more random samples if requested. It also
    handles resuming from previous runs.
    """
    # 1. Handle resuming from previous results
    processed_ids = set()
    if results_jsonl_path.is_file():
        logger.info(f"Resume file found at: {results_jsonl_path}. Loading...")
        processed_entries = load_jsonl(results_jsonl_path)
        processed_ids = {
            (entry["dicom_id"], entry["location"]) for entry in processed_entries
        }
        logger.info(f"Loaded {len(processed_ids):,} already processed entries.")
    else:
        logger.info("No resume file found. Starting a fresh evaluation.")

    # 2. Prepare the core sample (generate if it doesn't exist)
    if not core_sample_path.is_file() or force_regenerate_core_sample:
        core_sample = _generate_and_cache_core_sample(
            all_entries, core_sample_size, core_sample_path, random_state
        )
    else:
        logger.info(f"Loading core sample from cache: {core_sample_path}")
        core_sample = load_jsonl(core_sample_path)

    # 3. Determine the evaluation batch for this run
    # Start with the core samples that haven't been processed yet
    unprocessed_core_samples = [
        entry
        for entry in core_sample
        if (entry["dicom_id"], entry["location"]) not in processed_ids
    ]
    evaluation_batch = unprocessed_core_samples
    logger.info(
        f"Found {len(unprocessed_core_samples)} unprocessed entries from the core sample."
    )

    if skip_sampling_beyond_core_sample:
        logger.info("Skipping sampling beyond the core sample. Returning the core sample only.")
        return evaluation_batch

    # 4. Add more samples if the target size is larger than the available core samples
    remaining_needed = target_sample_size - len(evaluation_batch)
    if remaining_needed > 0:
        logger.info(
            f"Target sample size ({target_sample_size}) is larger than available core samples. "
            f"Need to sample {remaining_needed} additional entries."
        )
        core_sample_ids = {
            (entry["dicom_id"], entry["location"]) for entry in core_sample
        }
        # Create a pool of entries that are not in the core set and not processed
        additional_pool = [
            entry
            for entry in all_entries
            if (entry["dicom_id"], entry["location"]) not in core_sample_ids
            and (entry["dicom_id"], entry["location"]) not in processed_ids
        ]

        if len(additional_pool) > 0:
            n_to_sample = min(remaining_needed, len(additional_pool))
            logger.info(f"Sampling {n_to_sample} from a pool of {len(additional_pool)}.")
            additional_df = pd.DataFrame(additional_pool)
            additional_samples = additional_df.sample(
                n=n_to_sample, random_state=random_state
            ).to_dict("records")
            evaluation_batch.extend(additional_samples)
        else:
            logger.warning("No additional unprocessed entries available to sample from.")

    final_batch = evaluation_batch[:target_sample_size]
    logger.info(f"Final evaluation batch size: {len(final_batch):,}")
    return final_batch


def print_location_statistics(
    location_df: pd.DataFrame, location_name: str
):
    """Calculates and prints detailed statistics for a filtered location dataset."""
    total_instances = len(location_df)
    if total_instances == 0:
        logger.info(f"No instances found for location: '{location_name}'.")
        return

    stats = {
        "Total Instances": total_instances,
        "With Grounding (BBox)": location_df["has_bbox"].sum(),
        "Without Grounding (BBox)": (~location_df["has_bbox"]).sum(),
        "With Mini-Report": location_df["has_mini_report"].sum(),
        "Without Mini-Report": (~location_df["has_mini_report"]).sum(),
        "Mentions Abnormalities": location_df["abnormality"].eq("yes").sum(),
        "No Abnormality Mentioned": location_df["abnormality"].eq("no").sum(),
        "Mentions Devices": location_df["devices"].eq("yes").sum(),
        "No Device Mentioned": location_df["devices"].eq("no").sum(),
        "Missing Both Report & BBox": (
            (~location_df["has_mini_report"]) & (~location_df["has_bbox"])
        ).sum(),
    }

    logger.info(f"--- Statistics for Anatomical Location: '{location_name}' ---")
    for key, value in stats.items():
        percentage = (value / total_instances) * 100
        logger.info(f"- {key}: {value:,} ({percentage:.2f}%)")
    logger.info("---------------------------------------------------------")


def _generate_and_cache_location_core_sample(
    location_entries: List[Dict[str, Any]],
    location_name: str,
    cache_path: Path,
    core_sample_size: int,
    all_entries_df: pd.DataFrame,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generates a balanced, reproducible core sample for a location and saves it to a shared cache.
    This function contains the core sampling logic and should only run once.
    """
    logger.info(
        f"Generating new core sample (size: {core_sample_size}) for '{location_name}'..."
    )
    df = pd.DataFrame(location_entries)

    # Pre-process to add annotation columns
    df["has_bbox"] = df["bbox"].notna()
    df["has_mini_report"] = df["mini_report"].notna()

    def get_annotation_value(annotation, key):
        return (
            annotation.get(key, "unknown")
            if isinstance(annotation, dict)
            else "unknown"
        )

    df["abnormality"] = df["mini_report_annotations"].apply(
        lambda x: get_annotation_value(x, "mentions_abnormalities")
    )
    df["devices"] = df["mini_report_annotations"].apply(
        lambda x: get_annotation_value(x, "mentions_devices")
    )

    print_location_statistics(df, location_name)

    initial_count = len(df)
    df = df[df["has_mini_report"] | df["has_bbox"]].copy()
    if len(df) < initial_count:
        logger.info(
            f"Filtered out {initial_count - len(df)} entries that had neither a mini-report nor a bounding box."
        )

    if df.empty:
        logger.warning(
            f"No viable entries left for '{location_name}' after filtering. Cannot create sample."
        )
        return []

    df["priority_score"] = (
        df["has_bbox"].astype(int) + df["has_mini_report"].astype(int)
    )
    df = df.sort_values(by="priority_score", ascending=False)

    strata = ["abnormality", "devices"]
    df_stratify = df[
        df["abnormality"].isin(["yes", "no"]) & df["devices"].isin(["yes", "no"])
    ].copy()

    # Balanced sampling logic
    n_samples_per_stratum = core_sample_size // 4
    sampled_dfs = []
    if not df_stratify.empty:
        for _, group in df_stratify.groupby(strata):
            n_to_sample = min(n_samples_per_stratum, len(group))
            if n_to_sample > 0:
                sampled_dfs.append(
                    group.sample(n=n_to_sample, random_state=random_state)
                )

    final_sample_df = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()

    remaining_needed = core_sample_size - len(final_sample_df)
    if remaining_needed > 0 and not df.empty:
        # Avoid re-sampling items already selected
        remaining_pool = df.drop(final_sample_df.index) if not final_sample_df.empty else df
        if not remaining_pool.empty:
            n_to_sample_from_pool = min(remaining_needed, len(remaining_pool))
            additional_sample = remaining_pool.sample(
                n=n_to_sample_from_pool, random_state=random_state
            )
            final_sample_df = pd.concat([final_sample_df, additional_sample])

    remaining_needed = core_sample_size - len(final_sample_df)
    if remaining_needed > 0:
        logger.info(f"Need {remaining_needed} additional samples to reach the core sample size.")
        # Remove the dicom IDs that are already in the final sample
        dicom_ids_to_skip = set(final_sample_df["dicom_id"])
        logger.info(f"len(dicom_ids_to_skip): {len(dicom_ids_to_skip)}")
        logger.info(f"len(all_entries_df): {len(all_entries_df)}")
        additional_pool = all_entries_df[~all_entries_df["dicom_id"].isin(dicom_ids_to_skip)]
        logger.info(f"len(additional_pool): {len(additional_pool)}")
        # Remove rows that don't have both a mini-report and a bounding box
        additional_pool = additional_pool[additional_pool["mini_report"].notna() & additional_pool["bbox"].notna()]
        logger.info(f"len(additional_pool): {len(additional_pool)} (after filtering out rows that don't have both a mini-report and a bounding box)")
        # Keep the first row for each dicom ID
        additional_pool = additional_pool.groupby("dicom_id").first().reset_index()
        logger.info(f"len(additional_pool): {len(additional_pool)} (after keeping the first row for each dicom ID)")
        if not additional_pool.empty:
            n_to_sample = min(remaining_needed, len(additional_pool))
            additional_sample = additional_pool.sample(
                n=n_to_sample, random_state=random_state
            )
            final_sample_df = pd.concat([final_sample_df, additional_sample])

    core_sample = final_sample_df.to_dict("records")
    logger.info(
        f"Generated core sample of size {len(core_sample)} for '{location_name}'."
    )
    save_jsonl(core_sample, cache_path)
    logger.info(f"Saved core sample to {cache_path}")
    return core_sample


def prepare_location_evaluation_batch(
    all_location_entries: List[Dict[str, Any]],
    results_jsonl_path: Path,
    core_sample: List[Dict[str, Any]],
    target_eval_size: int,
    random_state: int = 42,
    skip_sampling_beyond_core_sample: bool = False,
) -> List[Dict[str, Any]]:
    """
    Prepares the final evaluation batch for a specific run, using the canonical
    core sample as a base and adding new samples if needed.
    """
    processed_ids = set()
    if results_jsonl_path.is_file():
        processed_entries = load_jsonl(results_jsonl_path)
        processed_ids = {
            (e["dicom_id"], e["location"]) for e in processed_entries
        }

    # Start with the core samples that haven't been processed yet for this experiment
    unprocessed_core_samples = [
        entry
        for entry in core_sample
        if (entry["dicom_id"], entry["location"]) not in processed_ids
    ]
    evaluation_batch = unprocessed_core_samples
    
    # If we have enough unprocessed core samples, just take a subset
    if len(evaluation_batch) >= target_eval_size:
        df_batch = pd.DataFrame(evaluation_batch)
        final_batch = df_batch.sample(n=target_eval_size, random_state=random_state).to_dict("records")
        logger.info(f"Selected {len(final_batch)} samples from the unprocessed core set for this run.")
        return final_batch

    if skip_sampling_beyond_core_sample:
        logger.info("Skipping sampling beyond the core sample. Returning the core sample only.")
        return evaluation_batch

    # If not enough, take all unprocessed core samples and find more
    remaining_needed = target_eval_size - len(evaluation_batch)
    if remaining_needed > 0:
        logger.info(
            f"Need {remaining_needed} additional samples beyond the unprocessed core set."
        )
        core_sample_ids = {
            (e["dicom_id"], e["location"]) for e in core_sample
        }
        
        # Pool of entries not in the core set and not already processed
        additional_pool = [
            entry
            for entry in all_location_entries
            if (entry["dicom_id"], entry["location"]) not in core_sample_ids
            and (entry["dicom_id"], entry["location"]) not in processed_ids
        ]

        if additional_pool:
            n_to_sample = min(remaining_needed, len(additional_pool))
            additional_df = pd.DataFrame(additional_pool)
            additional_samples = additional_df.sample(
                n=n_to_sample, random_state=random_state
            ).to_dict("records")
            evaluation_batch.extend(additional_samples)
        else:
            logger.warning("No additional unprocessed entries available to sample from.")

    final_batch = evaluation_batch[:target_eval_size]
    logger.info(f"Final evaluation batch size for this run: {len(final_batch):,}")
    return final_batch


# --- Model-Specific Functions ---


def run_medgemma_inference(
    evaluation_batch: List[Dict[str, Any]],
    model: torch.nn.Module,
    processor: Any,
    results_jsonl_path: Path,
    max_new_tokens: int,
    image_transforms: Optional[Callable] = None,
    use_localize_instead_of_locate: bool = False,
):
    """Runs phrase grounding inference on a batch using MedGemma."""
    if not evaluation_batch:
        logger.info("Evaluation batch is empty. Nothing to process.")
        return

    logger.info(
        f"Starting MedGemma inference on {len(evaluation_batch)} entries..."
    )
    error_log_path = results_jsonl_path.parent / "errors.jsonl"

    try:
        with open(results_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for idx, entry in enumerate(
                tqdm(evaluation_batch, desc="Grounding with MedGemma")
            ):
                try:
                    # --- Image Loading & Transformation ---
                    if image_transforms:
                        # Use the provided transform function
                        image_input = image_transforms(entry["image_path"])
                        if isinstance(image_input, dict):
                            image_input = image_input["pixel_values"]
                    else:
                        # Default behavior: load and convert to RGB
                        image_input = Image.open(entry["image_path"]).convert("RGB")                    
                    
                    location_phrase = entry["location"]

                    # --- Check if the entry has ground truth ---
                    has_bbox = entry.get("bbox") is not None
                    has_description = entry.get("mini_report") is not None
                    assert has_bbox or has_description, "No ground truth available for the entry."
                    
                    # --- Dynamically generate the prompt based on the location type ---
                    if location_phrase in SUPPORTED_LOCATE_AND_DESCRIBE_LOCATIONS:
                        prompt_text = (
                            f"Locate and describe the {location_phrase}."
                            if not use_localize_instead_of_locate
                            else f"Localize and describe the {location_phrase}."
                        )
                    elif location_phrase in SUPPORTED_DESCRIBE_LOCATIONS:
                        prompt_text = f"Describe the {location_phrase}."
                    elif location_phrase in SUPPORTED_LOCATE_LOCATIONS:
                        prompt_text = (
                            f"Locate the {location_phrase}."
                            if not use_localize_instead_of_locate
                            else f"Localize the {location_phrase}."
                        )
                    else:
                        raise ValueError(f"Unknown location: {location_phrase}")

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_input},
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
                    result_data["prompt"] = prompt_text
                    result_data["grounding_prediction"] = grounding_prediction
                    f_jsonl.write(json.dumps(result_data) + "\n")

                except Exception as e:
                    logger.error(
                        f"Error processing entry for {Path(entry['image_path']).name} (Index {idx}): {e}",
                        exc_info=True,
                    )
                    with open(error_log_path, "a", encoding="utf-8") as f_err:
                        f_err.write(
                            json.dumps(
                                {
                                    "entry": entry,
                                    "error": str(e),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                            + "\n"
                        )
    except KeyboardInterrupt:
        logger.warning("Inference loop interrupted by user.")
    finally:
        logger.info("Inference loop completed or was interrupted.")


def run_maira2_inference(
    evaluation_batch: List[Dict[str, Any]],
    model: torch.nn.Module,
    processor: Any,
    results_jsonl_path: Path,
    max_new_tokens: int,
):
    """Runs phrase grounding inference on a batch using MAIRA-2."""
    if not evaluation_batch:
        logger.info("Evaluation batch is empty. Nothing to process.")
        return

    logger.info(f"Starting MAIRA-2 inference on {len(evaluation_batch)} entries...")
    error_log_path = results_jsonl_path.parent / "errors.jsonl"
    device = model.device

    try:
        with open(results_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for idx, entry in enumerate(
                tqdm(evaluation_batch, desc="Grounding with MAIRA-2")
            ):
                try:
                    pil_image = Image.open(entry["image_path"]).convert("RGB")
                    location_phrase = entry["location"]

                    processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
                        frontal_image=pil_image,
                        phrase=location_phrase,
                        return_tensors="pt",
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

                    # --- Graceful Error Handling for Parsing ---
                    grounding_prediction = "PARSING_ERROR"
                    try:
                        # This is the step that can fail
                        grounding_prediction = processor.convert_output_to_plaintext_or_grounded_sequence(
                            raw_prediction
                        )
                    except AssertionError:
                        logger.warning(
                            f"AssertionError while parsing prediction for {Path(entry['image_path']).name}. "
                            f"Saving raw output. Raw: '{raw_prediction}'"
                        )
                    # --- End of Error Handling ---

                    result_data = entry.copy()
                    result_data["raw_prediction"] = raw_prediction
                    result_data["grounding_prediction"] = grounding_prediction
                    f_jsonl.write(json.dumps(result_data) + "\n")

                except Exception as e:
                    logger.error(
                        f"Error processing entry for {Path(entry['image_path']).name} (Index {idx}): {e}",
                        exc_info=True,
                    )
                    with open(error_log_path, "a", encoding="utf-8") as f_err:
                        f_err.write(
                            json.dumps(
                                {
                                    "entry": entry,
                                    "error": str(e),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                            + "\n"
                        )
    except KeyboardInterrupt:
        logger.warning("Inference loop interrupted by user.")
    finally:
        logger.info("Inference loop completed or was interrupted.")


# --- Helper Functions ---
        
def generate_output_directory(
    model_name: str,
    base_experiments_dir: str,
    adapter_path: Optional[str] = None,
    revision: Optional[str] = None,
) -> Path:
    """
    Generates a consistent output directory path for an evaluation run.

    - If an adapter_path is provided, it creates an 'evaluations' subfolder
      within the adapter's parent directory (the training experiment folder).
    - If no adapter_path is given (for base models), it creates a directory
      in a central 'evaluations' folder within the base_experiments_dir.

    Args:
        model_name: The name of the model being evaluated (e.g., 'medgemma').
        base_experiments_dir: The root directory for all experiments, used as a
                              fallback for base model evaluations.
        adapter_path: Optional path to a model adapter/checkpoint.
        revision: Optional model revision identifier (not used in current logic).

    Returns:
        A Path object for the evaluation output directory.
    """
    if adapter_path:
        # Logic for a fine-tuned model with a checkpoint (e.g., MedGemma with LoRA)
        adapter_path_obj = Path(adapter_path)
        # This is the specific checkpoint name, e.g., "checkpoint-1000"
        checkpoint_name = adapter_path_obj.name
        # This is the parent training experiment directory
        training_experiment_dir = adapter_path_obj.parent
        # The output path is now nested inside the training experiment folder
        output_dir = training_experiment_dir / "evaluations" / checkpoint_name
    else:
        # Fallback logic for a base model (no checkpoint)
        if "maira-2" in model_name and revision:
            run_tag = revision  # Use git revision for MAIRA-2 versioning
        else:
            run_tag = "default"
        
        # Sanitize model_name for the path
        safe_model_name = get_safe_filename(model_name)
        dir_name = f"{safe_model_name}-{run_tag}"
        output_dir = Path(base_experiments_dir) / "evaluations" / dir_name

    return output_dir


# --- Main Execution ---

def main(args: argparse.Namespace):
    """Main function to run the evaluation script."""
    args_dict = vars(args)
    args_str = json.dumps(args_dict, indent=4)
    logger.info(f"--- Evaluation Run Configuration ---\n{args_str}")

    # --- Common Setup: Load data and create transforms ---
    dependencies = load_dependencies()
    all_test_entries = build_full_test_set(**dependencies)

    image_transforms = None
    if args.image_transforms_kwargs:
        logger.info(
            f"Creating image transforms with kwargs: {args.image_transforms_kwargs}"
        )
        image_transforms = create_image_transforms(**args.image_transforms_kwargs)

    # --- Workflow Selection: Location-Specific vs. General ---
    if args.eval_locations:
        logger.info(">>> Starting Location-Specific Evaluation Mode <<<")
        run_location_specific_evaluation(args, all_test_entries, image_transforms)
    else:
        logger.info(">>> Starting General Evaluation Mode <<<")
        run_general_evaluation(args, all_test_entries, image_transforms)

    logger.info("Evaluation script finished.")


def run_general_evaluation(
    args: argparse.Namespace,
    all_test_entries: List[Dict[str, Any]],
    image_transforms: Optional[Callable],
):
    """Runs the original, general evaluation workflow."""
    # 1. Setup output directory and paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
        logger.info(f"Using manually specified output directory: {output_dir}")
    else:
        output_dir = generate_output_directory(
            model_name=args.model_name,
            base_experiments_dir=EXPERIMENTS_DIR,
            adapter_path=args.medgemma_adapter_path,
            revision=args.maira2_revision,
        )
        logger.info(f"Automatically generated output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl_path = output_dir / "predictions.jsonl"
    core_sample_path = Path(EXPERIMENTS_DIR) / "evaluations" / CORE_SAMPLE_FILENAME
    logger.info(f"Results will be saved to: {results_jsonl_path}")
    logger.info(f"Core sample cache path: {core_sample_path}")

    # 2. Prepare the specific batch for this run
    evaluation_batch = prepare_evaluation_sample(
        all_entries=all_test_entries,
        results_jsonl_path=results_jsonl_path,
        core_sample_path=core_sample_path,
        target_sample_size=args.sample_size,
        core_sample_size=args.core_sample_size,
        force_regenerate_core_sample=args.force_regenerate_core_sample,
        skip_sampling_beyond_core_sample=args.skip_sampling_beyond_core_sample,
    )

    if not evaluation_batch:
        logger.info("No new entries to process. Evaluation may be complete.")
        return
        
    # 3. Load model and run inference
    run_inference_for_batch(args, evaluation_batch, results_jsonl_path, image_transforms)


def run_location_specific_evaluation(
    args: argparse.Namespace,
    all_test_entries: List[Dict[str, Any]],
    image_transforms: Optional[Callable],
):
    """Runs the new, location-focused evaluation workflow."""
    base_output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else generate_output_directory(
            model_name=args.model_name,
            base_experiments_dir=EXPERIMENTS_DIR,
            adapter_path=args.medgemma_adapter_path,
            revision=args.maira2_revision,
        )
    )
    logger.info(f"Using base output directory for this run: {base_output_dir}")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Define the SHARED directory for canonical location samples
    location_samples_dir = (
        Path(EXPERIMENTS_DIR) / "evaluations" / "chest_imagenome_location_specific_samples"
    )
    location_samples_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Shared location sample cache: {location_samples_dir}")

    all_entries_df = pd.DataFrame(all_test_entries)
    all_available_locations = set(all_entries_df["location"].unique())

    for location in args.eval_locations:
        logger.info(f"\n===== Processing Location: {location} =====")

        if location not in all_available_locations:
            logger.warning(
                f"Location '{location}' not found in the test set. Skipping."
            )
            continue

        # Path to the canonical, shared sample for this location
        safe_location_name = get_safe_filename(location)
        location_core_sample_path = (
            location_samples_dir / f"{safe_location_name}_sample.jsonl"
        )

        # Filter all entries for the current location
        location_entries_list = all_entries_df[
            all_entries_df["location"] == location
        ].to_dict("records")

        # --- Manage the Core Sample ---
        core_sample = []
        if (
            location_core_sample_path.is_file()
            and not args.force_regenerate_core_sample
        ):
            logger.info(f"Loading cached core sample from {location_core_sample_path}")
            core_sample = load_jsonl(location_core_sample_path)
            # Log warning if cached size differs from requested core size
            if len(core_sample) != args.location_core_sample_size:
                logger.warning(
                    f"Cached core sample for '{location}' has {len(core_sample)} entries, "
                    f"but requested core size is {args.location_core_sample_size}. "
                    "Using the existing cached sample. Use --force_regenerate_core_sample to overwrite."
                )
        else:
            core_sample = _generate_and_cache_location_core_sample(
                location_entries=location_entries_list,
                location_name=location,
                all_entries_df=all_entries_df,
                cache_path=location_core_sample_path,
                core_sample_size=args.location_core_sample_size,
            )

        if not core_sample:
            logger.warning(
                f"No core sample available or generated for '{location}'. Skipping evaluation."
            )
            continue

        # --- Prepare the batch for THIS SPECIFIC RUN ---
        location_output_dir = base_output_dir / safe_location_name
        location_output_dir.mkdir(parents=True, exist_ok=True)
        results_jsonl_path = location_output_dir / "predictions.jsonl"
        logger.info(f"Results will be saved to: {results_jsonl_path}")
        
        evaluation_batch = prepare_location_evaluation_batch(
            all_location_entries=location_entries_list,
            results_jsonl_path=results_jsonl_path,
            core_sample=core_sample,
            target_eval_size=args.location_eval_size,
            skip_sampling_beyond_core_sample=args.skip_sampling_beyond_core_sample,
        )

        if not evaluation_batch:
            logger.info(
                f"No new entries to process for '{location}' in this run. Evaluation may be complete."
            )
            continue

        run_inference_for_batch(
            args, evaluation_batch, results_jsonl_path, image_transforms
        )


def run_inference_for_batch(
    args: argparse.Namespace,
    evaluation_batch: List[Dict[str, Any]],
    results_jsonl_path: Path,
    image_transforms: Optional[Callable],
):
    """Helper function to load a model and run inference on a given batch."""
    logger.info(f"Preparing to run inference on a batch of {len(evaluation_batch)} items.")
    # Load model and run inference based on selection
    if args.model_name in ["medgemma", "medgemma-4b-it"]:
        model, processor = load_medgemma_model(
            base_model_id=args.medgemma_base_model_id,
            adapter_path=args.medgemma_adapter_path,
        )
        run_medgemma_inference(
            evaluation_batch=evaluation_batch,
            model=model,
            processor=processor,
            results_jsonl_path=results_jsonl_path,
            max_new_tokens=args.max_new_tokens,
            image_transforms=image_transforms,
            use_localize_instead_of_locate=args.use_localize_instead_of_locate,
        )
    elif args.model_name == "maira-2":
        model, processor = load_maira2_model(
            model_id=args.maira2_model_id, revision=args.maira2_revision
        )
        run_maira2_inference(
            evaluation_batch=evaluation_batch,
            model=model,
            processor=processor,
            results_jsonl_path=results_jsonl_path,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Anatomy-Grounded Report Generation Evaluation."
    )

    # --- General Arguments ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None, # Changed from required=True
        help="Directory to save results. If not provided, a directory will be automatically generated in the EXPERIMENTS_DIR based on the model and checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["medgemma", "medgemma-4b-it", "maira-2"],
        help="Name of the model to evaluate.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Total number of samples to process in this run.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="Maximum number of new tokens for the model to generate.",
    )
    parser.add_argument(
        "--image_transforms_kwargs",
        type=str,
        default=None,
        help="Optional JSON string of keyword arguments to build an image transformation function.",
    )

    # --- Sampling Arguments ---
    parser.add_argument(
        "--core_sample_size",
        type=int,
        default=1000,
        help="Size of the fixed, balanced core sample for reproducibility.",
    )
    parser.add_argument(
        "--force_regenerate_core_sample",
        action="store_true",
        help="If set, ignores and overwrites an existing core sample cache.",
    )
    parser.add_argument(
        "--skip_sampling_beyond_core_sample",
        action="store_true",
        help=("If set, skips sampling beyond the core sample. This is useful for debugging or when you want to use the core sample only. "
              "By default, the script will sample beyond the core sample to reach the target sample size."),
        default=False,
    )

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
        help="Optional path to the LoRA adapter for a fine-tuned MedGemma model. If not provided, the original base model will be used.",
    )
    parser.add_argument(
        "--use_localize_instead_of_locate",
        action="store_true",
        help="If set, the prompt will use the word 'localize' instead of 'locate'. For backwards compatibility.",
        default=False,
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
        # default="33f99f8",
        default=None,
        help="Git revision for the MAIRA-2 model. Defaults to 'main' (latest).",
    )

    # --- Location-Specific Evaluation Arguments ---
    parser.add_argument(
        "--eval_locations",
        nargs="+",
        type=str,
        default=None,
        help="A list of specific anatomical locations to evaluate. "
             "If provided, the script runs a separate, focused evaluation for each location.",
    )
    parser.add_argument(
        "--location_eval_size",
        type=int,
        default=300,
        help="The target number of samples to process in this specific run for each location.",
    )
    parser.add_argument(
        "--location_core_sample_size",
        type=int,
        default=300,
        help="The size of the canonical, shared core sample to generate and cache for each location.",
    )

    # --- Parse and Run ---
    args = parser.parse_args()

    if args.image_transforms_kwargs:
        args.image_transforms_kwargs = json.loads(args.image_transforms_kwargs)

    main(args)
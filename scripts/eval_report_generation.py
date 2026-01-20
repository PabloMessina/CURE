import argparse
import json
import logging
import gc
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from vlm_research_kit.settings import EXPERIMENTS_DIR
from vlm_research_kit.data.datasets.mimiccxr_dataset import MIMICCXRDataset
from vlm_research_kit.data.datasets.chest_imagenome_dataset import (
    SUPPORTED_LOCATE_AND_DESCRIBE_LOCATIONS,
    SUPPORTED_DESCRIBE_LOCATIONS,
)
from vlm_research_kit.utils.logging_utils import setup_logging

from vlm_research_kit.utils.evaluation_utils import (
    generate_output_directory,
    prepare_evaluation_batch,
)
from vlm_research_kit.utils.model_utils import (
    load_maira2_model,
    load_medgemma_model,
    load_cxrmate_rrg24_model,
)

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
    Loads the specified report generation dataset for evaluation.

    Args:
        dataset_name: The name of the dataset to load ('mimic-cxr').
        split: The split of the dataset to load (e.g. 'train', 'validate', 'test').
        image_transforms_kwargs: Optional dictionary of keyword arguments to build an image transformation function.

    Returns:
        A tuple containing the initialized PyTorch Dataset and the unique ID key
        for that dataset used for resuming progress.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_name == "mimic-cxr":
        # For MIMIC-CXR in frontal_only mode, image_path is a unique identifier per sample
        unique_id_key = "image_path"
        dataset = MIMICCXRDataset(
            split=split,
            image_loading_mode="frontal_only",
            return_image_paths=True,
            return_study_info=True, # Good to keep for metadata
            image_transforms_kwargs=image_transforms_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown or unsupported dataset for this task: {dataset_name}"
        )

    logger.info(
        f"Loaded {len(dataset)} samples from the '{dataset_name}' {split} split."
    )
    return dataset, unique_id_key


# --- Model-Specific Functions ---

def run_medgemma_inference(
    full_dataset: Dataset,
    evaluation_batch_indices: List[int],
    model: torch.nn.Module,
    processor: Any,
    results_jsonl_path: Path,
    max_new_tokens: int,
    prompt_text: str = "Generate a grounded report.",
    system_instruction: Optional[str] = None,
):
    """Runs grounded report generation inference on a batch using MedGemma."""
    if not evaluation_batch_indices:
        logger.info("Evaluation batch is empty. Nothing to process.")
        return

    logger.info(
        f"Starting MedGemma inference on {len(evaluation_batch_indices)} entries..."
    )
    error_log_path = results_jsonl_path.parent / "errors.jsonl"

    try:
        with open(results_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for iter_count, idx in enumerate(tqdm(evaluation_batch_indices, desc="Generating Reports with MedGemma")):
                entry = full_dataset[idx]
                try:
                    pil_image = entry["image"]
                    messages = []
                    if system_instruction:
                        messages.append({
                            "role": "system",
                            "content": [{"type": "text", "text": system_instruction}],
                        })
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": prompt_text},
                        ],
                    })
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
                    prediction = decoded_response.split("model\n")[-1].strip()

                    result_data = entry.copy()
                    if "image" in result_data:
                        del result_data["image"]
                    if system_instruction:
                        result_data["system_instruction"] = system_instruction
                    result_data["prompt"] = prompt_text
                    result_data["prediction"] = prediction
                    f_jsonl.write(json.dumps(result_data) + "\n")
                except Exception as e:
                    logger.error(f"Error processing entry for {entry.get('image_path', 'N/A')} (Index {idx}): {e}", exc_info=True)
                    with open(error_log_path, "a", encoding="utf-8") as f_err:
                        error_entry = entry.copy()
                        if "image" in error_entry: del error_entry["image"]
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
                if (iter_count + 1) % 100 == 0: # Only collect garbage every 100 samples to avoid overhead
                    gc.collect()

    except KeyboardInterrupt:
        logger.warning("Inference loop interrupted by user.")
    finally:
        logger.info("Inference loop completed or was interrupted.")


def run_maira2_inference(
    full_dataset: Dataset,
    evaluation_batch_indices: List[int],
    model: transformers.AutoModelForCausalLM,
    processor: Any,
    results_jsonl_path: Path,
    max_new_tokens: int,
    get_grounding: bool = False,
):
    """Runs grounded report generation inference on a batch using MAIRA-2."""
    if not evaluation_batch_indices:
        logger.info("Evaluation batch is empty. Nothing to process.")
        return

    logger.info(f"Starting MAIRA-2 inference on {len(evaluation_batch_indices)} entries...")
    error_log_path = results_jsonl_path.parent / "errors.jsonl"
    device = model.device

    try:
        with open(results_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for iter_count, idx in enumerate(tqdm(evaluation_batch_indices, desc="Generating Reports with MAIRA-2")):
                entry = full_dataset[idx]
                try:
                    image_key = "image" if "image" in entry else "pixel_values"
                    pil_image = entry[image_key]

                    processed_inputs = (
                        processor.format_and_preprocess_reporting_input(
                            current_frontal=pil_image,
                            current_lateral=None,
                            prior_frontal=None,
                            indication=None,
                            technique=None,
                            comparison=None,
                            prior_report=None,
                            return_tensors="pt",
                            get_grounding=get_grounding,
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
                     # Findings generation completions have a single leading space
                    raw_prediction = raw_prediction.lstrip()

                    parsed_prediction = "PARSING_ERROR"
                    try:
                        parsed_prediction = (
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
                    del result_data[image_key]
                    result_data["raw_prediction"] = raw_prediction
                    result_data["parsed_prediction"] = parsed_prediction
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
                if (iter_count + 1) % 100 == 0: # Only collect garbage every 100 samples to avoid overhead
                    gc.collect()
    except KeyboardInterrupt:
        logger.warning("Inference loop interrupted by user.")
    finally:
        logger.info("Inference loop completed or was interrupted.")


def run_cxrmate_rrg24_inference(
    full_dataset: Dataset,
    evaluation_batch_indices: List[int],
    model: transformers.AutoModel,
    processor: Any, # This will be the tokenizer
    results_jsonl_path: Path,
    max_new_tokens: int,
    num_beams: int,
):
    """Runs report generation inference on a batch using CXRMate-RRG24."""
    if not evaluation_batch_indices:
        logger.info("Evaluation batch is empty. Nothing to process.")
        return

    logger.info(f"Starting CXRMate-RRG24 inference on {len(evaluation_batch_indices)} entries...")
    error_log_path = results_jsonl_path.parent / "errors.jsonl"
    device = model.device
    tokenizer = processor # Rename for clarity

    try:
        with open(results_jsonl_path, "a", encoding="utf-8") as f_jsonl:
            for iter_count, idx in enumerate(tqdm(evaluation_batch_indices, desc="Generating Reports with CXRMate-RRG24")):
                entry = full_dataset[idx]
                try:
                    # The dataset should return a preprocessed tensor
                    image_tensor = entry["image"]
                    # Add a batch dimension and image sequence dimension for single-image inference
                    pixel_values = image_tensor.unsqueeze(0).unsqueeze(0).to(device=device)

                    with torch.no_grad():
                        output_ids = model.generate( # type: ignore
                            pixel_values=pixel_values,
                            max_length=max_new_tokens,
                            num_beams=num_beams,
                            do_sample=False,
                            bad_words_ids=[
                                [tokenizer.convert_tokens_to_ids("[NF]")],
                                [tokenizer.convert_tokens_to_ids("[NI]")],
                            ],
                        )

                    # Decode using batch_decode and simple string replacement
                    raw_prediction = tokenizer.batch_decode(output_ids)[0]

                    # Clean up the special tokens
                    prediction = (
                        raw_prediction.replace("[BOS]", "")
                        .replace("[EOS]", "")
                        .replace("[SEP]", "\n\n") # Separate findings/impression
                        .strip()
                    )

                    result_data = entry.copy()
                    if "image" in result_data:
                        del result_data["image"]
                    result_data["raw_prediction"] = raw_prediction
                    result_data["prediction"] = prediction
                    f_jsonl.write(json.dumps(result_data) + "\n")

                except Exception as e:
                    logger.error(
                        f"Error processing entry for {entry.get('image_path', 'N/A')} (Index {idx}): {e}",
                        exc_info=True,
                    )
                    with open(error_log_path, "a", encoding="utf-8") as f_err:
                        error_entry = entry.copy()
                        if "image" in error_entry: del error_entry["image"]
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
                    os.fsync(f_jsonl.fileno())

                # --- Call garbage collector to free memory ---
                if (iter_count + 1) % 100 == 0:
                    gc.collect()

    except KeyboardInterrupt:
        logger.warning("Inference loop interrupted by user.")
    finally:
        logger.info("Inference loop completed or was interrupted.")


# --- Main Execution ---

def main(args: argparse.Namespace):
    """Main function to run the evaluation script."""
    args_str = json.dumps(vars(args), indent=4)
    logger.info(f"--- Evaluation Run Configuration ---\n{args_str}")

    # Validate that special modes are only used with a MedGemma adapter
    if args.generate_anatomy_grounded_report and not args.medgemma_adapter_path:
        raise ValueError(
            "Special generation mode --generate_anatomy_grounded_report requires a --medgemma_adapter_path to be specified."
        )
    if args.generate_grounded_report:
        if args.model_name in ["medgemma", "medgemma-4b-it"]:
            if not args.medgemma_adapter_path:
                raise ValueError(
                    "Special generation mode --generate_grounded_report requires a --medgemma_adapter_path to be specified."
                )
        elif args.model_name == "maira-2":
            pass # MAIRA-2 supports grounded report generation natively
        else:
            raise ValueError(
                "Special generation mode --generate_grounded_report is not supported for model type: {args.model_name}."
                "Supported model types are: MedGemma and MAIRA-2."
            )

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
            task_suffix="reportgen",
        )
        logger.info(f"Automatically generated output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DYNAMIC FILENAME LOGIC ---
    base_results_jsonl_path = Path() # Initialize
    if args.model_name == "cxrmate-rrg24":
        # Extract number of beams from args
        num_beams = args.cxrmate_rrg24_num_beams
        base_results_jsonl_path = output_dir / f"predictions_beams_{num_beams}.jsonl"
    else:
        if args.image_transforms_kwargs:
            image_transforms_kwargs_str = json.dumps(args.image_transforms_kwargs, indent=4)
            logger.info(f"Using image transforms kwargs: {image_transforms_kwargs_str}")
            # Extract image size and whether image augmentations are used
            image_size = args.image_transforms_kwargs["image_size"]
            logger.info(f"Image size: {image_size}")
            assert type(image_size) is list and len(image_size) == 2, "Image size must be a list of two integers"
            if (not args.image_transforms_kwargs.get("use_model_specific_transforms", False) or
                args.image_transforms_kwargs["model_name"] == "pil_with_augmentations"):
                base_results_jsonl_path = output_dir / f"predictions_{image_size[0]}x{image_size[1]}_augmented.jsonl"
            else:
                base_results_jsonl_path = output_dir / f"predictions_{image_size[0]}x{image_size[1]}.jsonl"
        else:
            base_results_jsonl_path = output_dir / "predictions.jsonl"

    # 2. Load model
    model, processor = None, None # Defer loading until we know we need it
    use_medgemma = False
    if args.model_name in ["medgemma", "medgemma-4b-it"]:
        use_medgemma = True
        model, processor = load_medgemma_model(
            base_model_id=args.medgemma_base_model_id,
            adapter_path=args.medgemma_adapter_path,
        )
    elif args.model_name == "maira-2":
        model, processor = load_maira2_model(
            model_id=args.maira2_model_id, revision=args.maira2_revision
        )
    elif args.model_name == "cxrmate-rrg24":
        model, processor = load_cxrmate_rrg24_model(
            model_id=args.cxrmate_rrg24_model_id, revision=args.cxrmate_rrg24_revision
        )
        args.image_transforms_kwargs['model'] = model # Pass the model to the image transforms kwargs
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    # 3. Load data
    full_dataset, unique_id_keys = load_evaluation_dataset(
        args.dataset, args.split, image_transforms_kwargs=args.image_transforms_kwargs
    )   

    # --- START OF NEW LOGIC FOR HANDLING GENERATION MODES ---
    if args.generate_anatomy_grounded_report:
        logger.info("--- Running in Anatomy-Grounded Report Generation Mode ---")
        
        assert use_medgemma, "Anatomy-Grounded Report Generation Mode requires MedGemma."
        
        # Validate provided locations
        for loc in args.generate_anatomy_grounded_report:
            if loc not in SUPPORTED_LOCATE_AND_DESCRIBE_LOCATIONS and loc not in SUPPORTED_DESCRIBE_LOCATIONS:
                raise ValueError(
                    f"Anatomical location '{loc}' is not supported. Please choose from the allowed list."
                )
        
        for location in args.generate_anatomy_grounded_report:
            safe_location_name = location.replace(" ", "_")
            logger.info(f"--- Processing anatomical location: {location} ---")
            
            if location in SUPPORTED_LOCATE_AND_DESCRIBE_LOCATIONS:
                prompt = f"Locate and describe the {location}."
            elif location in SUPPORTED_DESCRIBE_LOCATIONS:
                prompt = f"Describe the {location}."
            else:
                raise ValueError(f"Unsupported location: {location}")

            logger.info(f"Running inference for {location} with prompt: {prompt}")

            # Create a unique file path for this location
            location_results_path = base_results_jsonl_path.with_name(
                f"{base_results_jsonl_path.stem}_anatomy_{safe_location_name}{base_results_jsonl_path.suffix}"
            )
            logger.info(f"Results for this location will be saved to: {location_results_path}")

            evaluation_batch_indices = prepare_evaluation_batch(
                full_dataset=full_dataset,
                results_jsonl_path=location_results_path,
                unique_id_keys=unique_id_keys,
                limit=args.limit, return_indices=True, skip_image_loading=True,
            )

            if not evaluation_batch_indices:
                logger.info(f"No new entries to process for {location}. Skipping.")
                continue

            run_medgemma_inference(
                full_dataset=full_dataset,
                evaluation_batch_indices=evaluation_batch_indices,
                model=model, processor=processor,
                results_jsonl_path=location_results_path,
                max_new_tokens=args.max_new_tokens,
                prompt_text=prompt,
            )
        
    else:
        # This block handles both the default mode and the --generate_grounded_report mode
        prompt = args.medgemma_prompt_text
        results_path = base_results_jsonl_path

        if args.generate_grounded_report:
            logger.info("--- Running in Grounded Report Generation Mode ---")
            prompt = "Generate a grounded report."
            results_path = base_results_jsonl_path.with_name(
                f"{base_results_jsonl_path.stem}_grounded{base_results_jsonl_path.suffix}"
            )

        logger.info(f"Results will be saved to: {results_path}")
        evaluation_batch_indices = prepare_evaluation_batch(
            full_dataset=full_dataset, results_jsonl_path=results_path,
            unique_id_keys=unique_id_keys, limit=args.limit,
            return_indices=True, skip_image_loading=True,
        )

        if not evaluation_batch_indices:
            logger.info("No new entries to process. Evaluation may be complete.")
            return

        # Run inference based on model
        if args.model_name in ["medgemma", "medgemma-4b-it"]:
            run_medgemma_inference(
                full_dataset=full_dataset,
                evaluation_batch_indices=evaluation_batch_indices,
                model=model, processor=processor, results_jsonl_path=results_path,
                max_new_tokens=args.max_new_tokens, prompt_text=prompt,
                system_instruction=args.medgemma_system_instruction,
            )
        elif args.model_name == "maira-2":
            run_maira2_inference(
                full_dataset=full_dataset,
                evaluation_batch_indices=evaluation_batch_indices,
                model=model, processor=processor, results_jsonl_path=results_path,
                max_new_tokens=args.max_new_tokens,
                get_grounding=args.generate_grounded_report,
            )
        elif args.model_name == "cxrmate-rrg24":
            run_cxrmate_rrg24_inference(
                full_dataset=full_dataset,
                evaluation_batch_indices=evaluation_batch_indices,
                model=model, processor=processor, results_jsonl_path=results_path,
                max_new_tokens=args.max_new_tokens, num_beams=args.cxrmate_rrg24_num_beams,
            )
        else:
            raise ValueError(f"Unknown model name: {args.model_name}")
    # --- END OF NEW LOGIC ---

    logger.info("Evaluation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Grounded Report Generation Evaluation on PadChest-GR."
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
        choices=["medgemma", "medgemma-4b-it", "maira-2", "cxrmate-rrg24"],
        help="Name of the model to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mimic-cxr"],
        help="Name of the dataset to evaluate on.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "validation", "test"],
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
        default=350,
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
    parser.add_argument(
        "--medgemma_prompt_text",
        type=str,
        default="Describe this X-ray",
        help="Prompt text for MedGemma to generate a report.",
    )
    parser.add_argument(
        "--medgemma_system_instruction",
        type=str,
        default=None,
        help="System instruction for MedGemma to generate a report.",
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

    # --- CXRMate-RRG24 Specific Arguments ---
    parser.add_argument(
        "--cxrmate_rrg24_model_id",
        type=str,
        default="aehrc/cxrmate-rrg24",
        help="Model ID for CXRMate-RRG24 from Hugging Face.",
    )
    parser.add_argument(
        "--cxrmate_rrg24_revision",
        type=str,
        default="ef90c2725315efaabf1eb1762d9b903c5e57acdd",
        help="Git revision for the CXRMate-RRG24 model.",
    )
    parser.add_argument(
        "--cxrmate_rrg24_num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search generation with CXRMate-RRG24.",
    )

    # Create a mutually exclusive group for generation modes
    mode_group = parser.add_argument_group(
        "Special MedGemma Generation Modes (requires --medgemma_adapter_path)"
    ).add_mutually_exclusive_group()

    mode_group.add_argument(
        "--generate_grounded_report",
        action="store_true",
        help="If set, changes the MedGemma prompt to 'Generate a grounded report.' "
             "and saves results to a separate file.",
    )
    mode_group.add_argument(
        "--generate_anatomy_grounded_report",
        nargs="+",
        metavar="LOCATION",
        default=None,
        help="If set, generates a report for each specified anatomical location. "
             "Saves results to a file per location. Example: --generate_anatomy_grounded_report 'left lung' 'cardiac silhouette'",
    )

    # --- Parse and Run ---
    args = parser.parse_args()

    if args.model_name == "cxrmate-rrg24":
        # Override image transforms kwargs to use cxrmate-rrg24 transforms
        old_image_transforms_kwargs = None
        if args.image_transforms_kwargs is not None:
            old_image_transforms_kwargs = args.image_transforms_kwargs            
        args.image_transforms_kwargs = {
            "use_model_specific_transforms": True,
            "model_name": "aehrc/cxrmate-rrg24",
        }
        if old_image_transforms_kwargs is not None:
            logger.warning(
                f"Image transforms kwargs are being overridden for CXRMate-RRG24."
                f"\n\nOriginal: {old_image_transforms_kwargs}\n\nNew: {args.image_transforms_kwargs}"
            )
        else:
            logger.warning(
                f"Image transforms kwargs are being overridden for CXRMate-RRG24."
                f"\n\nNo original image transforms kwargs provided."
            )
    elif args.image_transforms_kwargs is not None: # Convert string to dict if necessary
        args.image_transforms_kwargs = json.loads(args.image_transforms_kwargs)

    main(args)
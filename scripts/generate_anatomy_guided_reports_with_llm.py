import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Set
from tqdm import tqdm
import pandas as pd

from vlm_research_kit.settings import LLM_PROMPTS_DIR, MIMIC_CXR_SPLIT_CSV_PATH
from vlm_research_kit.utils.logging_utils import setup_logging
from vlm_research_kit.utils.file_utils import load_pickle, read_txt, load_jsonl
from vlm_research_kit.utils.openai_api_utils import orchestrate_api_calls

DEFAULT_ANATOMIES = [
    'right lung',
    'left lung',
    'left costophrenic angle',
    'right costophrenic angle',
    'spine',
    'right clavicle',
    'left clavicle',
    'mediastinum',
    'cardiac silhouette',
]


def parse_llm_anatomy_output(llm_response_str: str) -> Dict[str, str]:
    """
    Parses the LLM response string for the anatomy-focused report generation task.
    We expect to find a JSON object in the following format:
    {
        "reasoning": "Concise explanation of the thought process.",
        "mini-report": "A brief, focused summary of the anatomical location."
    }
    Args:
        llm_response_str: The string response from the LLM.
    Returns:
        A dictionary containing the parsed 'reasoning' and 'mini-report'.
    """
    start = llm_response_str.find("{")
    end = llm_response_str.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"Could not find a valid JSON object in: {llm_response_str}"
        )
    json_str = llm_response_str[start : end + 1]
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError(f"Parsed data is not a dictionary: {data}")
    if "reasoning" not in data or "mini-report" not in data:
        raise ValueError(f"Missing expected keys 'reasoning' or 'mini-report' in parsed data: {data}")
    if not isinstance(data["reasoning"], str) or not isinstance(
        data["mini-report"], str
    ):
        raise ValueError(
            f"'reasoning' and 'mini-report' must be strings. Got: "
            f"{type(data['reasoning'])}, {type(data['mini-report'])}"
        )
    return {
        "reasoning": data["reasoning"].strip(),
        "mini-report": data["mini-report"].strip(),
    }


def _clean_report_text(report: str) -> str:
    """Cleans report text by stripping whitespace and normalizing spaces."""
    return " ".join(report.strip().split())


def generate_reports_main(args):
    """
    Main function to load reports and locations, query an LLM to generate
    anatomy-focused mini-reports, and save the results.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting anatomy-guided report generation process...")
    logger.info(f"Input PKL: {args.input_pkl}")
    logger.info(f"Processing split: '{args.split_to_process}'")
    logger.info(f"Output JSONL: {args.output_jsonl}")

    output_file_path = Path(args.output_jsonl)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.api_responses_filepath:

        # --- 1. Load System Instructions ---
        try:
            system_instructions_path = (
                LLM_PROMPTS_DIR / args.system_instructions_relative_path
            )
            system_instructions = read_txt(system_instructions_path).strip()
            logger.info(
                "Loaded system instructions (first 100 chars): "
                f"{system_instructions[:100]}..."
            )
        except FileNotFoundError:
            logger.error(
                f"System instructions file not found: {system_instructions_path}"
            )
            return

        # --- 2. Load already processed reports to avoid re-running ---
        already_processed_queries = set()
        if output_file_path.exists():
            processed_items = load_jsonl(output_file_path)
            for item in processed_items:
                if "metadata" in item and "query" in item["metadata"]:
                    already_processed_queries.add(item["metadata"]["query"])
            logger.info(
                f"Found {len(processed_items)} already processed items. "
                f"Loaded {len(already_processed_queries)} unique queries to skip."
            )

        # --- 3. Load MIMIC-CXR split data to filter reports ---
        split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV_PATH)
        target_dicom_ids: Set[str] = set(
            split_df[split_df["split"] == args.split_to_process]["dicom_id"]
        )
        logger.info(
            f"Loaded {len(target_dicom_ids)} DICOM IDs for the "
            f"'{args.split_to_process}' split."
        )

        # --- 4. Load main report data from pickle file ---
        try:
            report_data = load_pickle(args.input_pkl)
            logger.info(f"Loaded {len(report_data)} entries from {args.input_pkl}")
        except FileNotFoundError:
            logger.error(f"Input pickle file not found: {args.input_pkl}")
            return

        # --- 5. Prepare (report, location) queries for the LLM ---
        queries_to_process_for_llm = set()
        for entry in tqdm(report_data, desc="Preparing queries"):
            dicom_id = entry["dicom_id"]
            if dicom_id not in target_dicom_ids:
                continue

            report_text = _clean_report_text(entry["original_report"])
            location2report_snippet = entry["location2report_snippet"]

            if not report_text:
                continue

            locations = set(location2report_snippet.keys())
            locations.update(DEFAULT_ANATOMIES) # Ensure default anatomies are included
            locations.discard("unknown") # Remove 'unknown' if present

            for location in locations:
                query = f'Report: "{report_text}" | Location: "{location}"'
                if query not in already_processed_queries:
                    queries_to_process_for_llm.add(query)

                if (
                    args.max_queries_to_process is not None
                    and len(queries_to_process_for_llm)
                    >= args.max_queries_to_process
                ):
                    break
            if (
                args.max_queries_to_process is not None
                and len(queries_to_process_for_llm)
                >= args.max_queries_to_process
            ):
                logger.info(
                    "Reached max_queries_to_process limit: "
                    f"{args.max_queries_to_process}"
                )
                break

        if not queries_to_process_for_llm:
            logger.info("No new reports to process. Exiting.")
            return

        queries_to_process_for_llm = list(queries_to_process_for_llm)
        logger.info(
            f"Prepared {len(queries_to_process_for_llm)} new (report, location) "
            "queries for LLM."
        )

        # Print the first few queries for inspection
        logger.info("First few queries to process:")
        for i, query in enumerate(queries_to_process_for_llm[:5]):
            logger.info(f"Query {i + 1}: {query}")

    else:
    
        queries_to_process_for_llm = None
        system_instructions = None

    # --- 6. Run LLM API requests ---
    try:
        orchestrate_api_calls(
            api_responses_filepath=args.api_responses_filepath,
            texts=queries_to_process_for_llm,
            system_instructions=system_instructions,
            model_name=args.model_name,
            api_key_name=args.api_key_name,
            api_type=args.api_type,
            max_requests_per_minute=args.max_requests_per_minute,
            max_tokens_per_minute=args.max_tokens_per_minute,
            max_tokens_per_request=args.max_tokens_per_request,
            temperature=args.temperature,
            parse_output=parse_llm_anatomy_output,
            save_filepath=output_file_path,
            delete_api_requests_and_responses=not args.dont_delete_api_requests_and_responses,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            log_info_every_n_requests=args.log_info_every_n_requests,
        )
    except Exception as e:
        logger.error(f"Error during orchestrate_api_calls: {e}", exc_info=True)
        return

    logger.info("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate anatomy-guided mini-reports from full medical reports using an LLM."
    )

    # --- Input/Output Arguments ---
    parser.add_argument(
        "--input_pkl",
        type=str,
        required=True,
        help="Path to the input pickle file (e.g., location_report_snippets.pkl).",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to save the output JSONL file with generated mini-reports.",
    )
    parser.add_argument(
        "--system_instructions_relative_path",
        type=str,
        default="report_labeling/anatomy_focused_reports.txt",
        help="Relative path to the system instructions file for the LLM, relative to LLM_PROMPTS_DIR.",
    )
    parser.add_argument(
        "--api_responses_filepath",
        type=str,
        default=None,
        help="Path to precomputed API responses JSONL file. Useful when a previous run crashed. "
        "If provided, the script will skip API calls and use these responses.",
    )

    # --- Data Processing Arguments ---
    parser.add_argument(
        "--split_to_process",
        type=str,
        default="test",
        choices=["train", "validate", "test"],
        help="The data split to process from the MIMIC-CXR CSV.",
    )

    # --- LLM Configuration (reusing from your previous script) ---
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Name of the LLM model to use.",
    )
    parser.add_argument(
        "--api_key_name",
        type=str,
        default="OPENAI_API_KEY",
        help="Name of the environment variable holding the API key.",
    )
    parser.add_argument(
        "--api_type",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="Type of API to use.",
    )

    # --- Throttling and Request Parameters ---
    parser.add_argument(
        "--max_requests_per_minute", type=float, default=3000
    )
    parser.add_argument(
        "--max_tokens_per_minute", type=float, default=200000
    )
    parser.add_argument("--max_tokens_per_request", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--log_info_every_n_requests", type=int, default=50)

    # --- Processing Control ---
    parser.add_argument(
        "--max_queries_to_process",
        type=int,
        default=None,
        help="Maximum number of new (report, location) pairs to process. Processes all if None.",
    )
    parser.add_argument(
        "--dont_delete_api_requests_and_responses",
        action="store_true",
        help="If set, keeps temporary files created by orchestrate_api_calls.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level.",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level)

    generate_reports_main(args)
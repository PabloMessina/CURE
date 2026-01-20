import json
import argparse
import logging
import re
from pathlib import Path
from typing import List, Any, Optional
from tqdm import tqdm

from vlm_research_kit.data.datasets.mimiccxr_dataset import get_dicom_id_to_report_map
from vlm_research_kit.settings import LLM_PROMPTS_DIR
from vlm_research_kit.utils.logging_utils import setup_logging
from vlm_research_kit.utils.file_utils import read_txt, load_jsonl
from vlm_research_kit.utils.openai_api_utils import orchestrate_api_calls

def parse_llm_label_output__paultimothymooney_chest_xray_pneumonia(llm_response_str: str) -> Any:
    """
    Parses the LLM response string for the specific format used with
    PaulTimothyMooney/chest-xray-pneumonia dataset.
    We expect to find a JSON object with the following format:
    {
        "reason": "Brief explanation of your reasoning.",
        "has_abnormalities": "yes" | "no" | "unknown",
        "supports_pneumonia": "definitely no" | "probably no" | "unknown" | "probably yes" | "definitely yes"
    }
    Args:
        llm_response_str: The string response from the LLM.
    Returns:
        A dictionary containing the parsed label information.
    """
    start = llm_response_str.find('{')
    end = llm_response_str.rfind('}')
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find a valid JSON object in: {llm_response_str}")
    json_str = llm_response_str[start:end+1]
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError(f"Parsed data is not a dictionary: {data}")
    if "reason" not in data or "has_abnormalities" not in data or "supports_pneumonia" not in data:
        raise ValueError(f"Missing expected keys in parsed data: {data}")
    if data["has_abnormalities"] not in ["yes", "no", "unknown"]:
        raise ValueError(f"Invalid value for 'has_abnormalities': {data['has_abnormalities']}")
    if data["supports_pneumonia"] not in ["definitely no", "probably no", "unknown", "probably yes", "definitely yes"]:
        raise ValueError(f"Invalid value for 'supports_pneumonia': {data['supports_pneumonia']}")
    if not isinstance(data["reason"], str):
        raise ValueError(f"'reason' should be a string, got: {type(data['reason'])}")
    if data["reason"].strip() == "":
        raise ValueError("The 'reason' field should not be empty.")
    return {
        "reason": data["reason"].strip(),
        "has_abnormalities": data["has_abnormalities"],
        "supports_pneumonia": data["supports_pneumonia"]
    }


def parse_llm_label_output__anatomy_guided_mini_reports(llm_response_str: str) -> Any:
    """
    Parses the LLM response string for the anatomy-guided mini reports format.
    We expect the response to be a JSON object with the following structure:
    {
        "reason": "A brief explanation of your reasoning.",
        "mentions_abnormalities": "yes" | "no",
        "mentions_devices": "yes" | "no"
    }
    Args:
        llm_response_str: The string response from the LLM.
    Returns:
        A dictionary containing the parsed label information.
    """
    start = llm_response_str.find('{')
    end = llm_response_str.rfind('}')
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find a valid JSON object in: {llm_response_str}")
    json_str = llm_response_str[start:end+1]
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError(f"Parsed data is not a dictionary: {data}")
    if "reason" not in data or "mentions_abnormalities" not in data or "mentions_devices" not in data:
        raise ValueError(f"Missing expected keys in parsed data: {data}")
    if data["mentions_abnormalities"] not in ["yes", "no"]:
        raise ValueError(f"Invalid value for 'mentions_abnormalities': {data['mentions_abnormalities']}")
    if data["mentions_devices"] not in ["yes", "no"]:
        raise ValueError(f"Invalid value for 'mentions_devices': {data['mentions_devices']}")
    if not isinstance(data["reason"], str):
        raise ValueError(f"'reason' should be a string, got: {type(data['reason'])}")
    if data["reason"].strip() == "":
        raise ValueError("The 'reason' field should not be empty.")
    return {
        "reason": data["reason"].strip(),
        "mentions_abnormalities": data["mentions_abnormalities"],
        "mentions_devices": data["mentions_devices"]
    }

def parse_llm_label_output__gt_vs_gen_comparison(llm_response_str: str) -> Any:
    """
    Parses the LLM response for comparing a generated report against a ground-truth.
    We expect a JSON object with the following structure:
    {
        "reason": "Brief explanation of your reasoning.",
        "gt_has_abnormalities": "yes" | "no",
        "gt_has_devices": "yes" | "no",
        "gen_has_abnormalities": "yes" | "no",
        "gen_has_devices": "yes" | "no",
        "gen_has_correct_abnormalities": "yes" | "no",
        "gen_has_hallucinated_abnormalities": "yes" | "no",
        "gen_has_correct_devices": "yes" | "no",
        "gen_has_hallucinated_devices": "yes" | "no",
        "nli_status": "contradiction" | "entailment" | "neutral",
    }
    Args:
        llm_response_str: The string response from the LLM.
    Returns:
        A dictionary containing the parsed label information.
    """
    start = llm_response_str.find("{")
    end = llm_response_str.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"Could not find a valid JSON object in: {llm_response_str}"
        )
    json_str = llm_response_str[start : end + 1]
    data = json.loads(json_str)

    expected_keys = [
        "reason",
        "gt_has_abnormalities",
        "gt_has_devices",
        "gen_has_abnormalities",
        "gen_has_devices",
        "gen_has_correct_abnormalities",
        "gen_has_hallucinated_abnormalities",
        "gen_has_correct_devices",
        "gen_has_hallucinated_devices",
        "nli_status",
    ]

    # Check that all expected keys are present
    for key in expected_keys:
        if key not in data:
            raise ValueError(f"Missing expected key '{key}' in parsed data: {data}")

    # Validate the "yes" | "no" fields
    boolean_like_keys = [key for key in expected_keys if key != "reason" and key != "nli_status"]
    for key in boolean_like_keys:
        if data[key] not in ["yes", "no"]:
            raise ValueError(
                f"Invalid value for '{key}': '{data[key]}'. "
                "Expected 'yes' or 'no'."
            )

    # Validate the nli_status field
    if data["nli_status"] not in ["contradiction", "entailment", "neutral"]:
        raise ValueError(f"Invalid value for 'nli_status': '{data['nli_status']}'. Expected 'contradiction', 'entailment', or 'neutral'.")

    # Validate the reason field
    if not isinstance(data["reason"], str) or data["reason"].strip() == "":
        raise ValueError("The 'reason' field must be a non-empty string.")

    # Return the full, validated dictionary
    return {
        "reason": data["reason"].strip(),
        "gt_has_abnormalities": data["gt_has_abnormalities"],
        "gt_has_devices": data["gt_has_devices"],
        "gen_has_abnormalities": data["gen_has_abnormalities"],
        "gen_has_devices": data["gen_has_devices"],
        "gen_has_correct_abnormalities": data[
            "gen_has_correct_abnormalities"
        ],
        "gen_has_hallucinated_abnormalities": data[
            "gen_has_hallucinated_abnormalities"
        ],
        "gen_has_correct_devices": data["gen_has_correct_devices"],
        "gen_has_hallucinated_devices": data[
            "gen_has_hallucinated_devices"
        ],
        "nli_status": data["nli_status"],
    }
    
def _parse_maira2_grounding_prediction_to_text(raw_prediction: str) -> Optional[str]:
    """
    Parses the raw prediction from MAIRA-2 to a text string.
    Args:
        raw_prediction: The raw prediction from MAIRA-2.
    Returns:
        A string containing the parsed prediction.  
    """
    return re.sub(r'<[^<>]+>', '', raw_prediction).strip() # Between < and >, match any character except for < and >

def _concatenate_report_sentences(sentences: List[str]) -> str:
    """
    Concatenates a list of sentences into a single report text.
    Args:
        sentences: List of sentences to concatenate.
    Returns:
        A single string containing all sentences concatenated with spaces.
    """
    report = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if report and report[-1] != '.':
            report += '.' # Ensure sentences are properly punctuated
        report += f" {sentence}" if report else sentence
    report = report.strip()
    if report and report[-1] != '.':
        report += '.'
    return report


DESCRIPTION_REGEX = re.compile(r'descri.*?:\s*(.*?)\s*$', re.IGNORECASE)

def label_reports_main(args):
    """
    Main function to load reports, query an LLM for labels,
    and save the augmented reports.
    """
    logger.info("Starting report labeling process...")
    logger.info(f"Input JSONL: {args.input_jsonl}")
    logger.info(f"Output Labeled JSONL: {args.output_labeled_jsonl}")
    logger.info(f"System Instructions Relative Path: {args.system_instructions_relative_path}")
    logger.info(f"Report Generation Mode: {args.report_generation_mode}")
    logger.info(f"LLM Model: {args.model_name}")

    output_file_path = Path(args.output_labeled_jsonl)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.api_responses_filepath:

        # --- 1. Load System Instructions ---
        try:
            system_instructions_path = LLM_PROMPTS_DIR / args.system_instructions_relative_path
            system_instructions = read_txt(system_instructions_path).strip()
            if not system_instructions:
                logger.error(f"System instructions file '{system_instructions_path}' is empty.")
                return
            logger.info(f"Loaded system instructions (first 100 chars): {system_instructions[:100]}...")
        except FileNotFoundError:
            logger.error(f"System instructions file not found: {system_instructions_path}")
            return
        except Exception as e:
            logger.error(f"Error loading system instructions: {e}")
            return

        # --- 2. Load already processed reports ---
        already_processed_queries = set()
        if output_file_path.exists():
            logger.info(f"Output file {output_file_path} exists. Loading already processed queries.")
            processed_items = load_jsonl(output_file_path)
            logger.info(f"Found {len(processed_items)} already processed items in {output_file_path}.")
            for item in processed_items:
                already_processed_queries.add(item["metadata"]["query"])

        logger.info(f"Already processed queries loaded: {len(already_processed_queries)} unique queries.")

        # --- 3. Load input JSONL data ---
        input_data = load_jsonl(args.input_jsonl)
        if not input_data:
            logger.error(f"No data loaded from input JSONL: {args.input_jsonl}")
            return
        logger.info(f"Loaded {len(input_data)} entries from {args.input_jsonl}")

        # --- 4. Prepare report texts for LLM processing ---
        queries_to_process_for_llm = set()

        if args.report_generation_mode in [
            "gt_vs_gen_comparison",
            "gt_vs_gen_comparison_with_maira2_grounding"
        ]:
            dicom_id_to_report_map = get_dicom_id_to_report_map()


        skipped_entries_due_to_missing_or_malformed_data = 0
        skipped_entries_due_to_already_processed = 0

        for entry in tqdm(input_data, desc="Preparing reports for LLM"):
            if args.report_generation_mode == "MAIRA-2":
                grounded_report = entry.get("grounded_report")
                if not isinstance(grounded_report, list):
                    skipped_entries_due_to_missing_or_malformed_data += 1
                    logger.warning(f"Skipping entry for due to missing or malformed 'grounded_report'.")
                    continue

                report_sentences = []
                for report_item in grounded_report:
                    if isinstance(report_item, list) and len(report_item) > 0 and isinstance(report_item[0], str):
                        report_sentences.append(report_item[0])
            
                if not report_sentences:
                    skipped_entries_due_to_missing_or_malformed_data += 1
                    logger.warning(f"Skipping entry as no sentences found in 'grounded_report'.")
                    continue

                report_text_for_llm = _concatenate_report_sentences(report_sentences)
            elif args.report_generation_mode == "anatomy_guided_mini_reports":
                report_text_for_llm = entry['parsed_response']['mini-report']
                if report_text_for_llm == 'N/A': # Skip if report is not available
                    continue
            
            elif args.report_generation_mode == "gt_vs_gen_comparison":
                gt_report = dicom_id_to_report_map[entry["dicom_id"]]
                gen_prediction = entry["grounding_prediction"]

                if not gt_report or not isinstance(gt_report, str):
                    skipped_entries_due_to_missing_or_malformed_data += 1
                    logger.warning(f"Skipping entry due to missing or malformed 'gt_report'.")
                    continue # Skip if no ground-truth report

                if not gen_prediction or "description" not in gen_prediction.lower():
                    skipped_entries_due_to_missing_or_malformed_data += 1
                    logger.warning(f"Skipping entry due to missing or malformed 'gen_prediction'.")
                    continue # Skip if no generated description

                # Extract the description part from the grounding_prediction string
                try:
                    gen_report = DESCRIPTION_REGEX.search(gen_prediction).group(1).strip()
                except (AttributeError, IndexError):
                    skipped_entries_due_to_missing_or_malformed_data += 1
                    logger.warning(f"Skipping entry due to missing or malformed 'gen_prediction': {gen_prediction}")
                    continue

                # Format the query for the LLM
                report_text_for_llm = (
                    f"[GT]: {gt_report}\n\n"
                    f"[GEN]: {gen_report}"
                )
            elif args.report_generation_mode == "gt_vs_gen_comparison_with_maira2_grounding":
                gt_report = dicom_id_to_report_map[entry["dicom_id"]]
                gen_prediction = entry["raw_prediction"]

                if not gt_report or not isinstance(gt_report, str):
                    skipped_entries_due_to_missing_or_malformed_data += 1
                    logger.warning(f"Skipping entry due to missing or malformed 'gt_report'.")
                    continue # Skip if no ground-truth report

                gen_report = _parse_maira2_grounding_prediction_to_text(gen_prediction)

                # Format the query for the LLM
                report_text_for_llm = (
                    f"[GT]: {gt_report}\n\n"
                    f"[GEN]: {gen_report}"
                )
            else:
                logger.error(f"Unsupported report generation mode: {args.report_generation_mode}")
                return
            
            if report_text_for_llm in already_processed_queries: # Skip if already processed
                skipped_entries_due_to_already_processed += 1
                continue
            
            queries_to_process_for_llm.add(report_text_for_llm)

            if (args.max_reports_to_process is not None and
                len(queries_to_process_for_llm) >= args.max_reports_to_process):
                logger.info(f"Reached max_reports_to_process limit: {args.max_reports_to_process}")
                break
        
        queries_to_process_for_llm = list(queries_to_process_for_llm) # Convert to list for processing
        
        logger.info(f"Prepared {len(queries_to_process_for_llm)} new reports for LLM labeling.")
        logger.info(f"Skipped {skipped_entries_due_to_missing_or_malformed_data} entries due to missing or malformed data.")
        logger.info(f"Skipped {skipped_entries_due_to_already_processed} entries due to already processed.")

        if not queries_to_process_for_llm:
            logger.info("No new reports to process. Exiting.")
            return

        logger.info("First 10 report texts for LLM:")
        for i, report in enumerate(queries_to_process_for_llm[:10]):
            if len(report) > 200:
                report = report[:200] + "..."
            logger.info(f"Report {i+1}: {report}")
        
    else:

        queries_to_process_for_llm = None
        system_instructions = None

    # --- 5. Define the LLM output parser ---
    if args.llm_output_parser == "paultimothymooney_chest_xray_pneumonia":
        parse_llm_label_output = parse_llm_label_output__paultimothymooney_chest_xray_pneumonia
    elif args.llm_output_parser == "anatomy_guided_mini_reports":
        parse_llm_label_output = parse_llm_label_output__anatomy_guided_mini_reports
    elif args.llm_output_parser == "gt_vs_gen_comparison":
        parse_llm_label_output = parse_llm_label_output__gt_vs_gen_comparison
    else:
        logger.error(f"Unsupported LLM output parser: {args.llm_output_parser}")
        return

    # --- 6. Run LLM API requests using orchestrate_api_calls ---
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
            parse_output=parse_llm_label_output, # This will parse the LLM's string response
            save_filepath=output_file_path, # orchestrate_api_calls saves its direct output here
            delete_api_requests_and_responses=not args.dont_delete_api_requests_and_responses,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            log_info_every_n_requests=args.log_info_every_n_requests,
        )
    except Exception as e:
        logger.error(f"Error during orchestrate_api_calls: {e}", exc_info=True)
        logger.error("Processing halted. LLM output might be available at "
                     f"{output_file_path} if the error was not in file creation.")
        return

if __name__ == "__main__":

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Label MAIRA-2 reports using an LLM.")
    
    # Input/Output
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Path to the input JSONL file containing generated reports.")
    parser.add_argument("--output_labeled_jsonl", type=str, required=True,
                        help="Path to save the output JSONL file with LLM-assigned labels.")
    parser.add_argument("--system_instructions_relative_path", type=str, required=True,
                        help="Relative path to the system instructions file for the LLM. "
                                "This file should be located relative to the LLM_PROMPTS_DIR "
                                "defined in vlm_research_kit.settings.")
    parser.add_argument(
        "--api_responses_filepath",
        type=str,
        default=None,
        help="Path to precomputed API responses JSONL file. Useful when a previous run crashed. "
        "If provided, the script will skip API calls and use these responses.",
    )
    
    # Report Generation Mode
    parser.add_argument("--report_generation_mode", type=str, default="MAIRA-2",
                        choices=[
                            "MAIRA-2", "anatomy_guided_mini_reports", "gt_vs_gen_comparison",
                            "gt_vs_gen_comparison_with_maira2_grounding"
                        ],
                        help="The report generation mode to use.")
    
    # LLM Output Parser
    parser.add_argument("--llm_output_parser", type=str, default="paultimothymooney_chest_xray_pneumonia",
                        choices=[
                            "paultimothymooney_chest_xray_pneumonia",
                            "anatomy_guided_mini_reports",
                            "gt_vs_gen_comparison"
                        ],
                        help="The parser to use for LLM output. "
                                "Currently supports 'paultimothymooney_chest_xray_pneumonia', "
                                "'anatomy_guided_mini_reports', and 'gt_vs_gen_comparison'. "
                                "This determines how the LLM's response is interpreted and structured.")

    # LLM Configuration
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                        help="Name of the LLM model to use (e.g., 'gpt-4-turbo', 'gemini-pro').")
    parser.add_argument("--api_key_name", type=str, default="OPENAI_API_KEY",
                        help="Name of the environment variable holding the API key.")
    parser.add_argument("--api_type", type=str, default="openai", choices=["openai", "gemini"],
                        help="Type of API to use ('openai' or 'gemini').")
    
    # Throttling and Request Parameters (Defaults from your orchestrate_api_calls)
    parser.add_argument("--max_requests_per_minute", type=float, default=3000,
                        help="Maximum API requests per minute.")
    parser.add_argument("--max_tokens_per_minute", type=float, default=200000,
                        help="Maximum tokens per minute for API requests.")
    parser.add_argument("--max_tokens_per_request", type=int, default=2048,
                        help="Maximum tokens per single API request (for LLM completion).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM temperature for generation (0.0 for deterministic labeling).")
    parser.add_argument("--frequency_penalty", type=float, default=0.0, help="LLM frequency penalty.")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="LLM presence penalty.")
    parser.add_argument("--log_info_every_n_requests", type=int, default=50,
                        help="Log info every N API requests in orchestrate_api_calls.")

    # Processing Control
    parser.add_argument("--max_reports_to_process", type=int, default=None,
                        help="Maximum number of new MAIRA reports to process (for testing). Processes all if None.")
    parser.add_argument("--dont_delete_api_requests_and_responses", action="store_true",
                        help="If set, keeps temporary files created by orchestrate_api_calls "
                             "(e.g., the .tmp_llm_out file).")
    
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set the logging level.")

    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce noise from httpx library
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level) # Update log level from args

    label_reports_main(args)
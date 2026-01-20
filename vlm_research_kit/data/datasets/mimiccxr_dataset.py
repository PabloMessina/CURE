import logging
import os
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from PIL import Image

from vlm_research_kit.data.transforms_factory import create_image_transforms
from vlm_research_kit.settings import (
    MIMIC_CXR_IMAGES_DIR,
    MIMIC_CXR_METADATA_CSV_PATH,
    MIMIC_CXR_POSTPROCESSED_REPORTS_JSON_PATH,
    MIMIC_CXR_SPLIT_CSV_PATH,
)
from vlm_research_kit.utils.file_utils import load_json

logger = logging.getLogger(__name__)

# Type hint for the function that tokenizes a batch of reports
BatchTokenizerFn = Callable[[List[str]], Dict[str, torch.Tensor]]


def _extract_parts_from_path(path_str: str) -> Tuple[str, str, str]:
    """
    Extracts partition ID, subject ID, and study ID from a MIMIC-CXR file path.

    Assumes a path format like: ".../p<part_id>/p<subject_id>/s<study_id>.txt"

    Args:
        path_str: The file path string containing the report.

    Returns:
        A tuple containing (part_id, subject_id, study_id) as strings.
    """
    # Example path: "/mnt/data/mimic-cxr/files/p10/p10703179/s58829627.txt"
    match = re.search(r"p(\d+)/p(\d+)/s(\d+)\.txt", path_str)
    if match:
        part_id, subject_id, study_id = match.groups()
        # Return numeric study_id consistent with metadata CSV
        return part_id, subject_id, study_id
    else:
        raise ValueError(
            f"Path string '{path_str}' does not match expected format "
            "'.../p<part_id>/p<subject_id>/s<study_id>.txt'."
        )


def _construct_full_report(
    findings: Optional[str], impression: Optional[str]
) -> str:
    """
    Constructs a combined report string from findings and impression sections.

    Cleans up whitespace, adds a period if missing at the end of a section,
    and joins the sections with a space. Handles cases where one or both
    sections might be None or empty.

    Args:
        findings: The findings section of the report (or None).
        impression: The impression section of the report (or None).

    Returns:
        A single string representing the combined report.
    """
    report_parts = []
    # Process Findings
    if findings and isinstance(findings, str):
        findings = findings.strip()
        if findings: # Ensure non-empty after stripping
            # Add a period if it doesn't end with common punctuation
            if not re.search(r"[.!?,;:]$", findings):
                findings += "."
            report_parts.append(findings)

    # Process Impression
    if impression and isinstance(impression, str):
        impression = impression.strip()
        if impression: # Ensure non-empty after stripping
            # Add a period if it doesn't end with common punctuation
            if not re.search(r"[.!?,;:]$", impression):
                impression += "."
            report_parts.append(impression)

    # Join the parts with a single space
    return " ".join(report_parts)


def _build_image_path(
    images_dir: str, part_id: str, subject_id: str, study_id: str, dicom_id: str
) -> str:
    """
    Constructs the full path to a specific MIMIC-CXR image file.

    Uses the standard MIMIC-CXR directory structure:
    <images_dir>/p<part_id>/p<subject_id>/s<study_id>/<dicom_id>.jpg

    Args:
        images_dir: The base directory where all MIMIC-CXR images are stored.
        part_id: The partition ID (e.g., '10').
        subject_id: The subject ID (e.g., '10703179').
        study_id: The study ID (e.g., '58829627').
        dicom_id: The DICOM ID of the specific image.

    Returns:
        The absolute path to the image file.
    """
    return os.path.join(
        images_dir,
        f"p{part_id}",
        f"p{subject_id}",
        f"s{study_id}",
        f"{dicom_id}.jpg",
    )

def get_dicom_id_to_image_path_map(
    images_dir: str = MIMIC_CXR_IMAGES_DIR,
    postprocessed_reports_json_path: str = MIMIC_CXR_POSTPROCESSED_REPORTS_JSON_PATH,
    split_csv_path: str = MIMIC_CXR_SPLIT_CSV_PATH,
) -> Dict[str, str]:
    """
    Creates a mapping from DICOM IDs to their corresponding image file paths.

    Args:
        images_dir: The root directory where all MIMIC-CXR images are stored.
        postprocessed_reports_json_path: Path to the JSON file containing
            preprocessed report metadata. Each entry should have a 'path' key
            that encodes the partition, subject, and study IDs, which are used
            to reconstruct the image file paths.
        split_csv_path: Path to the CSV file that defines the train/validate/test
            split and contains DICOM, study, and subject IDs.

    Returns:
        A dictionary mapping each DICOM ID (as a string) to its absolute image file path.
    """
    
    # Load the preprocessed report metadata from JSON
    reports_data = load_json(postprocessed_reports_json_path)
    study_id_to_part_id_map = {}
    for report_item in reports_data:
        report_path = report_item["path"]
        # Extract partition ID (part_id) from the report path
        part_id, _, study_id = _extract_parts_from_path(report_path)
        # Map each study ID to its corresponding partition ID
        study_id_to_part_id_map[study_id] = part_id

    # Load the split CSV, which contains DICOM, study, and subject IDs
    split_df = pd.read_csv(split_csv_path)
    dicom_id_to_image_path_map = {}
    for dicom_id, study_id, subject_id in zip(
        split_df["dicom_id"].astype(str),
        split_df["study_id"].astype(str),
        split_df["subject_id"].astype(str),
    ):
        # Retrieve the partition ID for the current study
        part_id = study_id_to_part_id_map[study_id]
        # Construct the absolute image file path using all relevant IDs
        image_path = _build_image_path(
            images_dir, part_id, subject_id, study_id, dicom_id
        )
        # Add the mapping from DICOM ID to image path
        dicom_id_to_image_path_map[dicom_id] = image_path

    # Sanity check: Ensure at least one image path exists on disk
    assert os.path.exists(
        next(iter(dicom_id_to_image_path_map.values()))
    ), "At least one image path does not exist. Check the images directory."

    logger.info(
        f"Built DICOM ID to image path map with {len(dicom_id_to_image_path_map)} entries."
    )
    return dicom_id_to_image_path_map


def get_dicom_id_to_report_map(
    postprocessed_reports_json_path: str = MIMIC_CXR_POSTPROCESSED_REPORTS_JSON_PATH,
    split_csv_path: str = MIMIC_CXR_SPLIT_CSV_PATH,
) -> Dict[str, str]:
    """
    Creates a mapping from DICOM IDs to their corresponding radiology report text.

    Combines 'findings' and 'impression' sections using `_construct_full_report`
    and uses the study_id extracted from `path` to match DICOM images
    in the MIMIC-CXR split CSV file.

    Args:
        postprocessed_reports_json_path: Path to the JSON file containing
            preprocessed report items (with 'path', 'findings', 'impression').
        split_csv_path: CSV file containing DICOM, study, and subject IDs.

    Returns:
        A dictionary mapping DICOM IDs (as strings) to full report text strings.
    """
    # Load report metadata
    reports_data = load_json(postprocessed_reports_json_path)

    # Load split CSV to get DICOM-study mapping
    split_df = pd.read_csv(split_csv_path)
    split_df["dicom_id"] = split_df["dicom_id"].astype(str)
    split_df["study_id"] = split_df["study_id"].astype(str)

    # Build an index from study_id to all dicom_ids under that study
    study_id_to_dicom_ids = (
        split_df.groupby("study_id")["dicom_id"].apply(list).to_dict()
    )

    dicom_id_to_report_map: Dict[str, str] = {}

    # Iterate over all report entries in JSON
    for report_item in reports_data:
        path = report_item["path"]
        _, _, study_id = _extract_parts_from_path(path)
        findings = report_item.get("findings")
        impression = report_item.get("impression")

        # Construct the cleaned full report text
        full_report = _construct_full_report(findings, impression)

        # Get all dicoms for this study
        dicom_ids = study_id_to_dicom_ids.get(study_id, [])
        for dicom_id in dicom_ids:
            dicom_id_to_report_map[dicom_id] = full_report

    logger.info(
        f"Built DICOM ID to report map with {len(dicom_id_to_report_map)} entries."
    )
    return dicom_id_to_report_map

def _extract_study_id_from_path(path_str: str) -> Optional[str]:
    match = re.search(r"/s(\d+)", path_str)
    return match.group(1) if match else None

def get_study_id_to_report_map(
    postprocessed_reports_json_path: str = MIMIC_CXR_POSTPROCESSED_REPORTS_JSON_PATH,
) -> Dict[str, str]:
    """
    Creates a mapping from study IDs to their corresponding radiology report text.
    """
    study_id_to_report_map = {}
    reports_data = load_json(postprocessed_reports_json_path)
    for report_item in reports_data:
        report_path = report_item.get('path')
        study_id = _extract_study_id_from_path(report_path)
        assert study_id is not None, f"Study ID is None for report path: {report_path}"
        full_report = _construct_full_report(
            report_item.get("findings"), report_item.get("impression")
        )
        study_id_to_report_map[str(study_id)] = full_report

    logger.info(
        f"Built study ID to report map with {len(study_id_to_report_map)} entries."
    )
    return study_id_to_report_map


class MIMICCXRDataset(Dataset):
    """
    PyTorch Dataset for loading MIMIC-CXR studies.

    Each item in the dataset corresponds to a single radiology study, which
    may include multiple images and an associated radiology report.

    The dataset can be configured for standard deep learning models or for
    the MAIRA-2 model, which requires specific input formatting.

    Attributes:
        split (str): The data split ('train', 'validate', 'test').
        for_maira2 (bool): Flag to format data for the MAIRA-2 model.
        return_image_paths (bool): Whether to include image paths in the output.
        tokenizer_fn (Optional[BatchTokenizerFn]): Function for tokenizing reports.
        study_ids (List[str]): List of unique study IDs for the split.
        study_id_to_info_map (Dict[str, Dict]): Maps study IDs to their info.
        image_transforms (Optional[Callable]): Image transformation pipeline.
            (None if `for_maira2` is True).
    """

    def __init__(
        self,
        split: str,
        image_loading_mode: str = "standard",
        metadata_csv_path: str = MIMIC_CXR_METADATA_CSV_PATH,
        split_csv_path: str = MIMIC_CXR_SPLIT_CSV_PATH,
        postprocessed_reports_json_path: str = MIMIC_CXR_POSTPROCESSED_REPORTS_JSON_PATH,
        images_dir: str = MIMIC_CXR_IMAGES_DIR,
        return_image_paths: bool = False,
        return_study_info: bool = False,
        image_transforms_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_fn: Optional[BatchTokenizerFn] = None,
        diagnostic_mode: bool = False,
    ):
        """
        Initializes the MIMICCXRDataset.

        Args:
            split: The dataset split to load ('train', 'validate', 'test').
            image_loading_mode: The image loading strategy. One of:
                'standard': Load all images for a study.
                'maira2': Load one random frontal and one random lateral PIL image.
                'frontal_only': Load a single frontal image (PA > AP priority).
                Defaults to "standard".
            metadata_csv_path: Path to the MIMIC-CXR metadata CSV file.
            split_csv_path: Path to the CSV file defining data splits.
            postprocessed_reports_json_path: Path to the preprocessed reports JSON.
            images_dir: Root directory for MIMIC-CXR image files.
            return_image_paths: If True, include image paths in the output.
            return_study_info: If True, include study metadata in the output.
            image_transforms_kwargs: Keyword arguments for image transforms.
                Ignored if `for_maira2` is True.
            tokenizer_fn: Optional function to tokenize reports in `collate_fn`.
            diagnostic_mode: If True, include diagnostic information in the output.
        """
        super().__init__()

        if image_loading_mode == "maira2":
            for_maira2 = True
        elif image_loading_mode == "frontal_only":
            for_maira2 = False
        elif image_loading_mode == "standard":
            for_maira2 = False
            assert image_transforms_kwargs is not None, (
                "image_transforms_kwargs must be provided for 'standard' mode."
            )
        else:
            raise ValueError(
                f"Invalid image_loading_mode '{image_loading_mode}'. Must be "
                "'standard', 'maira2', or 'frontal_only'."
            )

        if split not in ["train", "validate", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be 'train', 'validate', or 'test'."
            )

        self.split = split
        self.return_image_paths = return_image_paths
        self.return_study_info = return_study_info
        self.tokenizer_fn = tokenizer_fn
        self.image_loading_mode = image_loading_mode
        self.for_maira2 = for_maira2
        self.image_transforms = None
        self.diagnostic_mode = diagnostic_mode
        self.dataset_name = "mimiccxr_reportgen"
        
        if self.diagnostic_mode:
            logger.info("Running in diagnostic mode.")

        # --- Load Data from CSV and JSON ---
        logger.info(f"Loading metadata from {metadata_csv_path}...")
        metadata_df = pd.read_csv(metadata_csv_path)
        logger.info(f"Loaded metadata with {len(metadata_df)} records.")

        logger.info(f"Loading split info from {split_csv_path}...")
        split_df = pd.read_csv(split_csv_path)
        logger.info(f"Loaded split info with {len(split_df)} records.")

        logger.info(f"Loading reports from {postprocessed_reports_json_path}...")
        reports_data = load_json(postprocessed_reports_json_path)
        logger.info(f"Loaded reports with {len(reports_data)} records.")

        # --- Filter by Split ---
        split_df = split_df[split_df["split"] == self.split].copy()
        assert not split_df.empty, f"No data for split '{self.split}'."
        logger.info(f"Filtered for '{self.split}' split: {len(split_df)} records.")

        study_ids_in_split = split_df["study_id"].unique().astype(str).tolist()
        study_ids_in_split_set = set(study_ids_in_split)
        self.study_ids = study_ids_in_split

        # --- Map Study IDs to Reports ---
        logger.info("Mapping study IDs to reports...")
        skipped_reports = 0
        study_id_to_info_map = {}
        for report_item in reports_data:
            part_id, subject_id, study_id = _extract_parts_from_path(
                report_item["path"]
            )
            if study_id not in study_ids_in_split_set:
                continue
            full_report = _construct_full_report(
                report_item["findings"], report_item["impression"]
            )
            if not full_report:
                skipped_reports += 1
                continue
            study_id_to_info_map[study_id] = {
                "full_report": full_report,
                "part_id": part_id,
                "subject_id": subject_id,
                "study_id": study_id,
            }
            
        if skipped_reports > 0:
            logger.warning(f"Skipped {skipped_reports}/{len(reports_data)} reports due to empty full reports.")
            logger.warning(f"Skipped reports percentage: {skipped_reports / len(reports_data) * 100}%")
            
        self.study_id_to_info_map = study_id_to_info_map

        # --- Create Image Transforms ---
        if image_transforms_kwargs is not None:
            logger.info("Creating image transforms...")
            self.image_transforms = create_image_transforms(
                **image_transforms_kwargs
            )

        # --- Collect Image Paths per Study (Mode-dependent) ---
        if self.image_loading_mode == "maira2":
            self._collect_image_paths_for_maira2(
                split_df, metadata_df, images_dir
            )
        elif self.image_loading_mode == "frontal_only":
            # Call the new method to handle frontal-only logic
            self._collect_image_paths_frontal_only(
                split_df, metadata_df, images_dir
            )
        else: # "standard" mode
            self._collect_image_paths_standard(split_df, images_dir)

        # Create indices list after all image paths are collected
        self.indices = list(range(len(self.study_ids))) # List of indices for the study IDs            

    def _collect_image_paths_standard(self, split_df: pd.DataFrame, images_dir: str):
        logger.info("Collecting standard image paths for each study...")
        for study_id in self.study_ids:
            self.study_id_to_info_map[study_id]["image_paths"] = []

        for _, row in tqdm(
            split_df.iterrows(),
            desc="Collecting image paths",
            total=len(split_df),
        ):
            study_id = str(row["study_id"])
            study_info = self.study_id_to_info_map[study_id]
            image_path = _build_image_path(
                images_dir,
                study_info["part_id"],
                str(row["subject_id"]),
                study_id,
                str(row["dicom_id"]),
            )
            study_info["image_paths"].append(image_path)

    def _collect_image_paths_frontal_only(
        self, split_df: pd.DataFrame, metadata_df: pd.DataFrame, images_dir: str
    ):
        """
        Collects a single frontal image path for each study, prioritizing PA views.

        Studies without any frontal views (PA or AP) are filtered out from the dataset.
        """
        logger.info("Collecting single frontal image paths (PA > AP priority)...")
        FRONTAL_VIEWS_PA = {"PA"}
        FRONTAL_VIEWS_AP = {"AP"}

        # Merge with metadata to get ViewPosition
        metadata_views_df = metadata_df[["dicom_id", "ViewPosition", "ProcedureCodeSequence_CodeMeaning"]].copy()
        merged_df = pd.merge(
            split_df.astype(str),
            metadata_views_df.astype(str),
            on="dicom_id",
            how="left",
        )

        # Group image paths by study and view type (PA or AP)
        study_to_frontal_paths = {}
        for study_id in self.study_ids:
            study_to_frontal_paths[study_id] = {"PA": [], "AP": []}

        for _, row in tqdm(
            merged_df.iterrows(),
            desc="Categorizing frontal paths",
            total=len(merged_df),
        ):
            study_id = row["study_id"]
            if study_id not in self.study_id_to_info_map:
                continue

            study_info = self.study_id_to_info_map[study_id]
            image_path = _build_image_path(
                images_dir,
                study_info["part_id"],
                row["subject_id"],
                study_id,
                row["dicom_id"],
            )

            view = row["ViewPosition"]
            if view in FRONTAL_VIEWS_PA:
                study_to_frontal_paths[study_id]["PA"].append(image_path)
            elif view in FRONTAL_VIEWS_AP:
                study_to_frontal_paths[study_id]["AP"].append(image_path)
            elif "AP" in row["ProcedureCodeSequence_CodeMeaning"]:
                study_to_frontal_paths[study_id]["AP"].append(image_path)

        # Select one path per study and filter out studies with no frontal images
        final_study_ids = []
        for study_id in self.study_ids:
            pa_paths = study_to_frontal_paths[study_id]["PA"]
            ap_paths = study_to_frontal_paths[study_id]["AP"]

            selected_path = None
            if pa_paths:
                selected_path = pa_paths[0]  # Prioritize PA
            elif ap_paths:
                selected_path = ap_paths[0]  # Fallback to AP

            if selected_path:
                self.study_id_to_info_map[study_id][
                    "image_path"
                ] = selected_path
                final_study_ids.append(study_id)

        original_count = len(self.study_ids)
        self.study_ids = final_study_ids
        logger.info(
            f"Filtered for studies with frontal views. Kept {len(self.study_ids)} "
            f"out of {original_count} studies."
        )

    def _collect_image_paths_for_maira2(
        self, split_df, metadata_df, images_dir
    ):
        logger.info("Collecting frontal/lateral image paths for MAIRA-2...")
        FRONTAL_VIEWS = {"PA", "AP"}
        LATERAL_VIEWS = {"LATERAL", "LL"}

        metadata_views_df = metadata_df[["dicom_id", "ViewPosition", "ProcedureCodeSequence_CodeMeaning"]].copy()
        merged_df = pd.merge(
            split_df.astype(str),
            metadata_views_df.astype(str),
            on="dicom_id",
            how="left",
        )

        for study_id in self.study_ids:
            info = self.study_id_to_info_map[study_id]
            info["frontal_image_paths"] = []
            info["lateral_image_paths"] = []

        for _, row in tqdm(
            merged_df.iterrows(),
            desc="Categorizing MAIRA-2 paths",
            total=len(merged_df),
        ):
            study_id = row["study_id"]
            study_info = self.study_id_to_info_map[study_id]
            image_path = _build_image_path(
                images_dir,
                study_info["part_id"],
                row["subject_id"],
                study_id,
                row["dicom_id"],
            )

            view = row["ViewPosition"]
            if view in FRONTAL_VIEWS:
                study_info["frontal_image_paths"].append(image_path)
            elif view in LATERAL_VIEWS:
                study_info["lateral_image_paths"].append(image_path)
            elif "AP" in row["ProcedureCodeSequence_CodeMeaning"]:
                study_info["frontal_image_paths"].append(image_path)

        # Filter out studies with neither frontal nor lateral images
        original_count = len(self.study_ids)
        final_study_ids = []
        for study_id in self.study_ids:
            study_info = self.study_id_to_info_map[study_id]
            if study_info["frontal_image_paths"] or study_info["lateral_image_paths"]:
                final_study_ids.append(study_id)
        self.study_ids = final_study_ids
        logger.info(
            f"Filtered for studies with frontal or lateral views. Kept {len(self.study_ids)} "
            f"out of {original_count} studies."
        )

    def shuffle_indices(self):
        """Shuffle the indices of the dataset."""
        random.shuffle(self.indices) # Shuffle the indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int, skip_image_loading: bool = False) -> Dict[str, Any]:
        if not 0 <= idx < len(self.indices):
            raise IndexError(f"Index {idx} out of bounds for size {len(self)}")

        if self.diagnostic_mode:
            orig_idx = idx # Store the original index for diagnostic purposes
        idx = self.indices[idx] # Get the index of the study in the dataset

        study_id = self.study_ids[idx]
        study_info = self.study_id_to_info_map[study_id]

        if self.image_loading_mode == "maira2":
            # --- MAIRA-2 Mode: Return PIL Images ---
            frontal_paths = study_info.get("frontal_image_paths", [])
            lateral_paths = study_info.get("lateral_image_paths", [])

            frontal_path = random.choice(frontal_paths) if frontal_paths else None
            lateral_path = random.choice(lateral_paths) if lateral_paths else None

            output = {
                "report": study_info["full_report"],
            }

            if not skip_image_loading:
                frontal_image = (
                    Image.open(frontal_path).convert("RGB") if frontal_path else None
                )
                lateral_image = (
                    Image.open(lateral_path).convert("RGB") if lateral_path else None
                )
                output["frontal_image"] = frontal_image
                output["lateral_image"] = lateral_image
            if self.return_image_paths:
                output["frontal_image_path"] = frontal_path
                output["lateral_image_path"] = lateral_path
            if self.return_study_info:
                output["study_info"] = study_info
            return output

        elif self.image_loading_mode == "frontal_only":
            # --- Frontal-Only Mode: Return single transformed image ---
            image_path = study_info["image_path"]

            transformed_image = None
            
            if not skip_image_loading:
                if self.image_transforms is not None:
                    # Transforms might return a PIL Image or a Tensor
                    transformed_image = self.image_transforms(image_path)
                    if isinstance(transformed_image, dict):
                        transformed_image = transformed_image["pixel_values"]
                else:
                    transformed_image = Image.open(image_path).convert("RGB") # Default behavior: No transforms, load with PIL
            
            output = {
                "image": transformed_image,
                "report": study_info["full_report"],
            }
            if self.return_image_paths:
                output["image_path"] = image_path
            if self.return_study_info:
                output["study_info"] = study_info
            return output

        else:
            # --- Standard Mode: Return Transformed Tensors ---
            image_paths = study_info["image_paths"]
            
            pixel_values = None
            if not skip_image_loading:
                image_tensors = [self.image_transforms(p) for p in image_paths]
                if isinstance(image_tensors[0], dict): # Check if transforms return dicts
                    image_tensors = [x["pixel_values"] for x in image_tensors]
                pixel_values = torch.stack(image_tensors, dim=0) # Shape: N_i x C x H x W

            output = {"pixel_values": pixel_values, "report": study_info["full_report"]}
            if self.return_image_paths:
                output["image_paths"] = image_paths
            if self.return_study_info:
                output["study_info"] = study_info
            if self.diagnostic_mode:
                output["diagnostic_orig_index"] = orig_idx
                output["diagnostic_actual_index"] = idx
                output["diagnostic_dataset_name"] = self.dataset_name
                output["diagnostic_category"] = "unknown" # TODO: Add category
            return output

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.image_loading_mode == "maira2":
            # ==== MAIRA-2 Collation: Group items into lists ====
            output = {}
            # Handle cases where a batch might have items with missing keys
            all_keys = set().union(*(d.keys() for d in batch))
            for key in all_keys:
                output[key] = [d.get(key) for d in batch]

            # Rename 'report' to 'reports' for consistency
            if "report" in output:
                output["reports"] = output.pop("report")
            
            return output

        assert self.image_loading_mode == "standard", "Standard mode is the only supported mode for this collate function."

        # ==== Standard Collation: Pad tensors ====
        output = {}

        # --- Batch Pixel Values ---
        # Input: List of tensors [(N1, C, H, W), (N2, C, H, W), ...]
        # Output: Tensor (batch_size, max_N, C, H, W)
        # Pad the pixel values to the maximum length in the batch
        pixel_values_list = [item["pixel_values"] for item in batch]
        pixel_values_batch = pad_sequence(pixel_values_list, batch_first=True, padding_value=0.0)
        output["pixel_values"] = pixel_values_batch

        # --- Batch Reports ---
        reports = [item['report'] for item in batch]
        # Apply Tokenization (if function provided)
        if self.tokenizer_fn is not None:
            try:
                # Call the provided function with the list of report strings
                tokenized_data = self.tokenizer_fn(reports)

                if not isinstance(tokenized_data, dict):
                    raise TypeError(
                        f"The provided tokenizer_fn was expected to return a dict, "
                        f"but got {type(tokenized_data)}."
                    )
                # Merge the tokenized tensors into the main batch dictionary
                output.update(tokenized_data)
            except Exception as e:
                logger.error(f"Error executing tokenizer_fn in collate: {e}", exc_info=True)
                raise e
        else:
            # If no tokenizer function, keep the raw reports (for inference)
            output["reports"] = reports

        # --- Batch Image Paths (if requested) ---
        if self.return_image_paths:
            output["image_paths"] = [item['image_paths'] for item in batch]

        # --- Batch Study Info (if requested) ---
        if self.return_study_info:
            output["study_info"] = [item['study_info'] for item in batch]

        return output
import logging
import os
import random
import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm_research_kit.settings import (
    PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH,
    PADCHEST_GR_MASTER_TABLE_CSV_PATH,
    PADCHEST_GR_JPG_DIR,
    PADCHEST_GR_PROGRESSION_PRIOR_STUDIES_JPG_DIR,
)
from vlm_research_kit.data.dataset_helpers import WeightedCompositeDataset, create_uniform_subset_from_indices_list
from vlm_research_kit.data.transforms_factory import create_image_transforms
from vlm_research_kit.utils.bbox_utils import xyxy_to_cxcywh
from vlm_research_kit.utils.file_utils import load_json

logger = logging.getLogger(__name__)

# Type hint for the function that tokenizes a batch of reports
BatchTokenizerFn = Callable[[List[str]], Dict[str, torch.Tensor]]

def _format_bboxes(
    bboxes: List[List[float]],
    decimal_places: int = 2,
) -> List[str]:
    """
    Converts bounding boxes (assumed to be normalized 0-1) to strings.

    Args:
        bboxes: List of bounding boxes, normalized to [0, 1] (e.g., [[0.45, 0.45, 0.55, 0.55]]).
        decimal_places: Number of decimal places to round to (default: 2).

    Returns:
        List of formatted bounding box strings, e.g., ["[0.45,0.45,0.55,0.55]"].
    """
    formatted_strings = []
    dp = decimal_places
    for box in bboxes:
        a, b, c, d = box
        formatted_strings.append(f"[{a:.{dp}f},{b:.{dp}f},{c:.{dp}f},{d:.{dp}f}]")

    return formatted_strings


class PadChestGRDataset(Dataset):
    """
    PyTorch Dataset for the PadChest-GR (Grounded Reporting) dataset.

    This dataset links chest X-ray images to their corresponding radiological
    reports. It supports generating reports either as plain text or grounded
    with bounding box annotations linked to specific sentences.

    Key Features:
    - Loads images and associated report data based on study IDs.
    - Filters data according to specified splits ('train', 'validation', 'test').
    - Optionally generates grounded reports where sentences mentioning specific
      findings are annotated with discretized bounding box coordinates.
    - Applies image transformations using a provided callable, which can handle
      bounding box augmentation during training if grounding is enabled.
    - Assumes a single current image per study, ignoring prior studies.
    - Handles different languages ('en', 'es') for reports.
    """

    def __init__(
        self,
        image_transforms_kwargs: Optional[Dict[str, Any]] = None,
        image_transforms: Optional[Callable] = None,
        json_path: str = PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH,
        csv_path: str = PADCHEST_GR_MASTER_TABLE_CSV_PATH,
        img_dir: str = PADCHEST_GR_JPG_DIR,
        prior_img_dir: str = PADCHEST_GR_PROGRESSION_PRIOR_STUDIES_JPG_DIR,
        split: Literal["train", "validation", "test", "all"] = "train",
        grounded: bool = True,
        language: Literal["en", "es"] = "en",
        bbox_format: Optional[str] = "cxcywh",
        image_format: Literal["jpg", "png"] = "jpg",
        return_image_path: bool = False,
        tokenizer_fn: Optional[BatchTokenizerFn] = None,
        diagnostic_mode: bool = False,
    ):
        """
        Initializes the PadChestGRDataset.

        Args:
            image_transforms_kwargs:
                Dictionary of keyword arguments used to build the image
                transformation function via `create_image_transforms`.
                The created function is expected to meet the following criteria:
                - If `grounded` is False, it receives the image path (str) and
                  should return the processed image tensor (e.g., `pixel_values`).
                - If `grounded` is True, it receives a dictionary:
                  {
                      'image_path': str,
                      'bboxes': List[List[float]] (normalized, in `bbox_format`),
                      'bbox_labels': List[int] (sentence indices)
                  }
                  It must return a dictionary containing:
                  {
                      'pixel_values': torch.Tensor,
                      'bboxes': List[List[float]] (potentially augmented),
                      'bbox_labels': List[int] (corresponding labels)
                  }
            image_transforms: Optional callable for custom image transformations.
                    If provided, it should match the expected input/output formats
                    described above. If None, `create_image_transforms` will be used
                    with the provided `image_transforms_kwargs`.
            json_path: Path to the JSON file containing report findings and
                       bounding boxes (e.g., 'PadChest-GR_grounded_reports.json').
                       Assumes bounding boxes in the JSON are in xyxy format.
            csv_path: Path to the master CSV file containing study metadata and
                      split information (e.g., 'PadChest-GR_master_table.csv').
            img_dir: Path to the directory containing the primary JPG/PNG images.
            prior_img_dir: Path to the directory containing prior study images
                           (currently unused but checked for existence).
            split: The dataset split to load ('train', 'validation', 'test', 'all').
            grounded: If True, load bounding boxes and generate grounded reports.
                      If False, generate plain text reports without box annotations.
            language: The language of the reports to use ('en' or 'es').
            bbox_format: The bounding box format expected by `image_transforms`
                         when `grounded` is True (e.g., 'cxcywh', 'xyxy').
                         The dataset internally handles conversion from the source
                         JSON's xyxy format if needed before passing to transforms.
                         Coordinates are always normalized [0, 1].
            image_format: The file extension of the images in `img_dir` ('jpg' or 'png').
            return_image_path: If True, include the absolute image path in the
                               output dictionary returned by `__getitem__`.
            tokenizer_fn: Optional function to tokenize a batch of reports. Used
                            primarily for tokenization during training, implementing
                            model-specific tokenization logic.
            diagnostic_mode: If True, include diagnostic information in the output.
        """
        super().__init__()

        # --- Input Validation ---
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.isdir(img_dir):
            raise NotADirectoryError(f"Image directory not found: {img_dir}")
        if not os.path.isdir(prior_img_dir):
            raise NotADirectoryError(f"Prior image directory not found: {prior_img_dir}")
        assert split in ["train", "validation", "test", "all"], \
            f"Invalid split '{split}'. Must be one of ['train', 'validation', 'test', 'all']."
        assert language in ["en", "es"], \
            f"Invalid language '{language}'. Must be one of ['en', 'es']."
        assert bbox_format in ["cxcywh", "xyxy"], \
            f"Invalid bbox_format '{bbox_format}'. Must be one of ['cxcywh', 'xyxy']."

        self.img_dir = img_dir
        self.prior_img_dir = prior_img_dir # Stored but not used in __getitem__
        
        if image_transforms is not None:
            self.image_transforms = image_transforms
            logger.info("Using provided image_transforms callable for processing images.")
        elif image_transforms_kwargs:
            self.image_transforms = create_image_transforms(**image_transforms_kwargs)
            logger.info("Using create_image_transforms with provided kwargs for image processing.")
        else:
            self.image_transforms = None
            logger.info(
                "No image_transforms or kwargs provided. "
                "Images will be loaded as PIL.Image objects."
            )
            if grounded:
                logger.warning(
                    "Running in 'grounded' mode without image transforms. "
                    "Bounding boxes will not be augmented."
                )

        self.split = split
        self.grounded = grounded
        self.language = language
        self.bbox_format = bbox_format
        self.image_format = image_format
        self.return_image_path = return_image_path
        self.tokenizer_fn = tokenizer_fn
        self.diagnostic_mode = diagnostic_mode
        self.dataset_name = "padchest_gr_grg" # Grounded Report Generation
        
        if self.diagnostic_mode:
            logger.info("Running in diagnostic mode.")
        
        # --- Load and Filter Metadata ---
        logger.info(f"Loading master CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Filtering dataset for split: {split}")
        # Filter based on the specified split
        if split == "all":
            self.metadata_df = df.copy()
        else:
            self.metadata_df = df[df['split'] == split].copy()
        if self.metadata_df.empty:
            raise ValueError(f"No data found for split '{split}' in {csv_path}")

        # Get unique StudyIDs for the selected split
        self.study_ids = self.metadata_df['StudyID'].unique().tolist()
        logger.info(
            f"Found {len(self.study_ids)} unique studies for split '{split}'"
        )

        # Create indices list
        self.indices = list(range(len(self.study_ids)))

        # --- Load Reports Data ---
        logger.info(f"Loading reports JSON from: {json_path}")
        reports_json_list = load_json(json_path)
        # Index reports by StudyID for efficient lookup
        self.reports_data = {
            item['StudyID']: item for item in reports_json_list
        }
        logger.info(f"Loaded {len(self.reports_data)} reports from JSON.")

        # --- Clean Sentences ---
        logger.info("Cleaning sentences in reports.")
        for report_info in self.reports_data.values():
            findings = report_info.get('findings', [])
            for finding in findings:
                s_en = finding['sentence_en']
                s_es = finding['sentence_es']
                s_en = " ".join(s_en.split()) # Remove extra spaces
                if s_en.endswith('.'):
                    s_en = s_en[:-1]
                s_es = " ".join(s_es.split()) # Remove extra spaces
                if s_es.endswith('.'):
                    s_es = s_es[:-1]
                finding['sentence_en'] = s_en
                finding['sentence_es'] = s_es

        # --- Pre-process Bounding Boxes (if grounded) ---
        # Convert bounding boxes from xyxy (source format) to the format
        # required by the image_transforms function, if necessary.
        # This conversion happens once during initialization.
        if self.grounded and self.bbox_format == "cxcywh":
            logger.info("Converting source bounding boxes (xyxy) to 'cxcywh' format.")
            for report_info in self.reports_data.values():
                findings = report_info.get('findings', [])
                for finding in findings:
                    original_boxes_xyxy = finding.get('boxes') # Assumed xyxy
                    if original_boxes_xyxy:
                        # Convert each box to cxcywh format
                        cxcywh_boxes = [xyxy_to_cxcywh(box) for box in original_boxes_xyxy]
                        finding['boxes'] = cxcywh_boxes # Overwrite with converted boxes

    def shuffle_indices(self):
        """
        Shuffles the indices of the dataset.
        """
        self.indices = list(range(len(self.study_ids))) # Create a new list of indices to shuffle
        random.shuffle(self.indices) # Shuffle the indices

    def create_uniform_subset(self, target_size: int, ensure_multiple_of: int = 1):
        """
        Alters the dataset to be a uniformly sampled subset of the original data.
        A new random subset is generated each time this method is called.

        Args:
            target_size: The desired approximate size of the validation subset.
            ensure_multiple_of: The number to ensure the target size is a multiple of.
        """
        target_size = math.ceil(target_size / ensure_multiple_of) * ensure_multiple_of # Ensure target size is a multiple of ensure_multiple_of
        target_size = min(target_size, len(self.study_ids)) # Ensure target size is at most the size of the dataset
        new_indices = random.sample(range(len(self.study_ids)), target_size) # Sample the indices
        self.indices = new_indices # Overwrite the indices with the new indices

    def __len__(self) -> int:
        """Returns the number of unique indices in the dataset split."""
        return len(self.indices)

    def __getitem__(self, idx: int, skip_image_loading: bool = False) -> Dict[str, Any]:
        """
        Retrieves a single data sample (image, report) for a given study index.

        Args:
            idx: The index of the study within the current split.
            skip_image_loading: If True, skip loading the image.
        Returns:
            A dictionary containing:
            - 'image': The processed image tensor or PIL Image.
            - 'report': The generated report string. If `grounded` is True,
                        this string includes bounding box annotations like
                        "[ (cx,cy,w,h), ... ]" appended to relevant sentences.
            - 'image_path': (Optional) The absolute path to the image file,
                            included if `return_image_path` is True.
        """
        if self.diagnostic_mode:
            orig_idx = idx # Store the original index for diagnostic purposes

        idx = self.indices[idx] # Get the index of the study in the dataset

        study_id = self.study_ids[idx]

        # Retrieve pre-loaded report data for the study
        report_info = self.reports_data[study_id]

        # Get image ID and construct the full image path
        image_id = report_info['ImageID']
        # Ensure correct file extension based on image_format
        base_name, _ = os.path.splitext(image_id)
        image_id_with_ext = f"{base_name}.{self.image_format}"
        image_path = os.path.join(self.img_dir, image_id_with_ext)

        # Extract findings (sentences and potentially boxes)
        findings = report_info['findings']
        lang_key = f'sentence_{self.language}' # Key for accessing sentence text

        # --- Prepare data for image transformation and report generation ---
        report_sentences = [] # Stores the text of each sentence
        bboxes_for_transform = [] # Stores boxes to pass to transforms (if grounded)
        bbox_labels_for_transform = [] # Stores corresponding sentence indices (if grounded)

        for sent_idx, finding in enumerate(findings):
            sentence_text = finding[lang_key]
            report_sentences.append(sentence_text) # Store sentence text

            if self.grounded:
                # Retrieve boxes (already converted to target format in __init__)
                sentence_boxes = finding.get('boxes')
                if sentence_boxes:
                    bboxes_for_transform.extend(sentence_boxes)
                    # Use the current sentence index as the label for these boxes
                    bbox_labels_for_transform.extend([sent_idx] * len(sentence_boxes))

        # --- Apply Image Transformations ---
        image: Optional[Union[torch.Tensor, Image.Image]] = None
        augmented_bboxes = [] # BBoxes after potential augmentation
        augmented_bbox_labels = [] # Corresponding labels after augmentation

        if not skip_image_loading:
            if self.image_transforms:
                # --- Use provided image transforms ---
                if self.grounded:
                    transform_input = {
                        'image_path': image_path,
                        'bboxes': bboxes_for_transform,
                        'bbox_labels': bbox_labels_for_transform,
                    }
                    transform_output = self.image_transforms(**transform_input)
                    image = transform_output['pixel_values'] # Still get pixel_values from transform
                    augmented_bboxes = transform_output['bboxes']
                    augmented_bbox_labels = [
                        int(label) for label in transform_output['bbox_labels']
                    ]
                else: # Non-grounded mode with transforms
                    image = self.image_transforms(image_path)
                    if isinstance(image, dict):
                        image = image['pixel_values']

            else:
                # --- Default behavior: No transforms, load with PIL ---
                image = Image.open(image_path).convert("RGB")
                if self.grounded:
                    # No augmentation is possible, so use original boxes/labels
                    augmented_bboxes = bboxes_for_transform
                    augmented_bbox_labels = bbox_labels_for_transform

        # --- Format BBoxes for Report String (if grounded) ---
        if self.grounded:
            bboxes_per_sentence = [[] for _ in range(len(report_sentences))]
            for box, label in zip(augmented_bboxes, augmented_bbox_labels):
                bboxes_per_sentence[label].append(box)
            for sent_idx, boxes in enumerate(bboxes_per_sentence):
                if boxes:
                    formatted_box_strings = _format_bboxes(boxes)
                    box_annotation = " ".join(formatted_box_strings)
                    report_sentences[sent_idx] += f" {box_annotation}"

        # --- Final Report Construction ---
        final_report = ". ".join(report_sentences).strip()

        # --- Prepare Output Dictionary ---
        output = {
            "image": image,
            "report": final_report,
        }
        if self.return_image_path:
            output["image_path"] = image_path
        if self.diagnostic_mode:
            output["diagnostic_orig_index"] = orig_idx
            output["diagnostic_actual_index"] = idx
            output["diagnostic_dataset_name"] = self.dataset_name
            output["diagnostic_category"] = "unknown" # TODO: Add category

        return output
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function to handle batching of data samples.

        Args:
            batch: A list of dictionaries, each representing a single data sample.

        Returns:
            A dictionary containing batched data.
            - 'pixel_values': Batched image tensors (torch.Tensor).
            - 'reports': Batched report strings (List[str]).
            - 'image_paths': (Optional) Batched image paths (List[str]),
                             included if `return_image_path` is True.
        """
        # Batch pixel values
        pixel_values = torch.stack([item['pixel_values'] for item in batch]) # shape: [B, C, H, W]
        pixel_values = pixel_values.unsqueeze(1) # shape: [B, 1, C, H, W] to support models expecting
        # multiple images per study (e.g., 1 current + 1 prior studies, or frontal + lateral views)
        
        # Gather raw reports
        reports = [item['report'] for item in batch]

        # Initialize output dictionary
        output = {"pixel_values": pixel_values}

         # --- Apply Tokenization (if function provided) ---
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

        if self.return_image_path:
            image_paths = [item['image_path'] for item in batch]
            output["image_paths"] = image_paths

        return output    


class PadChestGRPhraseGroundingDataset(Dataset):
    """
    PyTorch Dataset for PadChest-GR phrase grounding evaluation.

    Each item in this dataset corresponds to a single image paired with a
    single sentence (phrase) that has bounding box annotations.
    It's designed to provide data for evaluating phrase grounding models.
    """

    def __init__(
        self,
        json_path: str = PADCHEST_GR_GROUNDED_REPORTS_JSON_PATH,
        csv_path: str = PADCHEST_GR_MASTER_TABLE_CSV_PATH,
        img_dir: str = PADCHEST_GR_JPG_DIR,
        split: Literal["train", "validation", "test", "all"] = "validation",
        language: Literal["en", "es"] = "en",
        gt_bbox_format: Literal["xyxy", "cxcywh"] = "xyxy",
        image_format: Literal["jpg", "png"] = "jpg",
        image_transforms_kwargs: Optional[Dict[str, Any]] = None,
        include_labels_as_phrases: bool = False,
        return_image_path: bool = False,
        use_weighted_sampling: bool = False,
        diagnostic_mode: bool = False,
    ):
        """
        Initializes the PadChestGRPhraseGroundingDataset.

        Args:
            json_path: Path to the JSON file containing report findings and
                       bounding boxes (e.g., 'PadChest-GR_grounded_reports.json').
                       Assumes bounding boxes in the JSON are in xyxy format.
            csv_path: Path to the master CSV file containing study metadata and
                      split information (e.g., 'PadChest-GR_master_table.csv').
            img_dir: Path to the directory containing the primary JPG/PNG images.
            split: The dataset split to load ('train', 'validation', 'test', 'all').
            language: The language of the reports to use ('en' or 'es').
            gt_bbox_format: The format for the ground truth bounding boxes
                            returned by the dataset ('xyxy' or 'cxcywh').
                            Coordinates are always normalized [0, 1].
            image_format: The file extension of the images in `img_dir`.
            image_transforms_kwargs: Optional dictionary of keyword arguments
                                     to build an image transformation function.
                                     If None, images are loaded as PIL Images.
            include_labels_as_phrases: If True, use finding labels as additional phrases.
            return_image_path: If True, include the absolute image path in the
                               output dictionary.
            use_weighted_sampling: If True, use weighted sampling to balance the dataset.
            diagnostic_mode: If True, include diagnostic information in the output.
        """
        super().__init__()

        # --- Input Validation ---
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.isdir(img_dir):
            raise NotADirectoryError(f"Image directory not found: {img_dir}")
        assert split in ["train", "validation", "test", "all"], \
            f"Invalid split '{split}'. Must be one of ['train', 'validation', 'test', 'all']."
        assert language in ["en", "es"], \
            f"Invalid language '{language}'. Must be one of ['en', 'es']."
        assert gt_bbox_format in ["xyxy", "cxcywh"], \
            f"Invalid gt_bbox_format '{gt_bbox_format}'. Must be ['xyxy', 'cxcywh']."

        self.img_dir = img_dir
        self.split = split
        self.language = language
        self.gt_bbox_format = gt_bbox_format
        self.image_format = image_format
        self.return_image_path = return_image_path
        self.use_weighted_sampling = use_weighted_sampling
        self.diagnostic_mode = diagnostic_mode
        self.image_transforms = None
        if image_transforms_kwargs:
            self.image_transforms = create_image_transforms(**image_transforms_kwargs)

        self.dataset_name = "padchest_gr_pg" # Phrase Grounding

        if self.diagnostic_mode:
            logger.info("Running in diagnostic mode.")

        # --- Attributes for subsampling ---
        self.active_indices: Optional[List[int]] = None
        self.instances_by_group: Dict[str, List[int]] = defaultdict(list)

        # --- Load and Filter Metadata ---
        logger.info(f"Loading master CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Filtering dataset for split: {split}")

        if split == "all":
            metadata_df = df.copy()
        else:
            metadata_df = df[df['split'] == split].copy()
        if metadata_df.empty:
            raise ValueError(f"No data found for split '{split}' in {csv_path}")

        study_ids_for_split = metadata_df['StudyID'].unique().tolist()
        logger.info(
            f"Found {len(study_ids_for_split)} unique studies for split '{split}'"
        )

        # Map label to label_group
        label_to_group = { label: group for label, group in zip(
            df['label'].tolist(), df['label_group'].tolist()
        ) }

        # --- Load Reports Data ---
        logger.info(f"Loading reports JSON from: {json_path}")
        reports_json_list = load_json(json_path)
        reports_data_map = {
            item['StudyID']: item for item in reports_json_list
        }
        logger.info(f"Loaded {len(reports_data_map)} reports from JSON.")

        # --- Create a flat list of grounding instances ---
        self.grounding_instances = []
        lang_key = f'sentence_{self.language}'
        logger.info(
            f"Processing reports to extract grounded sentences for split '{split}'..."
        )

        num_sentence_phrases = 0
        num_label_phrases = 0

        for study_id in study_ids_for_split:
            report_info = reports_data_map.get(study_id)
            if not report_info:
                logger.warning(f"No report data found for StudyID: {study_id}")
                continue

            image_id = report_info['ImageID']
            findings = report_info.get('findings', [])

            for finding in findings:

                # Bounding boxes are assumed to be in xyxy format in the JSON
                original_boxes_xyxy = finding.get('boxes')

                if not (original_boxes_xyxy and isinstance(original_boxes_xyxy, list) and len(original_boxes_xyxy) > 0):
                    continue # Skip if no boxes for this finding

                processed_boxes = (
                    [xyxy_to_cxcywh(box) for box in original_boxes_xyxy]
                    if self.gt_bbox_format == "cxcywh" else
                    [list(box) for box in original_boxes_xyxy]
                )

                phrases_to_process = []
                # 1. Process the sentence
                sentence_text = finding.get(lang_key, "").strip()
                if sentence_text:
                    if sentence_text.endswith('.'):
                        sentence_text = sentence_text[:-1]
                    group = label_to_group[finding['labels'][0]]
                    phrases_to_process.append(("sentence", sentence_text, group))

                # 2. Process labels if requested
                if include_labels_as_phrases:
                    labels = finding['labels']
                    for label in labels:
                        group = label_to_group[label]
                        phrases_to_process.append(("label", label, group))

                # 3. Create instances for each phrase
                for phrase_type, phrase_text, label_group in phrases_to_process:
                    if not phrase_text: continue

                    if phrase_type == "sentence":
                        num_sentence_phrases += 1
                    else:
                        num_label_phrases += 1

                    self.grounding_instances.append(
                        {
                            "study_id": study_id,
                            "image_id": image_id,
                            "phrase": phrase_text,
                            "gt_bboxes": processed_boxes,
                            "label_group": label_group,
                        }
                    )

        # --- Finalize initialization by mapping instances to groups ---
        self._populate_instances_by_group()

        # --- Weighted sampling ---
        if self.use_weighted_sampling:
            indices_list = [indices for _, indices in self.instances_by_group.items()]
            weights = [1] * len(indices_list) # Use uniform sampling by default
            self.balanced_indices = WeightedCompositeDataset(indices_list, weights)

        # --- Log statistics ---
        if include_labels_as_phrases:
            logger.info(f"Using {num_sentence_phrases} original sentences as phrases.")
            logger.info(f"Using {num_label_phrases} labels as additional phrases.")

        logger.info(
            f"Created {len(self.grounding_instances)} phrase grounding instances for split '{split}'."
        )
        if not self.grounding_instances:
            logger.warning(f"No grounding instances found for split '{split}'. Check data and filters.")

    def _populate_instances_by_group(self):
        """
        Populates `self.instances_by_group` by mapping each label_group to
        a list of corresponding instance indices.
        """
        for i, instance in enumerate(self.grounding_instances):
            self.instances_by_group[instance["label_group"]].append(i)
        self.instances_by_group_list = [indices for _, indices in self.instances_by_group.items()] # List of lists of indices

    def get_num_label_groups(self) -> int:
        """
        Returns the number of unique label groups in the dataset.
        """
        return len(self.instances_by_group)

    def create_uniform_subset(self, target_size: int, ensure_multiple_of: int = 1):
        """
        Alters the dataset to be a uniformly sampled subset of the original data.

        A new random subset is generated each time this method is called.

        Args:
            target_size: The desired approximate size of the validation subset.
            ensure_multiple_of: The number to ensure the target size is a multiple of.
        """
        assert not self.use_weighted_sampling, "Uniform subset creation is not supported with weighted sampling."
        num_groups = self.get_num_label_groups()
        if num_groups == 0:
            logger.warning("Dataset has no label groups to sample from. Skipping subset creation.")
            return
        
        new_indices = create_uniform_subset_from_indices_list(
            target_size=target_size,
            indices_list=self.instances_by_group_list,
            ensure_multiple_of=ensure_multiple_of
        )
        self.active_indices = new_indices

    def update_sampling_weights(self, category_weights: Dict[str, float]):
        """
        Updates the sampling weights for each category.
        """
        assert self.use_weighted_sampling, "Update sampling weights is only supported with weighted sampling."
        weights = [category_weights[category] for category in self.instances_by_group.keys()]
        self.balanced_indices.update_weights(weights)

    def shuffle_indices(self):
        """
        Shuffles the indices of the dataset.
        """
        if self.use_weighted_sampling:
            for indices in self.balanced_indices.datasets: # Shuffle each dataset (which contains indices)
                random.shuffle(indices)
        elif self.active_indices is not None:
            random.shuffle(self.active_indices)
        else:
            random.shuffle(self.grounding_instances)

    def __len__(self) -> int:
        """Returns the number of unique image-phrase pairs."""
        if self.use_weighted_sampling:
            return len(self.balanced_indices)
        elif self.active_indices is not None:
            return len(self.active_indices)
        return len(self.grounding_instances)

    def __getitem__(self, idx: int, skip_image_loading: bool = False) -> Dict[str, Any]:
        """
        Retrieves a single phrase grounding sample.

        Args:
            idx: The index of the grounding instance.
            skip_image_loading: If True, skip loading the image.
        Returns:
            A dictionary containing:
            - 'image': PIL.Image.Image object (RGB).
            - 'phrase': The grounded sentence string.
            - 'gt_bboxes': List of ground truth bounding boxes for the phrase
                           (normalized, format specified by `gt_bbox_format`).
            - 'image_path': (Optional) Absolute path to the image file.
            - 'study_id': (Optional) The StudyID for reference.
        """
        if self.diagnostic_mode:
            orig_idx = idx # Store the original index for diagnostic purposes
        if self.use_weighted_sampling:
            idx = self.balanced_indices[idx]
        elif self.active_indices is not None:
            idx = self.active_indices[idx]
        instance_data = self.grounding_instances[idx]

        image_id = instance_data['image_id']
        phrase = instance_data['phrase']
        gt_bboxes = instance_data['gt_bboxes']
        study_id = instance_data['study_id']
        label_group = instance_data['label_group']

        # Construct image path
        base_name, _ = os.path.splitext(image_id)
        image_id_with_ext = f"{base_name}.{self.image_format}"
        image_path = os.path.join(self.img_dir, image_id_with_ext)
        
        if not skip_image_loading:
            if self.image_transforms:
                transform_input = {
                    'image_path': image_path,
                    'bboxes': gt_bboxes,
                    'bbox_labels': [0] * len(gt_bboxes), # All boxes are for the same sentence
                }
                transform_output = self.image_transforms(**transform_input)
                image = transform_output['pixel_values']
                gt_bboxes = transform_output['bboxes']
            else: # Default behavior: No transforms, load with PIL
                image = Image.open(image_path).convert("RGB")

        output = {
            "phrase": phrase,
            "gt_bboxes": gt_bboxes,
        }

        if not skip_image_loading:
            output["image"] = image

        if self.return_image_path:
            output["image_path"] = image_path
        
        # Additional metadata that might be useful
        output["study_id"] = study_id
        output["image_id"] = image_id
        output["label_group"] = label_group
        if self.diagnostic_mode:
            output["diagnostic_orig_index"] = orig_idx
            output["diagnostic_actual_index"] = idx
            output["diagnostic_dataset_name"] = self.dataset_name
            output["diagnostic_category"] = label_group            

        return output


def get_padchest_gr_phrase_grounding_train_val_test_statistics(
    **kwargs,
) -> dict:
    """
    Calculates and returns key statistics for the train, validation, and test
    splits of the PadChest-GR phrase grounding dataset.

    This function instantiates the PadChestGRPhraseGroundingDataset for each split,
    extracts the necessary data, and computes metrics like the number of
    instances, unique images, phrases, and label groups.

    An "instance" is defined as a unique (image_id, phrase) pair.

    Args:
        **kwargs: Keyword arguments to be passed directly to the
                  PadChestGRPhraseGroundingDataset constructor for all splits
                  (e.g., json_path, csv_path).

    Returns:
        A dictionary containing the statistics for 'train', 'val', and 'test' splits.
        Example structure:
        {
            'train': {
                'num_images': ...,
                'num_phrases': ...,
                'num_label_groups': ...,
                'num_instances': ...,
                'num_instances_per_label_group': {'group1': count, ...}
            },
            'val': {...},
            'test': {...}
        }
    """
    all_stats = {}
    # The dataset class uses 'validation', so we iterate over that
    splits_to_process = ["train", "validation", "test"]

    print("Calculating PadChest-GR phrase grounding dataset statistics...")

    for split_name in splits_to_process:
        try:
            print(f"  -> Loading split: '{split_name}'...")
            # Instantiate the dataset for the current split
            dataset = PadChestGRPhraseGroundingDataset(
                split=split_name, **kwargs
            )

            # Use the 'grounding_instances' list which contains all processed data
            instances = dataset.grounding_instances
            if not instances:
                print(f"     WARNING: No instances found for split '{split_name}'.")
                continue

            # For easier computation, convert the list of dicts to a DataFrame
            df = pd.DataFrame(instances)

            # Calculate statistics
            num_instances = len(df)
            num_images = df["image_id"].nunique()
            num_phrases = df["phrase"].nunique()
            num_label_groups = df["label_group"].nunique()
            instances_per_group = df["label_group"].value_counts().to_dict()

            # Rename 'validation' to 'val' for the output dictionary key
            output_key = "val" if split_name == "validation" else split_name

            all_stats[output_key] = {
                "num_images": num_images,
                "num_phrases": num_phrases,
                "num_label_groups": num_label_groups,
                "num_instances": num_instances,
                "num_instances_per_label_group": instances_per_group,
            }
            print(f"     ...done. Found {num_instances} instances.")

        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            print(
                f"  -> ERROR: Could not process split '{split_name}'. "
                f"Please check dataset paths. Details: {e}"
            )
            continue

    print("...statistics calculation complete.")
    return all_stats
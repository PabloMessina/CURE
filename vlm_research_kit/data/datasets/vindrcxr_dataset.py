import logging
import os
import hashlib
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd
import torch
from PIL import Image
from imagesize import get as get_image_size # can be installed with `pip install imagesize`
from torch.utils.data import Dataset

from vlm_research_kit.data.transforms_factory import create_image_transforms
from vlm_research_kit.utils.bbox_utils import xyxy_to_cxcywh
from vlm_research_kit.utils.file_utils import load_pickle, save_pickle
from vlm_research_kit.settings import (
    VINDRCXR_ANNOTATIONS_DIR,
    VINDRCXR_TRAIN_JPG_DIR,
    VINDRCXR_TEST_JPG_DIR,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)


# Type hint for the function that tokenizes a batch of reports
BatchTokenizerFn = Callable[[List[str]], Dict[str, torch.Tensor]]


VINDRCXR_CLASS_TO_PHRASE = {
    "Aortic enlargement": "Aortic enlargement",
    "Atelectasis": "Atelectasis",
    "Calcification": "Calcification",
    "Cardiomegaly": "Cardiomegaly",
    "Clavicle fracture": "Clavicle fracture",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Emphysema": "Emphysema",
    "Enlarged PA": "Enlarged pulmonary artery",
    "ILD": "Interstitial lung disease",
    "Infiltration": "Infiltration",
    "Lung Opacity": "Lung Opacity",
    "Lung cavity": "Lung cavity",
    "Lung cyst": "Lung cyst",
    "Mediastinal shift": "Mediastinal shift",
    "Nodule/Mass": "Nodule or mass",
    "Other lesion": "Other lesion",
    "COPD": "Chronic Obstructive Pulmonary Disease.",
    "Lung tumor": "Lung tumor",
    "Pneumonia": "Pneumonia",
    "Pleural effusion": "Pleural effusion",
    "Pleural thickening": "Pleural thickening",
    "Pneumothorax": "Pneumothorax",
    "Pulmonary fibrosis": "Pulmonary fibrosis",
    "Rib fracture": "Rib fracture",
    "Tuberculosis": "Tuberculosis",
    "Other diseases": "Other diseases",
    "No finding": "No finding",
}

# 22 classes with bounding box annotations
LOCALIZABLE_CLASSES: Set[str] = {
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Clavicle fracture", "Consolidation", "Edema", "Emphysema",
    "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", "Lung cavity",
    "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Other lesion",
    "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis", "Rib fracture",
}

# 6 classes that are global-only
GLOBAL_CLASSES: Set[str] = {
    "COPD", "Lung tumor", "Pneumonia", "Tuberculosis", "Other diseases",
    "No finding",
}

assert all(c in VINDRCXR_CLASS_TO_PHRASE for c in LOCALIZABLE_CLASSES)
assert all(c in VINDRCXR_CLASS_TO_PHRASE for c in GLOBAL_CLASSES)
assert len(LOCALIZABLE_CLASSES) + len(GLOBAL_CLASSES) == len(VINDRCXR_CLASS_TO_PHRASE)


def _format_bboxes(
    bboxes: List[List[float]],
    decimal_places: int = 2,
) -> List[str]:
    """
    Converts bounding boxes (assumed to be normalized 0-1) to strings.
    """
    formatted_strings = []
    dp = decimal_places
    for box in bboxes:
        a, b, c, d = box
        formatted_strings.append(f"[{a:.{dp}f},{b:.{dp}f},{c:.{dp}f},{d:.{dp}f}]")
    return formatted_strings


def _precompute_image_size_cache(train_img_dir: str, test_img_dir: str, image_format: str) -> Dict[str, Tuple[int, int]]:
    """
    Precomputes the sizes of all images in the directory.
    """
    name_hash = hashlib.sha256(f"{train_img_dir}_{test_img_dir}_{image_format}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"vindrcxr_image_size_cache_{name_hash}.pkl")
    if os.path.exists(cache_path):
        cache = load_pickle(cache_path)
        logger.info(f"Loading image size cache from {cache_path} with {len(cache)} images.")
        return cache
    image_sizes = {}
    for img_dir in [train_img_dir, test_img_dir]:
        for file in os.listdir(img_dir):
            if file.endswith(image_format):
                image_path = os.path.join(img_dir, file)
                image_sizes[image_path] = get_image_size(image_path)
    save_pickle(image_sizes, cache_path)
    logger.info(f"Saved image size cache to {cache_path} with {len(image_sizes)} images.")
    return image_sizes


def _load_vindrcxr_bboxes_and_preprocess(
    csv_path: str, img_dir: str, bbox_format: str, image_format: str, image_size_cache: Dict[str, Tuple[int, int]]
) -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Shared helper to load, normalize, filter, and format bounding boxes from a VinDr-CXR CSV file.
    """
    df = pd.read_csv(csv_path)
    df = df[df["x_min"].notna()]  # Filter out rows without boxes

    image_id_to_boxes = defaultdict(lambda: defaultdict(list))
    anomalous_count = 0

    for _, row in df.iterrows():
        image_id = row['image_id']
        class_name = row['class_name']
        image_path = os.path.join(img_dir, f"{image_id}.{image_format}")
        w, h = image_size_cache[image_path]
        if w <= 0 or h <= 0:
            anomalous_count += 1
            continue

        bbox_xyxy = [
            row['x_min'] / w, row['y_min'] / h,
            row['x_max'] / w, row['y_max'] / h,
        ]
        bbox_xyxy = [max(0.0, min(1.0, coord)) for coord in bbox_xyxy]
        
        if bbox_xyxy[0] >= bbox_xyxy[2] or bbox_xyxy[1] >= bbox_xyxy[3]:
            anomalous_count += 1
            continue
        
        final_bbox = xyxy_to_cxcywh(bbox_xyxy) if bbox_format == "cxcywh" else bbox_xyxy
        image_id_to_boxes[image_id][class_name].append(final_bbox)

    if anomalous_count > 0:
        logger.warning(f"Found and skipped {anomalous_count} anomalous bboxes in {os.path.basename(csv_path)}")
    
    return dict(image_id_to_boxes)


class VinDrCXR_GroundedReportGenerationDataset(Dataset):
    """
    PyTorch Dataset for generating pseudo-reports from the VinDr-CXR dataset.

    Since VinDr-CXR does not provide reports, this dataset synthesizes them
    by combining phrases corresponding to annotated findings for each image.
    The generated "report" consists of grounded phrases for localizable findings
    (with bounding boxes) followed by un-grounded phrases for global findings.
    Bounding boxes are pre-processed (normalized, cleaned) during initialization.
    """

    def __init__(
        self,
        image_transforms_kwargs: Optional[Dict[str, Any]] = None,
        image_transforms: Optional[Callable] = None,
        annotations_dir: str = VINDRCXR_ANNOTATIONS_DIR,
        train_img_dir: str = VINDRCXR_TRAIN_JPG_DIR,
        test_img_dir: str = VINDRCXR_TEST_JPG_DIR,
        split: Literal["train", "test", "all"] = "train",
        bbox_format: Optional[str] = "cxcywh",
        image_format: Literal["jpg"] = "jpg",
        return_image_path: bool = False,
    ):
        super().__init__()
        self.annotations_dir = annotations_dir
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.split = split
        self.bbox_format = bbox_format
        self.image_format = image_format
        self.return_image_path = return_image_path
        self.image_size_cache = _precompute_image_size_cache(train_img_dir, test_img_dir, image_format)
        
        if image_transforms is not None:
            self.image_transforms = image_transforms
        elif image_transforms_kwargs:
            self.image_transforms = create_image_transforms(**image_transforms_kwargs)
        else:
            self.image_transforms = None

        logger.info("Pre-processing bounding boxes for both train and test sets...")
        train_bboxes = _load_vindrcxr_bboxes_and_preprocess(
            csv_path=os.path.join(annotations_dir, 'annotations_train.csv'),
            img_dir=self.train_img_dir,
            bbox_format=self.bbox_format,
            image_format=self.image_format,
            image_size_cache=self.image_size_cache,
        )
        test_bboxes = _load_vindrcxr_bboxes_and_preprocess(
            csv_path=os.path.join(annotations_dir, 'annotations_test.csv'),
            img_dir=self.test_img_dir,
            bbox_format=self.bbox_format,
            image_format=self.image_format,
            image_size_cache=self.image_size_cache,
        )
        self.bbox_annotations = {**train_bboxes, **test_bboxes}
        logger.info(f"Finished pre-processing. Loaded annotations for {len(self.bbox_annotations)} images.")

        logger.info("Loading image-level labels...")
        train_labels_df = pd.read_csv(os.path.join(annotations_dir, 'image_labels_train.csv'))
        test_labels_df = pd.read_csv(os.path.join(annotations_dir, 'image_labels_test.csv'))
        test_labels_df = test_labels_df.rename(columns={"Other disease": "Other diseases"})
        
        logger.info(f"Processing instances for split: '{split}'...")
        self.report_instances = []
        
        if split in ['train', 'all']:
            agg_train_df = train_labels_df.drop(columns='rad_id').groupby('image_id').sum()
            
            for image_id, row in agg_train_df.iterrows():
                positive_findings = row[row > 0].index.tolist()
                self.report_instances.append(self._create_instance(image_id, positive_findings, 'train'))
            # --- END OF OPTIMIZED SECTION ---

        if split in ['test', 'all']:
            logger.info("Processing test labels...")
            for _, row in test_labels_df.iterrows():
                image_id = row['image_id']
                # For test data, we don't aggregate, just find positive labels in the row
                positive_findings = row.drop('image_id')[row.drop('image_id') > 0].index.tolist()
                self.report_instances.append(self._create_instance(image_id, positive_findings, 'test'))

        logger.info(f"Created {len(self.report_instances)} report generation instances for split '{split}'.")

    def _create_instance(self, image_id: str, findings: List[str], split_name: str):
        """Helper to create a single instance dictionary using pre-processed bboxes."""
        local_findings = [f for f in findings if f in LOCALIZABLE_CLASSES]
        global_findings = [f for f in findings if f in GLOBAL_CLASSES and f != 'No finding']
        
        all_bboxes = []
        all_bbox_labels = []
        
        image_annots = self.bbox_annotations.get(image_id, {})
        for i, class_name in enumerate(local_findings):
            boxes = image_annots.get(class_name, [])
            if boxes:
                all_bboxes.extend(boxes)
                all_bbox_labels.extend([i] * len(boxes))

        return {
            "image_id": image_id,
            "split_name": split_name,
            "local_findings": local_findings,
            "global_findings": global_findings,
            "has_no_finding_label": "No finding" in findings,
            "bboxes": all_bboxes,
            "bbox_labels": all_bbox_labels,
        }

    def __len__(self) -> int:
        return len(self.report_instances)

    def __getitem__(self, idx: int, skip_image_loading: bool = False) -> Dict[str, Any]:
        """
        Retrieves a single report generation sample.

        Args:
            idx: The index of the report generation instance.
            skip_image_loading: If True, skip loading the image.
        Returns:
            A dictionary containing:
            - 'image': The processed image tensor or PIL Image.
            - 'report': The generated report string. 
            - 'image_path': (Optional) The absolute path to the image file,
                            included if `return_image_path` is True.
        """
        instance = self.report_instances[idx]
        image_id = instance['image_id']
        img_dir = self.train_img_dir if instance['split_name'] == 'train' else self.test_img_dir
        image_path = os.path.join(img_dir, f"{image_id}.{self.image_format}")

        # BBoxes are already pre-processed (normalized and correct format)
        bboxes = instance['bboxes']
        bbox_labels = instance['bbox_labels']
        
        if not skip_image_loading:
            if self.image_transforms:
                transform_output = self.image_transforms(
                    image_path=image_path,
                    bboxes=bboxes,
                    bbox_labels=bbox_labels
                )
                image = transform_output['pixel_values']
                augmented_bboxes = transform_output['bboxes']
                augmented_bbox_labels = [int(l) for l in transform_output['bbox_labels']]
            else:
                image = Image.open(image_path).convert("RGB")
                augmented_bboxes = bboxes
                augmented_bbox_labels = bbox_labels
        else:
            image = None
            augmented_bboxes = bboxes
            augmented_bbox_labels = bbox_labels

        # --- Construct the final pseudo-report ---
        report_phrases = [VINDRCXR_CLASS_TO_PHRASE[f] for f in instance['local_findings']]
        
        boxes_per_phrase = [[] for _ in report_phrases]
        for box, label_idx in zip(augmented_bboxes, augmented_bbox_labels):
            boxes_per_phrase[label_idx].append(box)
        
        for i, boxes in enumerate(boxes_per_phrase):
            if boxes:
                formatted = _format_bboxes(boxes)
                report_phrases[i] += f" {' '.join(formatted)}"

        global_phrases = [VINDRCXR_CLASS_TO_PHRASE[f] for f in instance['global_findings']]
        report_phrases.extend(global_phrases)
        
        has_positive_finding = bool(instance['local_findings'] or instance['global_findings'])
        
        if instance['has_no_finding_label']:
            if has_positive_finding:
                report_phrases.append("No other relevant findings are noted.")
            else:
                report_phrases.append(VINDRCXR_CLASS_TO_PHRASE["No finding"])
        
        final_report = ". ".join(report_phrases) + "."

        output = {"image": image, "report": final_report}
        if self.return_image_path:
            output["image_path"] = image_path
        
        return output


class VinDrCXR_PhraseGroundingDataset(Dataset):
    """
    PyTorch Dataset for VinDr-CXR phrase grounding.

    Each item corresponds to a single image paired with a phrase for a
    localizable finding and all associated bounding boxes for that finding.
    Bounding boxes are pre-processed (normalized, cleaned) during initialization.
    """

    def __init__(
        self,
        image_transforms_kwargs: Optional[Dict[str, Any]] = None,
        annotations_dir: str = VINDRCXR_ANNOTATIONS_DIR,
        train_img_dir: str = VINDRCXR_TRAIN_JPG_DIR,
        test_img_dir: str = VINDRCXR_TEST_JPG_DIR,
        split: Literal["train", "test", "all"] = "train", # Removed 'validation'
        gt_bbox_format: Literal["xyxy", "cxcywh"] = "xyxy",
        image_format: Literal["jpg"] = "jpg",
        return_image_path: bool = False,
    ):
        super().__init__()
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.split = split
        self.gt_bbox_format = gt_bbox_format
        self.image_format = image_format
        self.return_image_path = return_image_path
        self.image_transforms = None
        if image_transforms_kwargs:
            self.image_transforms = create_image_transforms(**image_transforms_kwargs)

        self.image_size_cache = _precompute_image_size_cache(train_img_dir, test_img_dir, image_format)

        # --- Pre-computation using the shared helper function ---
        logger.info("Pre-processing bounding boxes for Phrase Grounding...")
        train_bboxes_map = _load_vindrcxr_bboxes_and_preprocess(
            csv_path=os.path.join(annotations_dir, 'annotations_train.csv'),
            img_dir=self.train_img_dir,
            bbox_format=self.gt_bbox_format,
            image_format=self.image_format,
            image_size_cache=self.image_size_cache,
        )
        test_bboxes_map = _load_vindrcxr_bboxes_and_preprocess(
            csv_path=os.path.join(annotations_dir, 'annotations_test.csv'),
            img_dir=self.test_img_dir,
            bbox_format=self.gt_bbox_format,
            image_format=self.image_format,
            image_size_cache=self.image_size_cache,
        )
        
        split_map = {} # To track if an image_id is originally from train or test
        for image_id in train_bboxes_map: split_map[image_id] = 'train'
        for image_id in test_bboxes_map: split_map[image_id] = 'test'
            
        if split == 'train':
            bboxes_to_process = train_bboxes_map
        elif split == 'test':
            bboxes_to_process = test_bboxes_map
        elif split == 'all':
            bboxes_to_process = {**train_bboxes_map, **test_bboxes_map}
        else:
            raise ValueError(f"Invalid split '{split}'. Must be one of ['train', 'test', 'all'].")

        # --- Create Grounding Instances from Pre-processed Data ---
        logger.info(f"Creating phrase grounding instances for split: '{split}'...")
        self.grounding_instances = []
        for image_id, class_map in bboxes_to_process.items():
            for class_name, bboxes in class_map.items():
                if class_name == "No finding": continue # Should already be filtered
                self.grounding_instances.append({
                    "image_id": image_id,
                    "phrase": VINDRCXR_CLASS_TO_PHRASE[class_name],
                    "gt_bboxes": bboxes,
                    "split_name": split_map[image_id],
                })
        
        logger.info(f"Created {len(self.grounding_instances)} instances.")

    def __len__(self) -> int:
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
            - 'phrase': The phrase for the grounding instance.
            - 'gt_bboxes': The bounding boxes for the grounding instance.
        """
        instance = self.grounding_instances[idx]
        image_id = instance['image_id']
        split_name = instance['split_name']
        img_dir = self.train_img_dir if split_name == 'train' else self.test_img_dir
        image_path = os.path.join(img_dir, f"{image_id}.{self.image_format}")
        
        gt_bboxes = instance['gt_bboxes']

        if not skip_image_loading:
            if self.image_transforms:
                transform_output = self.image_transforms(
                    image_path=image_path,
                    bboxes=gt_bboxes,
                    bbox_labels=[0] * len(gt_bboxes) # Dummy labels
                )
                image = transform_output['pixel_values']
                gt_bboxes = transform_output['bboxes']
            else:
                image = Image.open(image_path).convert("RGB")
        
        output = {
            "phrase": instance['phrase'],
            "gt_bboxes": gt_bboxes,
        }

        if not skip_image_loading:
            output["image"] = image

        if self.return_image_path:
            output["image_path"] = image_path
        
        output["image_id"] = image_id
        return output
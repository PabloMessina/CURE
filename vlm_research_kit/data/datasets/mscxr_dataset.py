import os
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from vlm_research_kit.data.dataset_helpers import WeightedCompositeDataset, create_uniform_subset_from_indices_list
from vlm_research_kit.data.datasets.mimiccxr_dataset import get_dicom_id_to_image_path_map
from vlm_research_kit.data.transforms_factory import create_image_transforms
from vlm_research_kit.settings import MS_CXR_LOCAL_ALIGNMENT_CSV_PATH
from vlm_research_kit.utils.bbox_utils import xyxy_to_cxcywh # If needed for cxcywh output


logger = logging.getLogger(__name__)


def get_mscxr_train_val_test_statistics():
    """
    Returns the statistics of the MS-CXR train, val, and test sets.
    """
    df = pd.read_csv(MS_CXR_LOCAL_ALIGNMENT_CSV_PATH)
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']
    return {
        "train": {
            "num_images": len(df_train['dicom_id'].unique()),
            "num_phrases": len(df_train['label_text'].unique()),
            "num_categories": len(df_train['category_name'].unique()),
            "num_instances": len(df_train.groupby(['dicom_id', 'label_text']).size()),
            "num_instances_per_category": {category: len(df_train[df_train['category_name'] == category].groupby(['dicom_id', 'label_text']).size()) for category in df_train['category_name'].unique()},
        },
        "val": {
            "num_images": len(df_val['dicom_id'].unique()),
            "num_phrases": len(df_val['label_text'].unique()),
            "num_categories": len(df_val['category_name'].unique()),
            "num_instances": len(df_val.groupby(['dicom_id', 'label_text']).size()),
            "num_instances_per_category": {category: len(df_val[df_val['category_name'] == category].groupby(['dicom_id', 'label_text']).size()) for category in df_val['category_name'].unique()},
        },
        "test": {
            "num_images": len(df_test['dicom_id'].unique()),
            "num_phrases": len(df_test['label_text'].unique()),
            "num_categories": len(df_test['category_name'].unique()),
            "num_instances": len(df_test.groupby(['dicom_id', 'label_text']).size()),
            "num_instances_per_category": {category: len(df_test[df_test['category_name'] == category].groupby(['dicom_id', 'label_text']).size()) for category in df_test['category_name'].unique()},
        },
    }


def _convert_and_normalize_bbox_mscxr(
    x: float, y: float, w: float, h: float,
    image_width: int, image_height: int
) -> List[float]:
    """
    Converts an absolute (x, y, w, h) bounding box to normalized (x_min, y_min, x_max, y_max).
    x, y are top-left.
    """
    assert image_width > 0 and image_height > 0, "Image dimensions must be positive."
    assert w > 0 and h > 0, "Width and height must be positive."
    x_min = x / image_width
    y_min = y / image_height
    x_max = (x + w) / image_width
    y_max = (y + h) / image_height
    return [x_min, y_min, x_max, y_max]


class MSCXRPhraseGroundingDataset(Dataset):
    """
    PyTorch Dataset for MS-CXR phrase grounding evaluation.

    Each item corresponds to a single image paired with a single phrase
    (label_text) and its associated bounding box annotations from the MS-CXR dataset.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        csv_path: str = MS_CXR_LOCAL_ALIGNMENT_CSV_PATH,
        gt_bbox_format: Literal["xyxy", "cxcywh"] = "xyxy",
        image_transforms_kwargs: Optional[Dict[str, Any]] = None,
        return_image_path: bool = False,
        use_weighted_sampling: bool = False,
        diagnostic_mode: bool = False,
    ):
        """
        Initializes the MSCXRPhraseGroundingDataset.

        Args:
            split: The dataset split to load ('train', 'val', 'test').
            csv_path: Path to the MS-CXR local alignment CSV file.
            gt_bbox_format: The format for the ground truth bounding boxes
                            returned by the dataset ('xyxy' or 'cxcywh').
                            Coordinates are always normalized [0, 1].
            image_transforms_kwargs: Optional dictionary of keyword arguments
                                     to build an image transformation function.
                                     If None, images are loaded as PIL Images.
            return_image_path: If True, include the absolute image path in the
                               output dictionary.
            use_weighted_sampling: If True, use weighted sampling to balance the dataset.
            diagnostic_mode: If True, include diagnostic information in the output.
        """
        super().__init__()

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"MS-CXR CSV file not found: {csv_path}")
        
        self.dicom_id_to_image_path_map = get_dicom_id_to_image_path_map() # Load DICOM ID to image path mapping
        self.split = split
        self.gt_bbox_format = gt_bbox_format
        self.return_image_path = return_image_path
        self.use_weighted_sampling = use_weighted_sampling
        self.diagnostic_mode = diagnostic_mode
        self.image_transforms = None
        if image_transforms_kwargs:
            self.image_transforms = create_image_transforms(**image_transforms_kwargs)
        self.dataset_name = "mscxr_pg" # Phrase Grounding

        if self.diagnostic_mode:
            logger.info("Running in diagnostic mode.")

        # --- Attributes for subsampling ---
        self.active_indices: Optional[List[int]] = None
        self.instances_by_category: Dict[str, List[int]] = defaultdict(list)

        logger.info(f"Loading MS-CXR CSV from: {csv_path}")
        df = pd.read_csv(csv_path)

        logger.info(f"Filtering MS-CXR dataset for split: {split}")
        df_split = df[df['split'] == split].copy()
        if df_split.empty:
            raise ValueError(f"No data found for split '{split}' in {csv_path}")

        self.grounding_instances = []
        logger.info(f"Processing MS-CXR data for split '{split}'...")

        # Group by 'dicom_id' and 'label_text' to collect all boxes for a given phrase on an image
        # 'label_text' is the phrase in MS-CXR
        grouped = df_split.groupby(['dicom_id', 'label_text'])

        for (dicom_id, phrase), group_df in grouped:
            image_path = self.dicom_id_to_image_path_map[dicom_id]

            # All rows in group_df share the same image_width, image_height, category_name
            # (assuming data consistency)
            image_width = group_df['image_width'].iloc[0]
            image_height = group_df['image_height'].iloc[0]
            category_name = group_df['category_name'].iloc[0] # The broader category

            instance_bboxes_xyxy_norm = []
            for _, row in group_df.iterrows():
                # BBoxes are given as x, y (top-left), w, h in absolute pixel coords
                bbox_xyxy_norm = _convert_and_normalize_bbox_mscxr(
                    row['x'], row['y'], row['w'], row['h'],
                    image_width, image_height
                )
                instance_bboxes_xyxy_norm.append(bbox_xyxy_norm)

            # Convert to target format if needed
            final_bboxes = []
            if self.gt_bbox_format == "cxcywh":
                final_bboxes = [xyxy_to_cxcywh(box) for box in instance_bboxes_xyxy_norm]
            else: # "xyxy"
                final_bboxes = instance_bboxes_xyxy_norm

            self.grounding_instances.append({
                "dicom_id": dicom_id,
                "image_path": image_path,
                "phrase": phrase, # This is the 'label_text' from MS-CXR
                "gt_bboxes": final_bboxes,
                "category_name": category_name, # Broader category
            })

        # --- Finalize initialization by mapping instances to categories ---
        self._populate_instances_by_category()

        # --- Weighted sampling ---
        if self.use_weighted_sampling:
            indices_list = [indices for _, indices in self.instances_by_category.items()]
            weights = [1] * len(indices_list) # Use uniform sampling by default
            self.balanced_indices = WeightedCompositeDataset(indices_list, weights)            
        
        # --- Logging ---
        logger.info(
            f"Created {len(self.grounding_instances)} phrase grounding instances for MS-CXR split '{split}'."
        )
        if not self.grounding_instances:
            logger.warning(f"No grounding instances created for split '{split}'. Check data and paths.")

    def _populate_instances_by_category(self):
        """
        Populates `self.instances_by_category` by mapping each category_name to
        a list of corresponding instance indices.
        """
        logger.info("Grouping instances by category_name for potential subsampling.")
        for i, instance in enumerate(self.grounding_instances):
            self.instances_by_category[instance["category_name"]].append(i)
        self.instances_by_category_list = [indices for _, indices in self.instances_by_category.items()]

    def get_num_categories(self) -> int:
        """
        Returns the number of unique categories in the dataset.
        """
        return len(self.instances_by_category)

    def create_uniform_subset(self, target_size: int, ensure_multiple_of: int = 1):
        """
        Alters the dataset to be a uniformly sampled subset of the original data.

        A new random subset is generated each time this method is called.

        Args:
            target_size: The desired approximate size of the validation subset.
            ensure_multiple_of: The desired multiple of the target size.
        """
        assert not self.use_weighted_sampling, "Uniform subset creation is not supported with weighted sampling."
        num_categories = self.get_num_categories()
        if num_categories == 0:
            logger.warning("Dataset has no categories to sample from. Skipping subset creation.")
            return

        new_indices = create_uniform_subset_from_indices_list(
            target_size=target_size,
            indices_list=self.instances_by_category_list,
            ensure_multiple_of=ensure_multiple_of
        )
        self.active_indices = new_indices

    def update_sampling_weights(self, category_weights: Dict[str, float]):
        """
        Updates the sampling weights for each category.
        """
        assert self.use_weighted_sampling, "Update sampling weights is only supported with weighted sampling."
        weights = [category_weights[category] for category in self.instances_by_category.keys()]
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
            - 'phrase': The phrase for the grounding instance.
            - 'gt_bboxes': The bounding boxes for the grounding instance.
        """
        if self.diagnostic_mode:
            orig_idx = idx # Store the original index for diagnostic purposes
        if self.use_weighted_sampling:
            idx = self.balanced_indices[idx]
        elif self.active_indices is not None:
            idx = self.active_indices[idx]
        instance_data = self.grounding_instances[idx]
        
        image_path = instance_data['image_path']
        gt_bboxes = instance_data['gt_bboxes']
        
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
            "phrase": instance_data['phrase'],
            "gt_bboxes": gt_bboxes,
            "category_name": instance_data['category_name'],
            "dicom_id": instance_data['dicom_id'],
        }

        if not skip_image_loading:
            output["image"] = image

        if self.return_image_path:
            output["image_path"] = image_path

        if self.diagnostic_mode:
            output["diagnostic_orig_index"] = orig_idx
            output["diagnostic_actual_index"] = idx
            output["diagnostic_dataset_name"] = self.dataset_name
            output["diagnostic_category"] = instance_data['category_name']
        
        return output
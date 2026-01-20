import logging
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

# Assume these are in an auxiliary file as requested
from vlm_research_kit.data.dataset_helpers import WeightedCompositeDataset
from vlm_research_kit.data.datasets.mimiccxr_dataset import get_dicom_id_to_image_path_map
from vlm_research_kit.data.transforms_factory import create_image_transforms
from vlm_research_kit.settings import CHEST_IMAGENOME_LOCATION_REPORT_SNIPPETS_PATH, MIMIC_CXR_SPLIT_CSV_PATH
from vlm_research_kit.utils.bbox_utils import xyxy_to_cxcywh
from vlm_research_kit.utils.file_utils import load_pickle

logger = logging.getLogger(__name__)

# Define valid prompt modes
PromptMode = Literal["describe", "locate", "locate_and_describe"]

# Type hint for the function that tokenizes a batch of prompts + reports
BatchTokenizerFn = Callable[[List[str], List[str]], Dict[str, torch.Tensor]]

# Define Binning Strategy by the number of boxes
NUM_BOXES_SPLITS = {
    (0, 0): "0 boxes",
    (1, 1): "1 box",
    (2, 2): "2 boxes",
    (3, 4): "3-4 boxes",
    (5, 9999): "5+ boxes",
}

SUPPORTED_LOCATE_AND_DESCRIBE_LOCATIONS = {
    'left lung', 'right lung', 'cardiac silhouette', 'left costophrenic angle', 'mediastinum', 'right costophrenic angle',
    'right hilar structures', 'left hilar structures', 'upper mediastinum', 'left lower lung zone', 'right lower lung zone',
    'abdomen', 'spine', 'right clavicle', 'left clavicle', 'trachea', 'left mid lung zone', 'right mid lung zone',
    'right apical zone', 'right hemidiaphragm', 'left apical zone', 'aortic arch', 'svc', 'left hemidiaphragm',
    'right upper lung zone', 'carina', 'left upper lung zone', 'right atrium', 'cavoatrial junction'
}

SUPPORTED_LOCATE_LOCATIONS = {
    'right upper abdomen', 'left cardiac silhouette', 'left upper abdomen', 'right cardiac silhouette',
    'right cardiophrenic angle', 'left cardiophrenic angle', 'descending aorta'
}
SUPPORTED_LOCATE_LOCATIONS.update(SUPPORTED_LOCATE_AND_DESCRIBE_LOCATIONS) # Add the locations that are both locate and describe

SUPPORTED_DESCRIBE_LOCATIONS = {
    'neck', 'right chest wall', 'left chest wall', 'right shoulder', 'left shoulder', 'right arm', 'right breast',
    'left arm', 'left breast'
}
SUPPORTED_DESCRIBE_LOCATIONS.update(SUPPORTED_LOCATE_AND_DESCRIBE_LOCATIONS) # Add the locations that are both locate and describe

def _format_bbox_string(
    bbox: List[float], decimal_places: int = 2
) -> str:
    """
    Formats a single bounding box into a string representation.
    e.g., [0.1, 0.2, 0.3, 0.4] -> "[0.10,0.20,0.30,0.40]"
    """
    return f"[{','.join([f'{coord:.{decimal_places}f}' for coord in bbox])}]"


class SubsampleableConcatDataset(ConcatDataset):
    """
    A ConcatDataset that supports creating a uniformly sampled subset of itself.
    This is intended for validation/test sets to speed up evaluation.
    """

    def create_uniform_subset(self, target_size: int, ensure_multiple_of: int = 1):
        """
        Alters the dataset to be a uniformly sampled subset of the original data.

        The target size is distributed as evenly as possible among the constituent
        datasets and their internal anatomical locations. A new random subset is
        generated each time this method is called.

        Args:
            target_size: The desired approximate size of the validation subset.
            ensure_multiple_of: The number to ensure the target size is a multiple of.
        """
        if not self.datasets:
            logger.warning("No datasets to subsample from.")
            return

        num_tasks = len(self.datasets)
        size_per_task = math.ceil(target_size / num_tasks)

        for dataset in self.datasets:
            num_locations = dataset.get_num_locations()
            assert num_locations > 0, f"Dataset {type(dataset).__name__} has no locations to sample from."

            num_samples = max(size_per_task, num_locations) # At least one sample per location
            num_samples = math.ceil(num_samples / ensure_multiple_of) * ensure_multiple_of # Ensure the number of samples is a multiple of the ensure_multiple_of
            dataset.create_uniform_subset(num_samples)

        # After subsetting, the cumulative sizes need to be recalculated
        self.cumulative_sizes = self.cumsum(self.datasets)
        assert self.cumulative_sizes[-1] % ensure_multiple_of == 0, (
            f"Cumulative sizes {self.cumulative_sizes} are not a multiple of {ensure_multiple_of}."
        )

class CustomWeightedCompositeDataset(WeightedCompositeDataset):

    def __init__(self, datasets, weights, dataset_names):
        super().__init__(datasets, weights, dataset_names)

    def update_sampling_weights(self, dataset_weights: List[float], category_weights: Dict[str, float]):
        for dataset, dataset_name in zip(self.datasets, self.dataset_names):
            dataset.update_sampling_weights(category_weights[dataset_name])
        self.update_weights(dataset_weights)

    def shuffle_indices(self):
        """
        Shuffles the indices of each dataset in the composite dataset.
        """
        for dataset in self.datasets:
            dataset.shuffle_indices()


# ----------------------------------------------------------------------------
# 1. Base Class for Common Logic
# ----------------------------------------------------------------------------

class _ChestImaGenomeBaseDataset(Dataset):
    def __init__(
        self,
        image_transforms_kwargs: Dict[str, Any],
        split: Literal["train", "val", "test"],
        bbox_format: Literal["xyxy", "cxcywh"] = "xyxy",
        return_image_path: bool = False,
        tokenizer_fn: Optional[BatchTokenizerFn] = None,
        disable_tqdm: bool = False,
        # Pre-loaded data to avoid redundant loads by child classes
        dicom_id_to_image_path_map: Optional[Dict] = None,
        study_data_for_split: Optional[List[Dict]] = None,
        use_weighted_sampling: bool = False,
        diagnostic_mode: bool = False,
        dataset_name: Optional[str] = None,
    ):
        super().__init__()
        self.split = split
        self.use_weighted_sampling = use_weighted_sampling
        self.image_transforms = None
        if image_transforms_kwargs:
            self.image_transforms = create_image_transforms(**image_transforms_kwargs)
        
        self.bbox_format = bbox_format
        self.return_image_path = return_image_path
        self.tokenizer_fn = tokenizer_fn
        self.instances_by_location: Dict[str, List[int]] = defaultdict(list)
        self.active_indices: Optional[List[int]] = None
        
        self.instances: List[Dict] = []
        self.disable_tqdm = disable_tqdm
        self.diagnostic_mode = diagnostic_mode
        self.dataset_name = dataset_name

        if self.diagnostic_mode:
            logger.info("Running in diagnostic mode.")

        if dicom_id_to_image_path_map is None:
            self.dicom_id_to_image_path_map = get_dicom_id_to_image_path_map()
        else:
            self.dicom_id_to_image_path_map = dicom_id_to_image_path_map

        if study_data_for_split is None:
            self.study_data_for_split = self._load_and_filter_data(
                CHEST_IMAGENOME_LOCATION_REPORT_SNIPPETS_PATH
            )
        else:
            self.study_data_for_split = study_data_for_split

    def get_num_locations(self) -> int:
        """Returns the number of unique anatomical locations in the dataset."""
        return len(self.instances_by_location)

    def _populate_instances_by_location(self):
        """Populates `self.instances_by_location`."""
        for i, instance in enumerate(self.instances):
            self.instances_by_location[instance["location"]].append(i)

    def _finalize_initialization(self):
        """Called by child classes after _create_instances is complete."""
        self._populate_instances_by_location()
        # --- Weighted sampling ---
        if self.use_weighted_sampling:
            indices_list = [indices for _, indices in self.instances_by_location.items()]
            weights = [1] * len(indices_list) # Use uniform sampling by default
            self.balanced_indices = WeightedCompositeDataset(indices_list, weights)
        
        # --- Log statistics ---
        logger.info(f"Created {len(self.instances)} instances for split '{self.split}'.")
        logger.info(f"Found {self.get_num_locations()} unique anatomical locations.")
        logger.info(f"Using weighted sampling: {self.use_weighted_sampling}.")
        # Number of unique images in the dataset
        num_unique_images = len(set([instance["dicom_id"] for instance in self.instances]))
        logger.info(f"Found {num_unique_images} unique images in the dataset.")


    def create_uniform_subset(self, num_samples: int):
        """Creates a subset by uniformly sampling from each location."""
        assert not self.use_weighted_sampling, "Uniform subset creation is not supported with weighted sampling."
        new_indices = []
        num_locations = self.get_num_locations()
        num_samples_per_location = num_samples // num_locations
        remainder = num_samples % num_locations
        for i, (loc, indices) in enumerate(self.instances_by_location.items()):
            if i < remainder:
                k = num_samples_per_location + 1
            else:
                k = num_samples_per_location
            if k <= len(indices): # Sample without replacement
                new_indices.extend(random.sample(indices, k=k))
            else: # Sample with replacement
                new_indices.extend(random.choices(indices, k=k))
        assert len(new_indices) == num_samples, f"Expected {num_samples} samples, but got {len(new_indices)}."
        self.active_indices = new_indices

    def update_sampling_weights(self, category_weights: Dict[str, float]):
        """
        Updates the sampling weights for each location.
        """
        assert self.use_weighted_sampling, "Update sampling weights is only supported with weighted sampling."
        weights = [category_weights[category] for category in self.instances_by_location.keys()]
        self.balanced_indices.update_weights(weights)

    def _clamp_and_validate_bbox(
        self, bbox: Tuple[float, ...]
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Clamps bounding box coordinates to the [0.0, 1.0] range and
        validates them.

        Returns:
            A valid, clamped bounding box tuple, or None if the box is invalid
            (e.g., x1 >= x2).
        """
        x1, y1, x2, y2 = bbox
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(1.0, x2)
        y2 = min(1.0, y2)

        # After clamping, check if the bbox is still valid
        if x1 >= x2 or y1 >= y2:
            return None  # Invalid bbox, discard

        return (x1, y1, x2, y2)

    def _load_and_filter_data(self, data_path: str) -> List[Dict]:
        """Loads data and filters it according to the specified split."""
        logger.info(f"Loading Chest-ImaGenome data from: {data_path}")
        all_study_data = load_pickle(data_path)

        logger.info(f"Loading split info from: {MIMIC_CXR_SPLIT_CSV_PATH}")
        split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV_PATH)
        split_name_in_csv = "validate" if self.split == "val" else self.split
        split_dicom_ids = set(
            split_df[split_df.split == split_name_in_csv].dicom_id.unique()
        )

        if not split_dicom_ids:
            raise ValueError(
                f"No dicom_ids found for split '{self.split}' in {MIMIC_CXR_SPLIT_CSV_PATH}"
            )

        logger.info(
            f"Filtering for {len(split_dicom_ids)} dicom_ids in '{self.split}' split."
        )
        return [
            study
            for study in all_study_data
            if study["dicom_id"] in split_dicom_ids
        ]

    def _create_instances(self):
        """
        Populates `self.instances`. Must be implemented by child classes.
        """
        raise NotImplementedError

    def _build_instance(
        self,
        base_info: Dict,
        mode: PromptMode,
        bbox_xyxy: Optional[Tuple] = None,
        snippet: Optional[str] = None,
    ) -> Dict:
        """Helper to construct a single instance dictionary."""
        instance = base_info.copy()
        instance["prompt_mode"] = mode
        if snippet:
            instance["report_snippet"] = snippet
        if bbox_xyxy:
            if self.bbox_format == "cxcywh":
                instance["bbox"] = xyxy_to_cxcywh(bbox_xyxy)
            else:
                instance["bbox"] = bbox_xyxy
        return instance

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
            random.shuffle(self.instances)

    def __len__(self) -> int:
        if self.use_weighted_sampling:
            return len(self.balanced_indices)
        elif self.active_indices is not None:
            return len(self.active_indices)
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieves and processes a single data sample by index."""
        if self.diagnostic_mode:
            orig_idx = idx # Store the original index for diagnostic purposes
        if self.use_weighted_sampling:
            idx = self.balanced_indices[idx]
        elif self.active_indices is not None:
            idx = self.active_indices[idx]
        instance = self.instances[idx]

        dicom_id = instance["dicom_id"]
        location = instance["location"]
        prompt_mode = instance["prompt_mode"]
        image_path = self.dicom_id_to_image_path_map[dicom_id]
        has_bbox = "bbox" in instance

        pixel_values: torch.Tensor
        augmented_bbox = None

        if has_bbox:
            original_bbox = instance["bbox"]
            transform_input = {
                "image_path": image_path,
                "bboxes": [original_bbox],
                "bbox_labels": [0],
            }
            transform_output = self.image_transforms(**transform_input)
            pixel_values = transform_output["pixel_values"]

            if transform_output["bboxes"]:
                augmented_bbox = transform_output["bboxes"][0]
            else:
                augmented_bbox = original_bbox
        else:
            transform_output = self.image_transforms(image_path)
            pixel_values = (
                transform_output["pixel_values"]
                if isinstance(transform_output, dict)
                else transform_output
            )

        prompt, report = "", ""
        if prompt_mode == "locate_and_describe":
            prompt = f"Locate and describe the {location}."
            bbox_str = _format_bbox_string(augmented_bbox)
            report = f"Location of the {location}: {bbox_str}. Description: {instance['report_snippet']}"
        elif prompt_mode == "describe":
            prompt = f"Describe the {location}."
            report = f"Description of the {location}: {instance['report_snippet']}"
        elif prompt_mode == "locate":
            prompt = f"Locate the {location}."
            bbox_str = _format_bbox_string(augmented_bbox)
            report = f"Location of the {location}: {bbox_str}"
        else:
            raise ValueError(f"Invalid prompt mode '{prompt_mode}'")

        output = {"pixel_values": pixel_values, "prompt": prompt, "report": report}
        
        # Expose the anatomical location for downstream analysis.
        output["location"] = location

        if self.return_image_path:
            output["image_path"] = image_path

        if self.diagnostic_mode:
            output["diagnostic_orig_index"] = orig_idx
            output["diagnostic_actual_index"] = idx
            output["diagnostic_dataset_name"] = self.dataset_name
            output["diagnostic_category"] = location

        return output

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function to handle batching of data samples."""
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        pixel_values = pixel_values.unsqueeze(1)

        prompts = [item["prompt"] for item in batch]
        reports = [item["report"] for item in batch]

        output = {"pixel_values": pixel_values}

        if self.tokenizer_fn is not None:
            try:
                tokenized_data = self.tokenizer_fn(prompts, reports)
                if not isinstance(tokenized_data, dict):
                    raise TypeError(
                        f"tokenizer_fn must return a dict, but got {type(tokenized_data)}."
                    )
                output.update(tokenized_data)
            except Exception as e:
                logger.error(f"Error in tokenizer_fn: {e}", exc_info=True)
                raise e
        else:
            output["prompts"] = prompts
            output["reports"] = reports

        if self.return_image_path:
            output["image_paths"] = [item["image_path"] for item in batch]

        return output


# ----------------------------------------------------------------------------
# 2. Specialized Dataset Classes per Mode
# ----------------------------------------------------------------------------
class LocateDataset(_ChestImaGenomeBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, dataset_name="cig_locate")
        self._create_instances()
        logger.info(f"Found {len(self.instances)} instances for mode 'locate'.")
        self._finalize_initialization()

    def _create_instances(self):
        for study in tqdm(
            self.study_data_for_split,
            desc="Creating 'locate' instances",
            disable=self.disable_tqdm,
        ):
            location2bbox = study["location2bbox"]
            for loc, bbox in location2bbox.items():
                if loc == "unknown":
                    continue
                
                valid_bbox = self._clamp_and_validate_bbox(bbox)
                if valid_bbox:
                    base_instance = {"dicom_id": study["dicom_id"], "location": loc}
                    self.instances.append(
                        self._build_instance(
                            base_instance, "locate", bbox_xyxy=valid_bbox
                        )
                    )


class DescribeDataset(_ChestImaGenomeBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, dataset_name="cig_describe")
        self._create_instances()
        logger.info(f"Found {len(self.instances)} instances for mode 'describe'.")
        self._finalize_initialization()

    def _create_instances(self):
        for study in tqdm(
            self.study_data_for_split,
            desc="Creating 'describe' instances",
            disable=self.disable_tqdm,
        ):
            location2snippet = study["location2report_snippet"]
            for loc, snippet in location2snippet.items():
                if loc == "unknown":
                    continue
                snippet = " ".join(snippet.strip().split())
                if not snippet:
                    continue
                base_instance = {"dicom_id": study["dicom_id"], "location": loc}
                self.instances.append(
                    self._build_instance(base_instance, "describe", snippet=snippet)
                )


class LocateAndDescribeDataset(_ChestImaGenomeBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, dataset_name="cig_locate_and_describe")
        self._create_instances()
        logger.info(f"Found {len(self.instances)} instances for mode 'locate_and_describe'.")
        self._finalize_initialization()

    def _create_instances(self):
        for study in tqdm(
            self.study_data_for_split,
            desc="Creating 'locate_and_describe' instances",
            disable=self.disable_tqdm,
        ):
            dicom_id = study["dicom_id"]
            location2bbox = study["location2bbox"]
            location2snippet = study["location2report_snippet"]
            
            valid_locations = set(location2bbox.keys()) & set(location2snippet.keys())
            valid_locations.discard("unknown")

            for loc in valid_locations:
                # First, validate snippet
                snippet = " ".join(location2snippet[loc].strip().split())
                if not snippet:
                    continue
                
                # Then, validate bbox using the helper
                bbox = location2bbox[loc]
                valid_bbox = self._clamp_and_validate_bbox(bbox)
                
                if valid_bbox:
                    base_instance = {"dicom_id": dicom_id, "location": loc}
                    self.instances.append(
                        self._build_instance(
                            base_instance,
                            "locate_and_describe",
                            bbox_xyxy=valid_bbox,
                            snippet=snippet,
                        )
                    )


# ----------------------------------------------------------------------------
# 3. Factory Function to Assemble the Final Dataset
# ----------------------------------------------------------------------------
def create_chest_imagenome_dataset(
    image_transforms_kwargs: Dict[str, Any],
    split: Literal["train", "val", "test"],
    prompt_modes: List[PromptMode],
    prompt_mode_weights: Optional[Dict[PromptMode, float]] = None,
    bbox_format: Literal["xyxy", "cxcywh"] = "xyxy",
    return_image_path: bool = False,
    tokenizer_fn: Optional[BatchTokenizerFn] = None,
    disable_tqdm: bool = False,
    use_weighted_sampling: bool = False,
    diagnostic_mode: bool = False,
) -> Dataset:
    """
    Factory function to create and assemble the Chest-ImaGenome dataset.

    For the 'train' split, it uses WeightedCompositeDataset for balanced sampling.
    For 'val' and 'test' splits, it uses ConcatDataset for sequential iteration.

    Args:
        image_transforms_kwargs: Kwargs for `create_image_transforms`.
        split: The dataset split ('train', 'val', 'test').
        prompt_modes: A list of prompt modes to use.
        prompt_mode_weights: (For 'train' split) A dict mapping each prompt
                             mode to a sampling weight.
        bbox_format: Target format for bounding boxes ('xyxy' or 'cxcywh').
        return_image_path: If True, include the image path in the output.
        tokenizer_fn: Optional function to tokenize prompts and reports.
        disable_tqdm: If True, disables the tqdm progress bar.
        use_weighted_sampling: If True, use weighted sampling to balance the dataset.
        diagnostic_mode: If True, include diagnostic information in the output.
    Returns:
        A PyTorch Dataset instance (either WeightedCompositeDataset or ConcatDataset)
        with a `collate_fn` method attached.
    """
    # Pre-load data once to share across specialized datasets
    base_ds = _ChestImaGenomeBaseDataset(
        image_transforms_kwargs={}, split=split
    )
    dicom_map = base_ds.dicom_id_to_image_path_map
    study_data = base_ds.study_data_for_split
    del base_ds

    # Map mode names to their respective classes
    dataset_class_map = {
        "locate": LocateDataset,
        "describe": DescribeDataset,
        "locate_and_describe": LocateAndDescribeDataset,
    }

    # Common arguments for all specialized dataset constructors
    shared_kwargs = {
        "image_transforms_kwargs": image_transforms_kwargs,
        "split": split,
        "bbox_format": bbox_format,
        "return_image_path": return_image_path,
        "tokenizer_fn": tokenizer_fn,
        "dicom_id_to_image_path_map": dicom_map,
        "study_data_for_split": study_data,
        "disable_tqdm": disable_tqdm,
        "use_weighted_sampling": use_weighted_sampling,
        "diagnostic_mode": diagnostic_mode,
    }

    # Instantiate only the required datasets
    datasets_to_compose = []
    for mode in prompt_modes:
        dataset_class = dataset_class_map[mode]
        datasets_to_compose.append(dataset_class(**shared_kwargs))

    if not datasets_to_compose:
        raise ValueError("No valid prompt modes specified or no data found.")

    # --- Assemble the final dataset ---
    if split == "train":
        if not prompt_mode_weights:
            raise ValueError("`prompt_mode_weights` must be provided for 'train' split.")
        
        weights = [prompt_mode_weights.get(ds.instances[0]['prompt_mode'], 1) for ds in datasets_to_compose]
        
        logger.info("Creating WeightedCompositeDataset for training.")
        logger.info(f"Using weights: {weights}")
        composite_dataset = CustomWeightedCompositeDataset(datasets=datasets_to_compose, dataset_names=prompt_modes, weights=weights)
    else:
        logger.info(f"Creating SubsampleableConcatDataset for {split} split.")
        composite_dataset = SubsampleableConcatDataset(datasets_to_compose)

    # Attach the collate_fn to the composite dataset for the DataLoader.
    # This is safe because all child datasets share the same collate_fn from the base class.
    composite_dataset.collate_fn = datasets_to_compose[0].collate_fn

    return composite_dataset
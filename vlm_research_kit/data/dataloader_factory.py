import logging
from typing import Any, Dict, Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import PreTrainedTokenizerBase

from vlm_research_kit.data.datasets.chest_imagenome_dataset import ChestImaGenomeDataset
from vlm_research_kit.data.datasets.mimiccxr_dataset import MIMICCXRDataset
from vlm_research_kit.data.datasets.padchest_dataset import PadChestGRDataset
from vlm_research_kit.data.samplers import BatchesPerEpochSampler, SubsetRandomSampler
from vlm_research_kit.training.model_specific_training_logic import get_batch_tokenizer_strategy
from vlm_research_kit.utils.logging_utils import ANSI_BLACK_BOLD, ANSI_RESET

logger = logging.getLogger(__name__)

# Dictionary mapping user-friendly dataset names to their corresponding classes.
SUPPORTED_DATASETS: Dict[str, type[Dataset]] = {
    "padchest-gr": PadChestGRDataset,
    "mimic-cxr": MIMICCXRDataset,
    "chest-imagenome": ChestImaGenomeDataset,
    # Add other dataset names and classes here
}

# --- Helper Function (Optional but Recommended) ---
def _instantiate_dataset(
    dataset_name: str, dataset_kwargs: Dict[str, Any]
) -> Dataset:
    """Instantiates a single dataset instance."""
    dataset_class = SUPPORTED_DATASETS.get(dataset_name.lower())
    if dataset_class is None:
        logger.error(f"Dataset '{dataset_name}' is not supported.")
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Supported datasets are: "
            f"{list(SUPPORTED_DATASETS.keys())}"
        )

    logger.info(
        f"Instantiating dataset '{dataset_name}' with arguments: "
        f"{list(dataset_kwargs.keys())}"
    )
    try:
        # Ensure dataset_kwargs contains necessary info like split, paths,
        # and potentially transform configurations.
        dataset = dataset_class(**dataset_kwargs)
        if len(dataset) == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' instantiated but is empty. "
                f"Check the dataset split and file paths."
            )
        logger.info(
            f"Dataset '{dataset_name}' instantiated successfully. Size: {len(dataset)}"
        )
        return dataset
    except (TypeError, FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.error(
            f"Failed to instantiate dataset '{dataset_name}'. "
            f"Error: {e}. Check dataset_kwargs: {dataset_kwargs}",
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during dataset instantiation: {e}",
            exc_info=True,
        )
        raise

# --- Inference Data Loader Creator ---
def create_inference_dataloader(
    dataset_name: str,
    dataset_kwargs: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    Factory function to create a DataLoader for inference or evaluation.

    Instantiates the specified dataset and wraps it in a DataLoader suitable
    for inference (no shuffling).

    Args:
        dataset_name: Name of the dataset (key in `SUPPORTED_DATASETS`).
        dataset_kwargs: Dictionary of keyword arguments for the dataset class
                        constructor (e.g., split='test', file paths, transforms).
        batch_size: Samples per batch.
        num_workers: Subprocesses for data loading.
        collate_fn: (Optional) Custom collate function. If None, checks for
                    `dataset.collate_fn`, otherwise uses default DataLoader collate.

    Returns:
        A configured PyTorch DataLoader instance.

    Raises:
        ValueError: If `dataset_name` is not supported.
        (Other exceptions from dataset instantiation or DataLoader).
    """
    logger.info(
        f"Attempting to create inference dataloader for dataset: '{dataset_name}'"
    )
    dataset = _instantiate_dataset(dataset_name, dataset_kwargs)

    # Determine Collate Function
    effective_collate_fn = collate_fn
    if effective_collate_fn is None and hasattr(dataset, "collate_fn") and callable(dataset.collate_fn):
        logger.info(f"Using collate_fn provided by the '{dataset_name}' dataset instance.")
        effective_collate_fn = dataset.collate_fn
    elif effective_collate_fn is not None:
        logger.info("Using collate_fn explicitly passed to create_inference_dataloader.")
    else:
        logger.info("Using default DataLoader collate_fn.")

    # Create DataLoader
    logger.info(
        f"Creating Inference DataLoader with batch_size={batch_size}, num_workers={num_workers}"
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False, # No shuffling for inference/evaluation
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=effective_collate_fn,
        # persistent_workers=(num_workers > 0), # Consider enabling
    )

    logger.info("Inference DataLoader created successfully.")
    return dataloader


# --- Training/Validation Data Loader Creator ---

def create_train_val_dataloaders(
    train_dataset_config: Dict[str, Any],
    val_dataset_config: Dict[str, Any],
    train_batches_per_epoch: Optional[int],
    tokenizer: PreTrainedTokenizerBase,
    train_tokenization_config: Dict[str, Any],
    generator: Optional[torch.Generator] = None, # Add generator for reproducibility,
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create training and validation DataLoaders.

    Parses configuration dictionaries to instantiate and wrap training and
    validation datasets into DataLoaders suitable for a training loop.

    Args:
        train_dataset_config: Dictionary defining the training dataset. Expected structure:
            {
                "name": "dataset_name",
                "constructor_kwargs": { ... } # Args for dataset __init__
                "batch_size": int
                "num_workers": int
            }
        val_dataset_config: Dictionary defining the validation dataset. Expected structure:
            {
                "name": "dataset_name",
                "constructor_kwargs": { ... } # Args for dataset __init__
                "batch_size": int
                "num_workers": int
            }
        tokenizer: PreTrainedTokenizerBase instance for tokenization.
        train_tokenization_config: Dictionary with tokenization parameters for training.
        train_batches_per_epoch: Optional int specifying the number of batches per epoch for training.

    Returns:
        A tuple containing:
            - train_dataloader: The DataLoader for the training set (shuffled).
            - val_dataloader: The DataLoader for the validation set (not shuffled).

    Raises:
        KeyError: If required keys are missing in the config dictionaries.
        ValueError: If dataset names are not supported.
        (Other exceptions from dataset instantiation or DataLoader).
    """
    logger.info("Creating training and validation dataloaders...")

    # --- Parse Loader Configuration ---
    try:
        train_batch_size = train_dataset_config["batch_size"]
        train_num_workers = train_dataset_config["num_workers"]
        train_dataset_name = train_dataset_config["name"]
        train_dataset_kwargs = train_dataset_config["constructor_kwargs"].copy() # Copy to avoid mutation
        
        val_batch_size = val_dataset_config["batch_size"]
        val_num_workers = val_dataset_config["num_workers"]
        val_dataset_name = val_dataset_config["name"]
        val_epoch_sample_size = val_dataset_config.get("epoch_sample_size", None) # Optional
        val_dataset_kwargs = val_dataset_config["constructor_kwargs"].copy() # Copy to avoid mutation
    except KeyError as e:
        logger.error(f"Missing required key in configuration: {e}")
        raise

    # --- Create Training DataLoader ---
    logger.info(f"{ANSI_BLACK_BOLD}--- Creating Training DataLoader ---{ANSI_RESET}")

    # Set up training dataset
    train_dataset_kwargs["tokenizer_fn"] = get_batch_tokenizer_strategy(tokenization_config=train_tokenization_config,
                                                                        tokenizer=tokenizer)
    train_dataset = _instantiate_dataset(train_dataset_name, train_dataset_kwargs)

    # Retrieve collate function from dataset
    assert hasattr(train_dataset, "collate_fn") and callable(train_dataset.collate_fn), (
        "Training dataset must have a callable collate_fn. "
    )
    collate_fn_train = train_dataset.collate_fn

    # Handle train_batches_per_epoch
    train_sampler: Optional[Sampler[int]] = None
    shuffle_train: bool = True # Default: shuffle full dataset
    if isinstance(train_batches_per_epoch, int) and train_batches_per_epoch > 0:
        if train_dataset is None or len(train_dataset) == 0:
            logger.warning("Cannot use BatchesPerEpochSampler with empty dataset. Using default loader.")
        else:
            logger.info(
                f"Using BatchesPerEpochSampler for {train_batches_per_epoch} batches per epoch."
            )
            train_sampler = BatchesPerEpochSampler(
                data_source=train_dataset,
                batches_per_epoch=train_batches_per_epoch,
                batch_size=train_batch_size,
                generator=generator,
            )
            # IMPORTANT: shuffle must be False when a sampler is provided
            shuffle_train = False
    elif train_batches_per_epoch is not None:
        logger.warning(
            f"Invalid value for 'train_batches_per_epoch' ({train_batches_per_epoch}). "
            f"Ignoring and using standard shuffled DataLoader."
        )

    # Create DataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle_train, # Set based on whether sampler is used
        sampler=train_sampler, # Pass sampler if created
        num_workers=train_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_train,
        persistent_workers=(train_num_workers > 0),
        # drop_last=True, # Consider uncommenting if needed
        generator=generator, # Pass generator for worker seeding
    )
    logger.info("Training DataLoader created successfully.")

    # --- Create Validation DataLoader ---
    logger.info(f"{ANSI_BLACK_BOLD}--- Creating Validation DataLoader ---{ANSI_RESET}")
    
    # Set up validation dataset
    val_dataset = _instantiate_dataset(val_dataset_name, val_dataset_kwargs)

    # Retrieve collate function from dataset
    assert hasattr(val_dataset, "collate_fn") and callable(val_dataset.collate_fn), (
        "Validation dataset must have a callable collate_fn. "
    )
    collate_fn_val = val_dataset.collate_fn

     # Conditionally create the custom sampler for validation
    val_sampler: Optional[Sampler[int]] = None
    if isinstance(val_epoch_sample_size, int) and val_epoch_sample_size > 0:
        logger.info(
            f"Using SubsetRandomSampler for validation with a sample size of {val_epoch_sample_size} per epoch."
        )
        val_sampler = SubsetRandomSampler(
            data_source=val_dataset,
            sample_size=val_epoch_sample_size,
            generator=generator, # Pass generator for reproducibility
        )
    elif val_epoch_sample_size is not None:
        logger.warning(
            f"Invalid value for 'epoch_sample_size' ({val_epoch_sample_size}). "
            f"Ignoring and using standard sequential validation."
        )

    # Create DataLoader
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False, # No shuffling for validation
        sampler=val_sampler, # Pass the custom sampler if created
        num_workers=val_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_val,
        persistent_workers=(val_num_workers > 0),
        drop_last=False, # Keep all validation samples
    )
    logger.info("Validation DataLoader created successfully.")
    
    # Return DataLoaders
    return train_dataloader, val_dataloader


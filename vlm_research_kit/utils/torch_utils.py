import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import transformers
import numpy as np
import random
import time
import logging
from typing import Dict, List, Optional, Union, Any
import re

from vlm_research_kit.utils.logging_utils import ANSI_RED_BOLD, ANSI_RESET

logger = logging.getLogger(__name__)


def get_compute_device(training_config: Optional[Dict[str, Any]] = None) -> torch.device:
    """
    Determines the best available compute device based on availability and
    an optional training configuration.

    Order of preference if config is not provided or doesn't specify:
    1. CUDA
    2. MPS (for Apple Silicon)
    3. CPU

    If training_config is provided and contains 'device_type', that will be
    honored if the device is available, otherwise it falls back.

    Args:
        training_config (Optional[Dict[str, Any]]): The 'training' section
            of the main configuration dictionary, which might contain a
            'device_type' key (e.g., "cuda", "mps", "cpu").

    Returns:
        torch.device: The selected torch.device object.
    """
    preferred_device_type = None
    if training_config and isinstance(training_config, dict):
        preferred_device_type = training_config.get("device_type")

    selected_device_type = "cpu" # Default

    if preferred_device_type:
        preferred_device_type = preferred_device_type.lower()
        if preferred_device_type == "cuda":
            if torch.cuda.is_available():
                selected_device_type = "cuda"
            else:
                logger.warning(
                    "Config specified 'cuda' but CUDA is not available. Falling back."
                )
        elif preferred_device_type == "mps":
            if torch.backends.mps.is_available():
                selected_device_type = "mps"
            else:
                logger.warning(
                    "Config specified 'mps' but MPS is not available. Falling back."
                )
        elif preferred_device_type == "cpu":
            selected_device_type = "cpu"
        else:
            logger.warning(
                f"Unknown device_type '{preferred_device_type}' in config. "
                "Attempting auto-detection."
            )
            # Fall through to auto-detection if preferred is unknown or unavailable

    # If no preference from config, or preferred was unavailable/unknown, auto-detect
    if selected_device_type == "cpu" and not (preferred_device_type and preferred_device_type == "cpu"): # Avoid re-checking if CPU was explicitly preferred
        if torch.cuda.is_available():
            selected_device_type = "cuda"
        elif torch.backends.mps.is_available():
            selected_device_type = "mps"
        # else remains "cpu"

    return torch.device(selected_device_type)


def freeze_model_parts(
    model: nn.Module,
    freeze_config: Optional[Dict[str, Union[bool, List[str], str]]] = None
) -> int:
    """
    Freezes specified parts of a model based on configuration.

    Sets `requires_grad = False` for parameters matching the criteria.

    Args:
        model: The model to modify.
        freeze_config: A dictionary specifying what to freeze. Examples:
            {
                "freeze_all": False, # If True, freezes the entire model initially
                "unfreeze_patterns": ["^classifier\\.", "^projection\\."], # Regex patterns to unfreeze
                "freeze_patterns": ["^vision_encoder\\."], # Regex patterns to freeze (overrides unfreeze)
                "unfreeze_modules": ["encoder.layer.0"], # Specific module names to unfreeze
                "freeze_modules": ["encoder.layer.11"], # Specific module names to freeze
                # Add more specific flags like "freeze_vision_encoder": True if needed
            }

    Returns:
        The number of parameters that remain trainable.
    """
    if not freeze_config:
        logger.info("No freeze configuration provided. All parameters are trainable by default.")
        # Ensure all params are trainable initially if no config
        for param in model.parameters():
            param.requires_grad = True
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = 0

    # Initial state based on freeze_all
    initial_freeze_state = freeze_config.get("freeze_all", False)
    for name, param in model.named_parameters():
        param.requires_grad = not initial_freeze_state
        total_params += param.numel()

    logger.info(f"Initial freeze state (freeze_all={initial_freeze_state}): All params require_grad = {not initial_freeze_state}")

    # Unfreeze based on patterns
    unfreeze_patterns = freeze_config.get("unfreeze_patterns", [])
    if initial_freeze_state and unfreeze_patterns: # Only unfreeze if initially frozen
        compiled_unfreeze_patterns = [re.compile(p) for p in unfreeze_patterns]
        for name, param in model.named_parameters():
            if any(pattern.match(name) for pattern in compiled_unfreeze_patterns):
                if not param.requires_grad:
                    logger.debug(f"Unfreezing parameter: {name}")
                    param.requires_grad = True

    # Freeze based on patterns (takes precedence over unfreeze)
    freeze_patterns = freeze_config.get("freeze_patterns", [])
    if freeze_patterns:
        compiled_freeze_patterns = [re.compile(p) for p in freeze_patterns]
        for name, param in model.named_parameters():
             if any(pattern.match(name) for pattern in compiled_freeze_patterns):
                if param.requires_grad:
                    logger.debug(f"Freezing parameter: {name}")
                    param.requires_grad = False

    # Unfreeze specific modules by name
    unfreeze_modules = freeze_config.get("unfreeze_modules", [])
    if unfreeze_modules:
        for module_name_to_unfreeze in unfreeze_modules:
            try:
                module = dict(model.named_modules())[module_name_to_unfreeze]
                for name, param in module.named_parameters():
                    full_name = f"{module_name_to_unfreeze}.{name}"
                    if not param.requires_grad:
                        logger.debug(f"Unfreezing parameter within module '{module_name_to_unfreeze}': {full_name}")
                        param.requires_grad = True
            except KeyError:
                logger.warning(f"Module name '{module_name_to_unfreeze}' not found for unfreezing.")

    # Freeze specific modules by name
    freeze_modules = freeze_config.get("freeze_modules", [])
    if freeze_modules:
        for module_name_to_freeze in freeze_modules:
            try:
                module = dict(model.named_modules())[module_name_to_freeze]
                for name, param in module.named_parameters():
                    full_name = f"{module_name_to_freeze}.{name}"
                    if param.requires_grad:
                        logger.debug(f"Freezing parameter within module '{module_name_to_freeze}': {full_name}")
                        param.requires_grad = False
            except KeyError:
                logger.warning(f"Module name '{module_name_to_freeze}' not found for freezing.")

    # Log final state and count trainable parameters
    frozen_params_list = []
    trainable_params_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params_count += param.numel()
        else:
            frozen_params_list.append(name)

    logger.info(f"Parameter freezing applied. {trainable_params_count}/{total_params} parameters are trainable.")
    if frozen_params_list:
        logger.info(f"Frozen parameters/prefixes: {', '.join(list(set(p.split('.')[0] for p in frozen_params_list)))}...") # Log top-level modules frozen

    return trainable_params_count


def _check_module_trainability_recursive(module: nn.Module, base_name: str = ""):
    """
    Recursive helper function to check and log fully frozen or fully trainable modules.

    Args:
        module: The current nn.Module being inspected.
        base_name: The full name path to this module from the root model.
    """
    has_parameters = False
    all_frozen = True
    all_trainable = True

    # Iterate through all parameters *within* this module and its descendants
    # module.parameters(recurse=True) gets all params in module + submodules
    for param in module.parameters(recurse=True):
        has_parameters = True
        if param.requires_grad:
            all_frozen = False # Found a trainable param, so not fully frozen
        else:
            all_trainable = False # Found a frozen param, so not fully trainable

        # Optimization: If we've already determined it's neither fully frozen nor fully trainable,
        # we can potentially break early, though the loop is still needed to set has_parameters
        # correctly based on *any* parameter existence.

    # --- Decision Logic ---

    # If the module has parameters:
    if has_parameters:
        # 1. Check if the module's subtree is fully frozen
        if all_frozen:
            logger.info(f"Module fully frozen ❄️: {base_name if base_name else 'model (root)'}")
            # Interrupt recursion for this branch as it's fully characterized
            return

        # 2. Check if the module's subtree is fully trainable
        if all_trainable:
            logger.info(f"Module fully trainable 💪: {base_name if base_name else 'model (root)'}")
            # Interrupt recursion for this branch as it's fully characterized
            return

    # If the module has parameters and is neither fully frozen nor fully trainable,
    # OR if the module has no parameters, recursively explore its immediate children.
    for child_name, child_module in module.named_children():
        # Construct the full name for the child module
        full_child_name = f"{base_name}.{child_name}" if base_name else child_name
        # Recursively call the function for the child
        _check_module_trainability_recursive(child_module, full_child_name)


def log_module_trainability(model: nn.Module):
    """
    Recursively traverses a PyTorch model and logs the names of modules
    that are either fully frozen or fully trainable.

    A module is considered "fully frozen" if it contains parameters, and all
    parameters within that module AND all of its submodules have
    `requires_grad=False`.

    A module is considered "fully trainable" if it contains parameters, and all
    parameters within that module AND all of its submodules have
    `requires_grad=True`.

    If a module's subtree is neither fully frozen nor fully trainable (i.e.,
    it contains a mix of trainable and frozen parameters), this function will
    recursively check its immediate children.

    Modules with no parameters (like activation functions) are effectively
    skipped in the output, but their children (if any) are still traversed.

    Args:
        model: The PyTorch model (nn.Module) to inspect.
    """
    logger.info("--- Checking for fully frozen and fully trainable modules ---")
    # Start recursion with the root model and an empty base name
    _check_module_trainability_recursive(model, "")
    logger.info("--- Finished checking ---")


def unnormalize_image(tensor, mean, std):
    """Reverses the normalization on a tensor image."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    img = TF.to_pil_image(tensor)
    return img


def turn_on_determinism(seed: int = 42):
    """
    Sets random seeds and deterministic flags for reproducible results
    in PyTorch, NumPy, and Python's random module, including
    torch.use_deterministic_algorithms(True).

    Args:
        seed: The integer seed value to use. Defaults to 42.
    """

    logger.info(f"{ANSI_RED_BOLD}Turning on determinism with seed: {seed}{ANSI_RESET}")

    # 1. Python's built-in random
    random.seed(seed)
    logger.debug(f"Python random seed set to {seed}")

    # 2. NumPy
    np.random.seed(seed)
    logger.debug(f"NumPy random seed set to {seed}")

    # 3. PyTorch
    torch.manual_seed(seed)
    logger.debug(f"PyTorch CPU seed set to {seed}")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        logger.debug(f"PyTorch CUDA seeds set to {seed}")

        # Set cuDNN for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("PyTorch cuDNN deterministic flags set (deterministic=True, benchmark=False)")

        # Set environment variable for CuBLAS determinism BEFORE enabling global deterministic algorithms
        # Use :4096:8 first, try :16:8 if you encounter memory issues.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        logger.debug(f"Set CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}")
    
    else:
        logger.debug("No CUDA available, skipping cuDNN deterministic settings.")

    # 4. Enable deterministic algorithms globally
    try:
        torch.use_deterministic_algorithms(True)
        logger.debug("PyTorch deterministic algorithms enabled globally.")
    except Exception as e:
        logger.warning(f"Could not enable torch.use_deterministic_algorithms(True): {e}. "
                       "Some operations might still be non-deterministic.")

    # 5. Hugging Face transformers (optional)
    # If you use Hugging Face models, you might want to uncomment this
    try:
        transformers.set_seed(seed)
        logger.debug(f"Hugging Face transformers seed set to {seed}")
    except Exception as e:
        logger.warning(f"Could not set Hugging Face seed: {e}")


def turn_off_determinism():
    """
    Attempts to revert random seed settings and PyTorch deterministic flags
    to potentially improve performance after a deterministic section.
    """

    logger.info(f"{ANSI_RED_BOLD}Turning off determinism.{ANSI_RESET}")

    # Restore cuDNN settings to non-deterministic
    if torch.cuda.is_available():
        # Set common non-deterministic settings
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.debug("PyTorch cuDNN settings restored (deterministic=False, benchmark=True)")
    else:
        logger.debug("No CUDA available, skipping cuDNN settings restoration.")

    # Restore original deterministic algorithms flag state
    try:
        torch.use_deterministic_algorithms(False)
        logger.debug("PyTorch deterministic algorithms disabled.")
    except Exception as e:
        logger.warning(f"Could not disable torch.use_deterministic_algorithms(False): {e}. "
                       "Some operations might still be non-deterministic.")

    # Reset seeds using time-based randomness to ensure different runs
    new_seed = int(time.time()) % (2**32 - 1)
    torch.manual_seed(new_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(new_seed)
    random.seed(new_seed)
    np.random.seed(new_seed)
    transformers.set_seed(new_seed) # If using Hugging Face transformers
    logger.debug(f"Random seeds reset to time-based seed: {new_seed}")

    # Unset environment variables if you set them in turn_on_determinism
    if 'CUBLAS_WORKSPACE_CONFIG' in os.environ:
        del os.environ['CUBLAS_WORKSPACE_CONFIG']
        logger.debug("Unset CUBLAS_WORKSPACE_CONFIG")


def seed_random_by_process_rank(base_seed: int = 42):
    """
    Sets random seeds based on the current process rank for distributed training.
    
    This function ensures that each process in a distributed training setup gets
    a different but deterministic seed, which is crucial for proper data loading
    and model initialization across multiple processes.
    
    The seed is calculated as: base_seed + process_rank
    
    Args:
        base_seed: The base seed value to use. Each process will get base_seed + rank.
                  Defaults to 42.
    
    Returns:
        int: The actual seed value that was set for this process.
    """
    # Get the current process rank from environment variables
    # LOCAL_RANK is set by torchrun for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Calculate the process-specific seed
    process_seed = base_seed + local_rank
    
    logger.info(f"{ANSI_RED_BOLD}Seeding random with process rank {local_rank}, seed: {process_seed}{ANSI_RESET}")
    
    # Set seeds for all random number generators
    # 1. Python's built-in random
    random.seed(process_seed)
    logger.debug(f"Python random seed set to {process_seed} (rank {local_rank})")
    
    # 2. NumPy
    np.random.seed(process_seed)
    logger.debug(f"NumPy random seed set to {process_seed} (rank {local_rank})")
    
    # 3. PyTorch
    torch.manual_seed(process_seed)
    logger.debug(f"PyTorch CPU seed set to {process_seed} (rank {local_rank})")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)
        torch.cuda.manual_seed_all(process_seed)  # for multi-GPU
        logger.debug(f"PyTorch CUDA seeds set to {process_seed} (rank {local_rank})")
    
    # 4. Hugging Face transformers
    try:
        transformers.set_seed(process_seed)
        logger.debug(f"Hugging Face transformers seed set to {process_seed} (rank {local_rank})")
    except Exception as e:
        logger.warning(f"Could not set Hugging Face seed: {e}")
    
    return process_seed

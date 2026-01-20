import torch
import torch.nn as nn
import logging
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from vlm_research_kit.settings import EXPERIMENTS_DIR
from vlm_research_kit.training.factories import create_optimizer, create_scheduler
from vlm_research_kit.utils.file_utils import save_yaml
from vlm_research_kit.utils.hf_model_patches import apply_model_specific_patches
from vlm_research_kit.utils.logging_utils import ANSI_BLACK_BOLD, ANSI_BLUE_BOLD, ANSI_MAGENTA_BOLD, ANSI_RESET
from vlm_research_kit.utils.torch_utils import freeze_model_parts, log_module_trainability

logger = logging.getLogger(__name__)

# --- Save Checkpoint State ---
def save_checkpoint(
    output_dir: Union[str, Path],
    epoch: int,
    model: nn.Module, # Can be base model or PeftModel
    optimizer: Optimizer,
    lr_scheduler: Optional[_LRScheduler],
    current_score: float,
    best_score: float,
    config: Dict[str, Any],
    save_last: bool = True,
    is_best: bool = False,
) -> None:
    """
    Saves a training checkpoint. Handles both standard and LoRA (PEFT) models.

    For LoRA models, saves adapter weights to separate subdirectories ('adapter_last',
    'adapter_best') and excludes the full model state_dict from the .pth file.
    The .pth files ('last.pth', 'best.pth') store metadata and optimizer/scheduler state.

    Args:
        output_dir: Directory to save the checkpoint files.
        epoch: The completed epoch number.
        model: The model instance.
        optimizer: The optimizer instance.
        lr_scheduler: The learning rate scheduler instance (optional).
        current_score: The validation score for the current epoch.
        best_score: The best validation score achieved so far.
        config: The main training configuration dictionary.
        save_last: If True, save/overwrite 'last.pth'.
        is_best: If True, save/overwrite 'best.pth'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    is_peft_model = isinstance(model, PeftModel)
    log_prefix = "PEFT" if is_peft_model else "Standard"
    logger.debug(f"Saving checkpoint for {log_prefix} model.")

    # --- Prepare State Dictionary (Common parts) ---
    if save_last or is_best:
        state = {
            'epoch': epoch,
            'best_score': best_score,
            'current_score': current_score,
            # 'model_state_dict': model.state_dict(), # Not saving full model state for PEFT
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'config': config, # Embed config for easy resumption
            'is_peft_checkpoint': is_peft_model,
        }
        # --- Add Model State (Only for non-PEFT models) ---
        if not is_peft_model:
            state['model_state_dict'] = model.state_dict()
    
    # --- Save last.pth ---
    if save_last:
        last_ckpt_path = output_dir / "last.pth"
        torch.save(state, last_ckpt_path)
        logger.info(f"Saved {log_prefix} metadata/state checkpoint to: {ANSI_BLACK_BOLD}{last_ckpt_path}{ANSI_RESET}")
        if is_peft_model:
            try:
                # Save current adapter weights to the 'adapter_last' subdir
                last_adapter_dir = output_dir / "adapter_last"
                last_adapter_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                model.save_pretrained(last_adapter_dir)
                logger.info(f"Saved LAST PEFT adapter weights to: {ANSI_BLUE_BOLD}{last_adapter_dir}{ANSI_RESET}")
            except Exception as e:
                logger.error(f"Failed to save LAST PEFT adapter weights: {e}", exc_info=True)
    
    # --- Save best.pth ---
    if is_best:
        best_ckpt_path = output_dir / "best.pth"
        torch.save(state, best_ckpt_path)
        logger.info(f"Saved BEST {log_prefix} metadata/state checkpoint to: {ANSI_MAGENTA_BOLD}{best_ckpt_path}{ANSI_RESET}")
        if is_peft_model:
            try:
                # Save current (best) adapter weights to the 'adapter_best' subdir
                best_adapter_dir = output_dir / "adapter_best"
                best_adapter_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                model.save_pretrained(best_adapter_dir)
                logger.info(f"Saved BEST PEFT adapter weights to: {ANSI_BLUE_BOLD}{best_adapter_dir}{ANSI_RESET}")
            except Exception as e:
                logger.error(f"Failed to save BEST PEFT adapter weights: {e}", exc_info=True)
    
    # --- Save config separately (if it's the first time) ---
    config_path = output_dir / "training_config.yaml"
    if not config_path.is_file():
        save_yaml(config, config_path, sort_keys=False)
        logger.info(f"Saved training configuration to: {config_path}")


# --- Load Checkpoint State ---
def load_checkpoint_state(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads the state dictionary from a .pth checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    logger.info(f"Loading checkpoint state from: {checkpoint_path}")
    try:
        # Load onto CPU first to avoid GPU memory issues
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint file did not contain a dictionary: {checkpoint_path}")
        logger.info(f"Checkpoint state loaded successfully. Keys: {list(state.keys())}")
        return state
    except Exception as e:
        logger.error(f"Failed to load checkpoint state from {checkpoint_path}: {e}", exc_info=True)
        raise


# --- Internal Helper Function (_load_model_tokenizer_from_state) ---
# This helper is primarily for loading FULL state dicts. We will bypass
# its model loading part when dealing with PEFT checkpoints.
def _load_model_tokenizer_from_state(
    state: Dict[str, Any], load_weights: bool = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, Dict]:
    """
    Internal helper to instantiate model, tokenizer, and optionally load weights
    from a state dict (intended for FULL model state dicts).

    Args:
        state: The loaded checkpoint state dictionary.
        load_weights: If True, attempt to load 'model_state_dict' from state.

    Returns:
        Tuple: (model, tokenizer, config) - Model is on CPU.
    """
    # 1. Extract config (required)
    config = state.get('config')
    if config is None or not isinstance(config, dict):
        raise ValueError("Checkpoint state does not contain a valid 'config' dictionary.")
    model_cfg = config.get("model", {})

    # 2. Instantiate Tokenizer (based on config)
    try:
        tokenizer_name = model_cfg.get("tokenizer_name_or_path") or model_cfg.get("model_name_or_path")
        if not tokenizer_name:
            raise ValueError("Could not determine tokenizer name from config.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logger.error(f"Helper: Failed to load tokenizer based on config: {e}", exc_info=True)
        raise

    # 3. Instantiate Model Architecture (based on config)
    model = None # Initialize model to None
    try:
        model_identifier = model_cfg.get("model_name_or_path")
        if not model_identifier:
            raise ValueError("Could not determine model name/type from config.")
        logger.info(f"Helper: Instantiating base model architecture for: {model_identifier}")
        model = AutoModel.from_pretrained(model_identifier, trust_remote_code=True)
        logger.info(f"Helper: Base model {type(model).__name__} instantiated.")
        apply_model_specific_patches(model, model_identifier) # Apply patches based on model type

    except Exception as e:
        logger.error(f"Helper: Failed to instantiate model architecture or apply patches: {e}", exc_info=True)
        raise

    # Ensure model was instantiated before proceeding
    if model is None:
        raise RuntimeError("Model instantiation failed unexpectedly.") # Should be caught above, but safety check

    # 4. Load Model State Dict (Only if requested and present)
    if load_weights:
        model_state_dict = state.get('model_state_dict')
        if model_state_dict is None:
            # This is expected if it's a PEFT checkpoint state
            logger.warning("Helper: 'model_state_dict' not found in state. Weights not loaded by helper.")
        else:
            logger.info("Helper: Attempting to load 'model_state_dict'...")
            try:
                # Handle potential 'module.' prefix
                if all(k.startswith('module.') for k in model_state_dict.keys()):
                    logger.debug("Helper: Removing 'module.' prefix from state dict keys.")
                    model_state_dict = {k[len('module.'):]: v for k, v in model_state_dict.items()}

                missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=True) # Use strict=True for full checkpoints
                if missing_keys: logger.warning(f"Helper: Missing keys loading model state: {missing_keys}")
                if unexpected_keys: logger.warning(f"Helper: Unexpected keys loading model state: {unexpected_keys}")
                logger.debug("Helper: Model state dict loaded successfully by helper.")
            except Exception as e:
                logger.error(f"Helper: Failed to load model state dict: {e}", exc_info=True)
                # Decide if this should be fatal when loading full weights
                raise

    return model, tokenizer, config


def _move_optimizer_state_to_device(optimizer_state: Dict[str, Any], device: Union[str, torch.device]):
    if not optimizer_state: return
    if 'state' in optimizer_state and isinstance(optimizer_state['state'], dict):
        for state_values in optimizer_state['state'].values():
            if isinstance(state_values, dict):
                for k, v in state_values.items():
                    if isinstance(v, torch.Tensor):
                        state_values[k] = v.to(device)
    else:
        logger.warning("Optimizer state dictionary structure not as expected. Cannot move tensors.")

# --- Load Training Components (for Training/Resuming) ---
def load_training_components(
    full_config: Dict[str, Any], # Pass the entire config dictionary
    device: Union[str, torch.device],
    resume_checkpoint_path: Optional[Union[str, Path]] = None,
    # pretrained_checkpoint_path is now handled via full_config['model'].get(...)
) -> Tuple[
    Union[PreTrainedModel, PeftModel], # model (on target device)
    PreTrainedTokenizerBase,           # tokenizer
    Optimizer,                         # optimizer (managing device params, state loaded to device)
    Optional[_LRScheduler],            # lr_scheduler
    int,                               # start_epoch
    float                              # best_score
]:
    """
    Loads or initializes all components needed for training, including model (patched,
    LoRA-wrapped/frozen), optimizer, and scheduler, all on the target device.
    Handles resuming from standard or LoRA checkpoints.
    """
    model_config = full_config["model"]
    optimizer_config = full_config["optimizer"]
    scheduler_config = full_config["lr_scheduler"]
    lora_config_dict = full_config.get("lora", {})
    is_lora_enabled_in_config = lora_config_dict.get("enabled", False)

    model: Union[PreTrainedModel, PeftModel]
    tokenizer: PreTrainedTokenizerBase
    optimizer: Optimizer
    lr_scheduler: Optional[_LRScheduler]
    start_epoch: int
    best_score: float
    optimizer_state_to_load: Optional[Dict[str, Any]] = None

    if resume_checkpoint_path:
        # --- Resume from Checkpoint ---
        resume_checkpoint_path = Path(resume_checkpoint_path)
        output_dir = resume_checkpoint_path.parent
        state = load_checkpoint_state(resume_checkpoint_path) # Loads to CPU
        config_from_ckpt = state['config'] # Use config from checkpoint for consistency
        is_peft_resume = state.get('is_peft_checkpoint', False)

        # Override current configs with those from checkpoint for critical parts
        # This ensures consistency if config file was changed since checkpoint
        model_config = config_from_ckpt.get("model", model_config)
        optimizer_config = config_from_ckpt.get("optimizer", optimizer_config)
        scheduler_config = config_from_ckpt.get("lr_scheduler", scheduler_config)
        # Check LoRA status from checkpoint config
        lora_config_from_ckpt = config_from_ckpt.get("lora", {})
        is_lora_enabled_in_ckpt_config = lora_config_from_ckpt.get("enabled", False)

        if is_peft_resume != is_lora_enabled_in_ckpt_config:
            logger.warning(
                f"Mismatch between checkpoint's PEFT status ({is_peft_resume}) and "
                f"its embedded config's LoRA status ({is_lora_enabled_in_ckpt_config}). "
                f"Trusting checkpoint's PEFT status."
            )
        if is_lora_enabled_in_config != is_lora_enabled_in_ckpt_config:
            logger.warning(
                f"Current training config LoRA status ({is_lora_enabled_in_config}) "
                f"differs from checkpoint's config LoRA status ({is_lora_enabled_in_ckpt_config}). "
                f"Using checkpoint's config for LoRA setup during resume."
            )
        # Use LoRA settings from the checkpoint's config for consistency
        lora_config_dict = lora_config_from_ckpt


        logger.info("Loading base model architecture and tokenizer for resume (CPU)...")
        # _load_model_tokenizer_from_state loads base model to CPU and applies patches
        base_model_cpu, tokenizer, _ = _load_model_tokenizer_from_state(state, load_weights=False)

        if is_peft_resume:
            adapter_dir_to_load = output_dir / "adapter_last"
            logger.info(f"Resuming PEFT. Loading adapter from: {adapter_dir_to_load} (CPU)")
            if not adapter_dir_to_load.exists(): raise FileNotFoundError(f"Adapter dir not found: {adapter_dir_to_load}")
            try:
                # Load adapter onto the CPU base model
                model = PeftModel.from_pretrained(base_model_cpu, adapter_dir_to_load, is_trainable=True)
                logger.info("PEFT adapter loaded to CPU model.")
            except Exception as e: logger.error(f"Failed to load PEFT adapter: {e}", exc_info=True); raise
        else: # Standard resume
            logger.info("Resuming standard checkpoint. Loading full model weights (CPU)...")
            model_state_dict = state.get('model_state_dict')
            if model_state_dict is None: raise ValueError("Standard checkpoint missing 'model_state_dict'.")
            try:
                if all(k.startswith('module.') for k in model_state_dict.keys()): model_state_dict = {k[len('module.'):]: v for k, v in model_state_dict.items()}
                base_model_cpu.load_state_dict(model_state_dict, strict=True)
                model = base_model_cpu # Model is the base model with weights loaded
            except Exception as e: logger.error(f"Failed to load full model state dict: {e}", exc_info=True); raise

        start_epoch = state.get('epoch', -1) + 1
        best_score = state.get('best_score', float('-inf'))
        if 'optimizer_state_dict' in state:
            optimizer_state_to_load = state['optimizer_state_dict'] # Keep on CPU for now
            logger.info("Optimizer state dict retrieved from checkpoint.")

        # Scheduler state will be loaded after optimizer is fully set up
        scheduler_state_to_load = state.get('scheduler_state_dict')

    else: # --- Initialize from Scratch ---
        logger.info("Initializing training components from scratch (CPU)...")
        start_epoch = 0
        best_score = float('-inf')
        scheduler_state_to_load = None # No scheduler state to load

        model_name_or_path = model_config["model_name_or_path"]
        logger.info(f"Loading base model from: {model_name_or_path}")
        model_cpu = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True) # Loads on CPU
        apply_model_specific_patches(model_cpu, model_name_or_path) # Patch CPU model

        pretrained_checkpoint_path = model_config.get("pretrained_checkpoint_path")
        if pretrained_checkpoint_path:
            logger.info(f"Loading pretrained base model weights from: {pretrained_checkpoint_path}")
            try: # Load weights into CPU model
                _pth = Path(pretrained_checkpoint_path)
                if not _pth.is_absolute(): _pth = Path(EXPERIMENTS_DIR) / _pth
                assert _pth.exists(), f"Pretrained .pth not found: {_pth}"
                _state = torch.load(_pth, map_location='cpu')
                _m_state = _state.get('model_state_dict')
                if not _m_state: raise KeyError("model_state_dict not in pretrained .pth")
                if all(k.startswith('module.') for k in _m_state.keys()): _m_state = {k[len('module.'):]: v for k, v in _m_state.items()}
                model_cpu.load_state_dict(_m_state, strict=True)
                logger.info("Successfully loaded pretrained base model weights.")
            except Exception as e: logger.error(f"Failed to load pretrained weights: {e}", exc_info=True); raise

        tokenizer_name_or_path = model_config.get("tokenizer_name_or_path", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # --- Handle LoRA or Standard Freezing (on CPU model) ---
        if is_lora_enabled_in_config:
            logger.info("LoRA is enabled in config. Applying PEFT (CPU)...")
            lora_actual_config = LoraConfig(
                r=lora_config_dict["r"],
                lora_alpha=lora_config_dict["lora_alpha"],
                target_modules=lora_config_dict["target_modules"],
                lora_dropout=lora_config_dict.get("lora_dropout", 0.0),
                bias=lora_config_dict.get("bias", "none"),
                modules_to_save=lora_config_dict.get("modules_to_save"),
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            model = get_peft_model(model_cpu, lora_actual_config) # Wraps CPU model
            logger.info("PEFT model created (CPU).")
            model.print_trainable_parameters()
        else: # Standard model, apply freezing if configured
            model = model_cpu # Assign the base model
            logger.info("LoRA not enabled. Applying standard freezing if configured (CPU)...")
            freeze_config = model_config.get("freeze_config")
            if freeze_config:
                trainable_param_count = freeze_model_parts(model, freeze_config) # model is on CPU
                logger.info(f"Standard freezing applied. {trainable_param_count} params trainable.")
            else:
                logger.info("No standard freeze configuration found.")
            log_module_trainability(model)

    # --- Common steps for both resume and scratch ---

    # 1. Move final model to target device
    logger.info(f"Moving final model to device: {device}...")
    model.to(device)
    logger.info("Model moved successfully.")

    # 2. Create Optimizer (linked to model parameters now on target device)
    logger.info(f"Creating optimizer for model on {device}...")
    optimizer = create_optimizer(optimizer_config, model) # model is now on device

    # 3. Load Optimizer State (if resuming and state exists)
    if optimizer_state_to_load:
        logger.info(f"Moving loaded optimizer state to device: {device}...")
        try:
            _move_optimizer_state_to_device(optimizer_state_to_load, device)
            logger.info("Loading optimizer state dict...")
            optimizer.load_state_dict(optimizer_state_to_load)
            logger.info("Optimizer state loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load optimizer state: {e}", exc_info=True)
            raise
    elif resume_checkpoint_path: # Resuming but no optimizer state found
        logger.warning("Resuming run, but no optimizer state found in checkpoint.")

    # 4. Create and Load Scheduler State
    logger.info("Creating LR scheduler...")
    lr_scheduler = create_scheduler(scheduler_config, optimizer)
    if scheduler_state_to_load and lr_scheduler:
        logger.info("Loading LR scheduler state...")
        try:
            lr_scheduler.load_state_dict(scheduler_state_to_load)
            logger.info("LR Scheduler state loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}", exc_info=True)
            logger.warning("Continuing without loaded scheduler state.")
    elif resume_checkpoint_path:
        logger.info("Resuming run, but no scheduler state found or scheduler not used.")

    if resume_checkpoint_path:
        logger.info(f"Resumed. Final model/optimizer on {device}. Start epoch {start_epoch}. Best score: {best_score:.4f}")
    else:
        logger.info(f"Initialized from scratch. Final model/optimizer on {device}.")

    return model, tokenizer, optimizer, lr_scheduler, start_epoch, best_score



# --- Load Inference Components (Revised for Adapter Subdirs) ---
def load_inference_components(
    experiment_dir: Union[str, Path],
    checkpoint_filename: str = "best.pth", # Can be 'best.pth' or 'last.pth'
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Union[PreTrainedModel, PeftModel], PreTrainedTokenizerBase, Dict]:
    """
    Loads model, tokenizer, and config for inference. Handles standard or LoRA
    checkpoints, loading adapter from 'adapter_best' or 'adapter_last' subdir
    based on checkpoint_filename.
    """
    experiment_dir = Path(experiment_dir)
    checkpoint_path = experiment_dir / checkpoint_filename
    logger.info(f"Loading inference components from checkpoint: {checkpoint_path}")

    state = load_checkpoint_state(checkpoint_path)
    config = state['config']
    is_peft_checkpoint = state.get('is_peft_checkpoint', False)

    logger.info("Loading base model architecture and tokenizer for inference...")
    base_model, tokenizer, _ = _load_model_tokenizer_from_state(state, load_weights=False)

    if is_peft_checkpoint:
        # --- Determine which adapter subdir to load from ---
        if checkpoint_filename == "best.pth":
            adapter_dir_to_load = experiment_dir / "adapter_best"
        elif checkpoint_filename == "last.pth":
            adapter_dir_to_load = experiment_dir / "adapter_last"
        else:
            # Fallback or error if using a different naming scheme
            logger.warning(f"Unknown checkpoint filename '{checkpoint_filename}'. "
                           "Assuming adapter is in main directory or 'adapter_best'. Trying 'adapter_best'.")
            adapter_dir_to_load = experiment_dir / "adapter_best"
            if not adapter_dir_to_load.exists():
                adapter_dir_to_load = experiment_dir # Try main dir as last resort

        logger.info(f"Loading PEFT adapter for inference from: {adapter_dir_to_load}")
        if not adapter_dir_to_load.exists():
            raise FileNotFoundError(f"Required PEFT adapter directory not found for inference: {adapter_dir_to_load}")
        try:
            model = PeftModel.from_pretrained(base_model, adapter_dir_to_load, is_trainable=False)
            logger.info("PEFT adapter loaded for inference.")
        except Exception as e:
            logger.error(f"Failed to load PEFT adapter from {adapter_dir_to_load}: {e}", exc_info=True)
            raise
    else:
        # --- Load Standard Model Weights ---
        logger.info("Loading full model weights for inference...")
        model_state_dict = state.get('model_state_dict')
        if model_state_dict is None: raise ValueError("Standard checkpoint state missing 'model_state_dict'.")
        try:
            if all(k.startswith('module.') for k in model_state_dict.keys()):
                model_state_dict = {k[len('module.'):]: v for k, v in model_state_dict.items()}
            base_model.load_state_dict(model_state_dict, strict=True)
            model = base_model
            logger.info("Full model weights loaded successfully.")
        except Exception as e: logger.error(f"Failed to load full model state dict: {e}", exc_info=True); raise

    model.to(device)
    model.eval()
    logger.info(f"Model moved to {device} and set to eval mode.")

    return model, tokenizer, config
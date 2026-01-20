import logging
from typing import Any, Dict, Optional
import torch.nn as nn
from torch.optim import Optimizer, AdamW # Add other optimizers as needed
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
try: # Import GaLore optimizers if available
    from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
except ImportError:
    GaLoreAdamW = GaLoreAdamW8bit = GaLoreAdafactor = None
from vlm_research_kit.training.schedulers import ExponentialCyclicLR # Add others


logger = logging.getLogger(__name__)


def create_optimizer(optimizer_config: Dict[str, Any], model: nn.Module) -> Optimizer:
    """
    Factory function to create an optimizer based on configuration.

    Args:
        optimizer_config: The 'optimizer' section of the config. Expected keys:
                          'type' (e.g., 'AdamW'), 'lr', optional 'kwargs'.
        model: The model whose parameters the optimizer should manage.

    Returns:
        An instantiated optimizer.

    Raises:
        ValueError: If optimizer type is missing or unsupported.
        TypeError: If config format is invalid.
    """
    if not isinstance(optimizer_config, dict):
        raise TypeError("optimizer_config must be a dictionary.")

    optimizer_type = optimizer_config.get("type")
    if not optimizer_type:
        raise ValueError("Optimizer 'type' must be specified in optimizer_config.")

    lr = optimizer_config.get("lr")
    if lr is None:
        raise ValueError("Optimizer 'lr' must be specified in optimizer_config.")

    kwargs = optimizer_config.get("kwargs", {})
    galore_cfg = optimizer_config.get("galore", {})
    opt_type_lower = optimizer_type.lower()

    logger.info(f"Creating optimizer: type='{optimizer_type}', lr={lr}, kwargs={kwargs}")

    # --- GaLore parameter grouping ---
    if "galore" in opt_type_lower:
        if GaLoreAdamW is None:
            raise ImportError("galore-torch is not installed. Please install it to use GaLore optimizers.")
        
        logger.info(f"Using GaLore optimizer. Config: {galore_cfg}")

        # Find target modules (default: all nn.Linear layers with 'attn' or 'mlp' in their name)
        target_modules = galore_cfg.get("target_modules", ["attn", "mlp"])
        galore_params = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(target in name for target in target_modules):
                continue
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        non_galore_params = [p for p in model.parameters() if id(p) not in id_galore_params]

        param_groups = [
            {'params': non_galore_params},
            {
                'params': galore_params,
                'rank': galore_cfg.get("rank", 128),
                'update_proj_gap': galore_cfg.get("update_proj_gap", 50),
                'scale': galore_cfg.get("scale", 1.0),
                'proj_type': galore_cfg.get("proj_type", "std"),
            }
        ]

        if opt_type_lower == "galoreadamw":
            optimizer = GaLoreAdamW(param_groups, lr=lr, **kwargs)
        elif opt_type_lower == "galoreadamw8bit":
            optimizer = GaLoreAdamW8bit(param_groups, lr=lr, **kwargs)
        elif opt_type_lower == "galoreadafactor":
            optimizer = GaLoreAdafactor(param_groups, lr=lr, **kwargs)
        else:
            raise ValueError(f"Unsupported GaLore optimizer type: '{optimizer_type}'")
    else:
        params = model.parameters()
        if opt_type_lower == "adamw":
            optimizer = AdamW(params, lr=lr, **kwargs)
        # Add other optimizers here
        else:
            raise ValueError(f"Unsupported optimizer type: '{optimizer_type}'")

    logger.info(f"Optimizer '{optimizer_type}' created successfully.")
    return optimizer


def create_scheduler(
    scheduler_config: Dict[str, Any], optimizer: Optimizer
) -> Optional[_LRScheduler]:
    """
    Factory function to create an LR scheduler based on configuration.

    Args:
        scheduler_config: The 'lr_scheduler' section of the config. Expected keys:
                          optional 'type', optional 'kwargs'.
        optimizer: The optimizer instance the scheduler should wrap.

    Returns:
        An instantiated scheduler, or None if no 'type' is specified.

    Raises:
        ValueError: If scheduler type is specified but unsupported.
        TypeError: If config format is invalid.
    """
    if not isinstance(scheduler_config, dict):
        # Allow empty dict or None if no scheduler is intended
        if scheduler_config is None:
            logger.info("No LR scheduler configured.")
            return None
        raise TypeError("scheduler_config must be a dictionary or None.")

    scheduler_type = scheduler_config.get("type")
    if not scheduler_type:
        logger.info("No LR scheduler 'type' specified. No scheduler will be used.")
        return None

    kwargs = scheduler_config.get("kwargs", {})
    logger.info(f"Creating LR scheduler: type='{scheduler_type}', kwargs={kwargs}")

    sched_type_lower = scheduler_type.lower()
    if sched_type_lower == "reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer, **kwargs)
    elif sched_type_lower == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer, **kwargs)
    elif sched_type_lower == "exponentialcycliclr":
        scheduler = ExponentialCyclicLR(optimizer, **kwargs)
    elif sched_type_lower == "cycliclr":
        # Example for CyclicLR, adjust as needed
        scheduler = CyclicLR(optimizer, **kwargs)
    # Add other schedulers here
    # elif sched_type_lower == "steplr":
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler type: '{scheduler_type}'")

    logger.info(f"LR Scheduler '{scheduler_type}' created successfully.")
    return scheduler


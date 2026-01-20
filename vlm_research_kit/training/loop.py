import logging
import time
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from vlm_research_kit.training.model_specific_training_logic import LossComputationFn
from vlm_research_kit.utils.tracking_utils import log_metrics

logger = logging.getLogger(__name__)

def train_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: Optional[_LRScheduler],
    train_dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device],
    device_type: str,
    epoch: int,
    config: Dict[str, Any],
    compute_loss_fn: Optional[LossComputationFn],
    tokenizer: Optional[PreTrainedTokenizerBase], # Required by compute_loss_fn
    wandb_run: Optional[Any] = None, # wandb run object
    local_tracking_filepath: Optional[str] = None, # Local tracking file path
    scaler: Optional[GradScaler] = None, # Pass GradScaler if using mixed precision
) -> float:
    """
    Performs one training epoch.

    Args:
        model: The model to train.
        optimizer: The optimizer.
        lr_scheduler: The learning rate scheduler (optional).
        train_dataloader: The training data loader.
        device: The compute device.
        device_type: The type of device (e.g., 'cuda', 'cpu').
        epoch: The current epoch number.
        config: The main training configuration dictionary. Used to get settings
                like gradient accumulation, logging frequency, task specifics.
        compute_loss_fn: Function to compute loss. Should accept model, batch,
                            device, tokenizer as arguments.
        tokenizer: Tokenizer, needed for text generation tasks.
        wandb_run: Optional wandb run object for logging.
        local_tracking_filepath: Optional path for local tracking.
        scaler: Optional GradScaler for mixed precision training.

    Returns:
        The average training loss for the epoch.

    Assumptions:
        - The model's forward pass returns a dictionary containing a 'loss' key
          (common in Hugging Face models).
        - `train_iterator` yields batches compatible with the model's forward method.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad() # Ensure grads are zero at the start

    # -- Get training settings from config --
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    log_interval = config["training"].get("log_interval_steps", 50)
    scheduler_step_mode = config["lr_scheduler"]["step_mode"]
    if scheduler_step_mode not in ["batch", "epoch"]:
        raise ValueError(f"Invalid scheduler step mode: {scheduler_step_mode}. Must be 'batch' or 'epoch'.")
    max_grad_norm = config["training"].get("max_grad_norm", None)
    if max_grad_norm is not None and max_grad_norm <= 0:
        logger.warning(f"max_grad_norm is set to {max_grad_norm}, disabling clipping.")
        max_grad_norm = None
    
    # Check if we are using mixed precision
    use_amp = scaler is not None

    # Setup progress bar
    batches_per_epoch = len(train_dataloader)
    pbar = tqdm(
        enumerate(train_dataloader),
        total=batches_per_epoch,
        desc="Training",
        mininterval=2, # Set minimum interval for progress bar updates
        ncols=100, # Set width of progress bar
    )

    start_time = time.time()

    for step, batch in pbar:
        # Move batch to device
        assert isinstance(batch, dict), "Batch must be a dictionary."
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Mixed Precision Context
        with autocast(device_type=device_type, enabled=use_amp):
            # --- Forward Pass ---
            loss = compute_loss_fn(model=model, batch=batch, device=device, tokenizer=tokenizer)

            # Normalize loss for gradient accumulation
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps

        # --- Backward Pass ---
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Accumulate loss (use detached item)
        batch_loss = loss.detach().item() * grad_accum_steps # Scale back for logging
        total_loss += batch_loss

        # --- Optimizer Step ---
        if (step + 1) % grad_accum_steps == 0:
            if use_amp:
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            # Zero gradients after optimizer step
            optimizer.zero_grad()

            # Conditionally step scheduler (BATCH MODE ONLY)
            if lr_scheduler is not None and scheduler_step_mode == "batch":
                lr_scheduler.step()

        # --- Logging ---
        pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if (step + 1) % log_interval == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = log_interval / elapsed_time
            current_lr = optimizer.param_groups[0]['lr']
            log_data = {
                "train/batch_loss": batch_loss,
                "train/learning_rate": current_lr,
                "train/steps_per_sec": steps_per_sec,
            }
            # Log to Wandb (global step calculation)
            global_step = epoch * batches_per_epoch + step
            log_metrics(metrics_dict=log_data, step=global_step, wandb_run=wandb_run,
                        log_prefix="Train", log_to_terminal=False,
                        log_to_file=local_tracking_filepath)
            start_time = time.time() # Reset timer for next interval

    # --- End of Epoch ---
    avg_loss = total_loss / batches_per_epoch
    return avg_loss
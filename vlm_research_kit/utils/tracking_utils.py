import json
import os
import logging
import glob
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Try importing wandb, but allow the code to run without it if not used
try:
    import wandb
    # Check if a run is active (wandb.run is None if not initialized)
    WANDB_ACTIVE = wandb.run is not None
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None # Define wandb as None if import fails
    WANDB_ACTIVE = False
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# Wandb Initialization/Finalization (Keep these separate)
# =============================================================================

def find_latest_wandb_run_id(wandb_dir):
    """
    Finds the most recent wandb run ID in the given directory.
    Returns None if not found.
    """
    run_dirs = glob.glob(os.path.join(wandb_dir, "run-*"))
    if not run_dirs:
        return None
    # Sort by modification time, descending
    run_dirs = sorted(run_dirs, key=os.path.getmtime, reverse=True)
    latest_run_dir = run_dirs[0]
    # Extract run_id from directory name: run-YYYYMMDD_HHMMSS-<run_id>
    run_id = latest_run_dir.split("-")[-1]
    return run_id
    
def initialize_wandb(
    experiment_dir: Union[str, os.PathLike],
    config: Dict[str, Any],
    resume_if_possible: bool = True,
    **kwargs # Allow passing extra args to wandb.init
) -> Optional[Any]: # Returns wandb.Run object or None
    """
    Initializes a Weights & Biases run.
    Args:
        experiment_dir: Directory where the wandb run will be stored.
        config: Configuration dictionary to log with wandb.
        resume_if_possible: If True, attempts to resume the latest run if available.
        **kwargs: Additional arguments to pass to wandb.init.
    Returns:
        wandb_run: The initialized wandb run object, or None if wandb is not available.
    """
    experiment_dir = Path(experiment_dir) # Ensure experiment_dir is a Path object
    use_wandb = config.get("tracking", {}).get("wandb", {}).get("enabled", False)
    if not use_wandb:
        logger.info("Weights & Biases tracking is disabled in the configuration.")
        return None

    if not WANDB_AVAILABLE:
        logger.warning("Wandb tracking is enabled, but the 'wandb' library is not installed. Skipping.")
        return None

    try:
        wandb_config = config.get("tracking", {}).get("wandb", {})
        project = wandb_config.get("project")
        if project is None:
            raise ValueError("Wandb project name is required in the configuration.")
        entity = wandb_config.get("entity")
        run_name = wandb_config.get("run_name")
        if run_name is None:
            raise ValueError("Wandb run name is required in the configuration.")
        notes = wandb_config.get("notes")
        tags = wandb_config.get("tags")
        epochwise_metrics = wandb_config.get("epochwise_metrics")
        stepwise_metrics = wandb_config.get("stepwise_metrics")

        # --- Resume logic ---
        run_id = None
        resume = None
        if resume_if_possible:
            run_id = find_latest_wandb_run_id(experiment_dir / "wandb")
            if run_id:
                logger.info(f"Found previous wandb run with id: {run_id}. Will attempt to resume.")
                resume = "allow"

        logger.info(f"Initializing Wandb run: project='{project}', entity='{entity}', name='{run_name}'")
        wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            dir=experiment_dir,
            notes=notes,
            tags=tags,
            id=run_id,
            resume=resume,
            **kwargs
        )

        # --- Set up metrics ---
        if epochwise_metrics is not None:
            logger.info(f"Defining epoch-wise metrics: {epochwise_metrics}")
            wandb_run.define_metric("epoch")
            for metric in epochwise_metrics:
                wandb_run.define_metric(metric, step_metric="epoch")
        if stepwise_metrics is not None:
            logger.info(f"Defining step-wise metrics: {stepwise_metrics}")
            wandb_run.define_metric("step")
            for metric in stepwise_metrics:
                wandb_run.define_metric(metric, step_metric="step")

        global WANDB_ACTIVE
        WANDB_ACTIVE = True
        logger.info(f"Wandb run initialized. Run page: {wandb_run.url}")
        return wandb_run

    except Exception as e:
        logger.error(f"Failed to initialize Wandb: {e}", exc_info=True)
        WANDB_ACTIVE = False
        return None

def finalize_wandb(wandb_run: Optional[Any]) -> None:
    """Finalizes the current Weights & Biases run, if active."""
    global WANDB_ACTIVE
    if wandb_run is not None and WANDB_ACTIVE and WANDB_AVAILABLE:
        try:
            wandb.finish()
            logger.info("Wandb run finished.")
            WANDB_ACTIVE = False
        except Exception as e:
            logger.error(f"Error finishing Wandb run: {e}", exc_info=True)
    elif WANDB_ACTIVE:
        logger.warning("Attempted to finalize wandb, but no valid run object or library was found.")
        WANDB_ACTIVE = False # Ensure flag is reset


# =============================================================================
# Metric Logging Function
# =============================================================================

def log_metrics(
    metrics_dict: Dict[str, Any],
    step: int,
    step_metric: str = "step",
    wandb_run: Optional[Any] = None,
    log_prefix: str = "",
    log_to_terminal: bool = True,
    log_to_file: Optional[str] = None,
) -> None:
    """
    Logs metrics locally (optionally to terminal and/or a jsonl file) and to Weights & Biases (if active).
    Args:
        metrics_dict: Dictionary where keys are metric names (e.g., 'train/loss')
                      and values are the corresponding scores or data. Values
                      should ideally be numbers or simple types wandb can handle.
        step: The current step (e.g., epoch number, global batch step) to associate
              with the metrics in wandb.
        step_metric: The name of the step metric (default is "step").
        wandb_run: The active wandb run object (returned by initialize_wandb).
                   If None or wandb is disabled/unavailable, logs only locally.
        log_prefix: Optional string to prepend to local log messages for context.
        log_to_terminal: If True, logs to terminal using Python's logging.
        log_to_file: If provided, logs to a jsonl file at this path.
    """

    # --- 1. Local Terminal Logging ---
    if log_to_terminal:
        log_header = (
            f"{log_prefix} {step_metric.capitalize()} {step} Metrics:"\
                if log_prefix else f"{step_metric.capitalize()} {step} Metrics:"
        )
        logger.info(log_header)
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                log_message = f"  {key}: {value:.4f}"
            elif isinstance(value, dict):
                log_message = f"  {key}: (dict - see wandb for full details)"
            elif isinstance(value, torch.Tensor):
                try:
                    float_val = value.item()
                    log_message = f"  {key}: {float_val:.4f} (from tensor)"
                except Exception:
                    log_message = f"  {key}: (Tensor - could not get scalar value)"
            else:
                log_message = f"  {key}: {value}"
            logger.info(log_message)

    # --- 2. Local File Logging (jsonl) ---
    if log_to_file is not None:
        # Prepare the log entry
        log_entry = {
            "step_metric": step_metric, 
            "step": step,
            "metrics": metrics_dict,
        }
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_to_file), exist_ok=True)
        # Append as a new line
        with open(log_to_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # --- 3. Wandb Logging ---
    if WANDB_AVAILABLE and wandb_run is not None and hasattr(wandb_run, "log"):
        try:
            wandb_run.log({step_metric: step, **metrics_dict})
        except Exception as e:
            logger.error(
                f"Failed to log metrics to wandb at step {step}: {e}", exc_info=True
            )
    elif wandb_run is not None:
        if not WANDB_AVAILABLE:
            logger.warning(
                f"Wandb run object provided for step {step}, but wandb library not installed. Skipping wandb log."
            )
        elif not hasattr(wandb_run, "log"):
            logger.warning(
                f"Object provided as wandb_run for step {step} lacks .log method. Skipping wandb log."
            )

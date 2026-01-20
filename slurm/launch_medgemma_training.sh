#!/bin/bash
#SBATCH --job-name=medgemma-train # Default/fallback name
#SBATCH -t 1-00:00:00
#SBATCH -p batch
#SBATCH -q batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH -o ./logs/default-%j.out # Default/fallback output. Feel free to change this to a more specific path.

# --- Script Argument Validation ---
if [ -z "$1" ]; then
    echo "ERROR: This script requires a config path."
    exit 1
fi
CONFIG_PATH=$1
CONDA_ENV=${2:-"py313"} # <-- NEW: Read conda env from second argument
echo "Using training config: $CONFIG_PATH"

# --- Environment Setup ---
echo "Loading conda..."
module load conda
echo "Activating conda environment: $CONDA_ENV" # <-- Modified log message
conda activate "$CONDA_ENV" # <-- Use the variable

# --- GPU Setup and Diagnostics ---
# Use the SLURM_GPUS_ON_NODE variable, which is reliably set by SLURM.
# Default to 1 if it's not set for any reason.
export N_GPUS=${SLURM_GPUS_ON_NODE:-1}

echo "----------------------------------------------------"
echo "SLURM JOB DIAGNOSTICS:"
echo "SLURM Job ID:     $SLURM_JOB_ID"
echo "SLURM Node List:  $SLURM_JOB_NODELIST"
echo "SLURM GPUs on Node: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "N_GPUS variable set to: $N_GPUS"
echo "----------------------------------------------------"

# --- PRE-FLIGHT CHECKS (RECOMMENDED ADDITION) ---
echo "Running nvidia-smi to see available GPUs:"
if ! nvidia-smi; then
    echo "FATAL: nvidia-smi command failed. There is a critical GPU driver issue on node $(hostname)."
    exit 1
fi
echo "----------------------------------------------------"
echo "Checking GPU accessibility with PyTorch:"
python -c "
import torch
import sys
import os
if not torch.cuda.is_available():
    print('FATAL: PyTorch reports CUDA is not available.')
    sys.exit(1)
count = torch.cuda.device_count()
print(f'PyTorch sees {count} CUDA devices.')
n_gpus_requested = int(os.environ.get('N_GPUS', 1))
if count < n_gpus_requested:
    print(f'FATAL: SLURM requested {n_gpus_requested} GPUs, but PyTorch only sees {count}.')
    sys.exit(1)
for i in range(count):
    try:
        print(f'  - Device {i}: {torch.cuda.get_device_name(i)}')
    except Exception as e:
        print(f'FATAL: Could not get name for device {i}. Error: {e}')
        sys.exit(1)
"
# Exit if the python check fails
if [ $? -ne 0 ]; then
    echo "FATAL: PyTorch GPU check failed. Aborting job."
    exit 1
fi
echo "----------------------------------------------------"


# --- Conditional Launch Logic ---
# This is the key fix: use `python` for single-GPU, `torchrun` for multi-GPU.

if [ "$N_GPUS" -le 1 ]; then
    echo "Single GPU job detected. Launching with plain 'python'."
    python ../scripts/train_medgemma.py \
        --train_config_path "$CONFIG_PATH"
else
    echo "Multi-GPU job detected ($N_GPUS GPUs). Launching with 'torchrun'."
    # Set master address and port for torchrun
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=29500
    # NCCL_DEBUG=INFO is useful for any distributed job failure
    export NCCL_DEBUG=INFO

    torchrun --nproc_per_node=$N_GPUS --nnodes="$SLURM_NNODES" --node_rank="$SLURM_NODEID" \
        --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        ./scripts/train_medgemma.py \
        --train_config_path "$CONFIG_PATH"
fi

echo "Job finished."
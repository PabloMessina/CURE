#!/bin/bash
#
# SLURM submission script for running VLM Standard Report Generation evaluation.
#
# ------------------------------------------------------------------------------
# Usage:
#   sbatch launch...sh model_name dataset [optional_python_args...]
#
# Example Usage for MAIRA-2 on MIMIC-CXR:
#   CONDA_ENV=maira sbatch ./slurm/launch_report_generation_evaluation.sh maira-2 mimic-cxr
#
# Example Usage for CXRMATE-RRG24 on MIMIC-CXR:
#   CONDA_ENV=maira sbatch ./slurm/launch_report_generation_evaluation.sh cxrmate-rrg24 mimic-cxr
#
# Example Usage for a fine-tuned MedGemma on MIMIC-CXR:
#   CONDA_ENV=vlm sbatch ./slurm/launch_report_generation_evaluation.sh medgemma mimic-cxr \
#     --medgemma_adapter_path /path/to/your/checkpoint-1000
#
# Note: This script is configured for standard report generation on datasets
#       like 'mimic-cxr'.
# ------------------------------------------------------------------------------

# --- SLURM Configuration ---
#SBATCH --job-name=vlm-std-rg-eval
#SBATCH --output=./logs/vlm-std-rg-eval-%j.out # Feel free to change this to a more specific path.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=0-12:00:00
#SBATCH --partition=batch
#SBATCH -q batch

# --- Bash Script Best Practices ---
set -euo pipefail

# --- Argument Validation ---
if [ "$#" -lt 2 ]; then
    echo "ERROR: Invalid arguments."
    echo "Usage: sbatch $0 model_name dataset [optional_python_args...]"
    echo "Note: Supported datasets include 'mimic-cxr'."
    exit 1
fi

MODEL_NAME=$1
DATASET_NAME=$2
shift 2 # Remove the model_name and dataset, leaving only optional python args

# --- Dynamic Job Name ---
# Try to find an adapter path in the args to create a more descriptive job name
JOB_TAG=""
PYTHON_ARGS=("$@")
for i in "${!PYTHON_ARGS[@]}"; do
    if [[ "${PYTHON_ARGS[$i]}" == "--medgemma_adapter_path" ]]; then
        ADAPTER_PATH="${PYTHON_ARGS[$i+1]}"
        JOB_TAG="-$(basename "$ADAPTER_PATH")" # e.g., "-checkpoint-1000"
        break
    fi
done
JOB_NAME="eval-std-rg-${MODEL_NAME}-on-${DATASET_NAME}${JOB_TAG:-"-base"}" # Fallback to "-base"
scontrol update job "$SLURM_JOB_ID" JobName="$JOB_NAME"

# --- Dynamic Log File Redirection ---
# Use a relative directory within the repo or a generic cluster path
LOG_DIR="./logs" # Feel free to change this to a more specific path.
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}-${SLURM_JOB_ID}.out"
echo "----------------------------------------------------"
echo "SLURM log redirection is active."
echo "The original SLURM log file will be mostly empty."
echo "Find the detailed output in: ${LOG_FILE}"
echo "----------------------------------------------------"
# Redirect stdout and stderr to the new log file
exec > "$LOG_FILE" 2>&1

# --- Environment Setup ---
echo "Loading conda..."
module load conda
# Use the provided CONDA_ENV or default to 'vlm'
CONDA_ENV=${CONDA_ENV:-"vlm"}
echo "Activating conda environment: $CONDA_ENV"
# Temporarily disable 'nounset' to allow for conda's scripts
# which may have unbound variables.
set +u
conda activate "$CONDA_ENV"
# Re-enable 'nounset' for the rest of our script.
set -u

# --- SLURM & GPU Diagnostics ---
echo "----------------------------------------------------"
echo "SLURM JOB DIAGNOSTICS:"
echo "Job ID:               $SLURM_JOB_ID"
echo "Job Name:             $JOB_NAME"
echo "Assigned Node:        $SLURM_JOB_NODELIST"
echo "GPUs on Node:         $SLURM_GPUS_ON_NODE"
echo "CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
echo "Model Name:           $MODEL_NAME"
echo "Dataset Name:         $DATASET_NAME"
echo "Optional Arguments:   $@"
echo "----------------------------------------------------"

# --- Pre-flight GPU Checks ---
echo "Running nvidia-smi to see available GPUs:"
if ! nvidia-smi; then
    echo "FATAL: nvidia-smi command failed. There is a critical GPU driver issue on node $(hostname)."
    exit 1
fi
echo "----------------------------------------------------"
echo "Checking GPU accessibility with PyTorch:"
python -c "
import torch, sys
if not torch.cuda.is_available():
    print('FATAL: PyTorch reports CUDA is not available.')
    sys.exit(1)
count = torch.cuda.device_count()
print(f'PyTorch sees {count} CUDA devices.')
if count < 1:
    print(f'FATAL: At least 1 GPU is required, but PyTorch sees {count}.')
    sys.exit(1)
print(f'  - Device 0: {torch.cuda.get_device_name(0)}')
"
if [ $? -ne 0 ]; then
    echo "FATAL: PyTorch GPU check failed. Aborting job."
    exit 1
fi
echo "----------------------------------------------------"

# --- Execute the Evaluation Script ---
echo "Starting Python report generation evaluation script..."
python ./scripts/eval_report_generation.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET_NAME" \
    "$@" # Pass all remaining arguments directly

echo "Job finished successfully."
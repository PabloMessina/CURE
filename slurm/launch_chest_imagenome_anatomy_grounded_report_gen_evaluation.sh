#!/bin/bash
#
# SLURM submission script for running VLM evaluation.
#
# ------------------------------------------------------------------------------
# Usage:
#   sbatch launch...sh model_name [optional_python_args...]
#
# Example Usage for MAIRA-2:
#   This will run the evaluation for 'maira-2' using the 'maira' conda env.
#
#   CONDA_ENV=maira sbatch ./slurm/launch_chest_imagenome_anatomy_grounded_report_gen_evaluation.sh maira-2
#
# Example Usage for MedGemma:
#   This will run the evaluation for 'medgemma' using the 'vlm' conda env.
#
#   CONDA_ENV=vlm sbatch ./slurm/launch_chest_imagenome_anatomy_grounded_report_gen_evaluation.sh medgemma \
#   --medgemma_adapter_path /path/to/your/checkpoint-1000
#
# Example Usage for MedGemma with image transforms:
#   This will run the evaluation for 'medgemma' using the 'vlm' conda env with image transforms.
#
#   CONDA_ENV=vlm sbatch ./slurm/launch_chest_imagenome_anatomy_grounded_report_gen_evaluation.sh medgemma \
#   --medgemma_adapter_path /path/to/your/checkpoint-1000 \
#   --image_transforms_kwargs '{"use_model_specific_transforms": true, "model_name": "pil_image_only", "image_size": [448, 448], "bbox_format": "cxcywh", "is_train": false}'
#
# ------------------------------------------------------------------------------

# --- SLURM Configuration ---
# Hardcode a reasonable default. This is now much safer.
#SBATCH --job-name=vlm-eval
#SBATCH --output=./logs/vlm-eval-%j.out # Feel free to change this to a more specific path.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=0-5:00:00
#SBATCH --partition=batch
#SBATCH -q batch

# --- Bash Script Best Practices ---
set -euo pipefail

# --- Argument Validation ---
if [ "$#" -lt 1 ]; then
    echo "ERROR: Invalid arguments."
    echo "Usage: sbatch $0 model_name [optional_python_args...]"
    exit 1
fi

MODEL_NAME=$1
shift 1 # Remove the model_name, leaving only the optional python args

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
JOB_NAME="eval-${MODEL_NAME}${JOB_TAG:-"-base"}" # Fallback to "-base" if no adapter
scontrol update job "$SLURM_JOB_ID" JobName="$JOB_NAME"

# --- ** NEW: DYNAMIC LOG FILE REDIRECTION ** ---
# The #SBATCH --output directive is processed before the job name is updated.
# To get a descriptively named log file, we redirect this script's output
# to a new file based on the dynamic JOB_NAME.
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
# Temporarily relax strict mode for conda activation
set +u
# Use the provided CONDA_ENV or default to 'py313'
CONDA_ENV=${CONDA_ENV:-"py313"}
echo "Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"
# Re-enable strict mode for the rest of our script.
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
echo "Starting Python evaluation script..."
python ./scripts/eval_chest_imagenome_anatomy_grounded_report_generation.py \
    --model_name "$MODEL_NAME" \
    "$@" # Pass all remaining arguments directly

echo "Job finished successfully."
# Wrapper script to submit SLURM jobs for MedGemma training.
#
# Usage:
#   ./slurm/submit_medgemma_training.sh /path/to/config.yaml [num_gpus] [memory_gb] [time_limit] [conda_env] [target_node]
#
# Arguments:
#   config_path: (Required) Path to the training YAML configuration file.
#   num_gpus:    (Optional) Number of GPUs to request. Defaults to 1.
#   memory_gb:   (Optional) Amount of RAM in GB to request. Defaults to 64.
#   time_limit:  (Optional) Max job runtime in SLURM format (e.g., D-HH:MM:SS).
#                Defaults to "1-00:00:00" (1 day).
#   conda_env:   (Optional) Name of the conda environment to use. Defaults to "py313".
#   target_node: (Optional) Specific node hostname to target. If omitted, SLURM
#                will schedule the job on any available eligible node.
#
# Examples:
#   # 1. Run with all defaults (1 GPU, 64GB RAM, 1 day limit)
#   ./slurm/submit_medgemma_training.sh configs/my_experiment.yaml
#
#   # 2. Run on 2 GPUs, with default memory and time
#   ./slurm/submit_medgemma_training.sh configs/my_experiment.yaml 2
#
#   # 3. Run on 2 GPUs with 64GB of RAM, with default time
#   ./slurm/submit_medgemma_training.sh configs/my_experiment.yaml 2 64
#
#   # 4. Run on 3 GPUs, 90GB RAM, and a 2.5 day time limit
#   ./slurm/submit_medgemma_training.sh configs/my_experiment.yaml 3 90 2-12:00:00
#
#   # 5. Run a short test on 1 GPU with a 2-hour time limit
#   ./slurm/submit_medgemma_training.sh configs/my_experiment.yaml 1 32 0-02:00:00
#
#   # 6. Use a different conda environment
#   ./slurm/submit_medgemma_training.sh configs/my_experiment.yaml 1 32 1-00:00:00 myenv
#
#   # 7. Target a specific node
#   ./slurm/submit_medgemma_training.sh configs/my_experiment.yaml 1 64 1-00:00:00 vlm node01

# --- Validate Input ---
if [ -z "$1" ]; then
    echo "ERROR: No training config path provided."
    echo "Usage: $0 /path/to/config.yaml [num_gpus] [memory_gb] [time_limit] [conda_env] [target_node]"
    exit 1
fi
CONFIG_PATH=$1
NUM_GPUS=${2:-1}
MEMORY_GB=${3:-64}
TIME_LIMIT=${4:-"1-00:00:00"}
CONDA_ENV=${5:-"py313"}
TARGET_NODE=${6:-""}  # <-- NEW: optional node hostname to target

echo "Requesting ${NUM_GPUS} GPUs, ${MEMORY_GB}GB RAM, for a max of ${TIME_LIMIT}."
echo "Using Conda environment: ${CONDA_ENV}"
if [ -n "$TARGET_NODE" ]; then
    echo "Targeting specific node: ${TARGET_NODE}"
fi

# --- Generate Dynamic Names ---
FILENAME=$(basename "$CONFIG_PATH")
JOB_NAME="${FILENAME%.*}"
OUTPUT_FILE="./logs/${JOB_NAME}-%j.out" # Feel free to change this to a more specific path.

# --- Submit the Job ---
# Using the --gres flag for GPU allocation, which is a common and robust method.

# sbatch \
#   --job-name="$JOB_NAME" \
#   --output="$OUTPUT_FILE" \
#   --gres=gpu:"$NUM_GPUS" \
#   --mem="${MEMORY_GB}G" \
#   --time="$TIME_LIMIT" \
#   ./slurm/launch_medgemma_training.sh "$CONFIG_PATH"

# --- Submit the Job ---
SBATCH_COMMAND=(
  sbatch
  --job-name="$JOB_NAME"
  --output="$OUTPUT_FILE"
  --gres=gpu:"$NUM_GPUS"
  --mem="${MEMORY_GB}G"
  --time="$TIME_LIMIT"
)

# If node argument provided, use it; otherwise, run on any available node
if [ -n "$TARGET_NODE" ]; then
  SBATCH_COMMAND+=(--nodelist="$TARGET_NODE")
fi

# Pass arguments through to the launch script
"${SBATCH_COMMAND[@]}" ./slurm/launch_medgemma_training.sh "$CONFIG_PATH" "$CONDA_ENV"

echo "==> Submitted job '$JOB_NAME' with ${NUM_GPUS} GPUs."
echo "==> Output will be saved to '$OUTPUT_FILE'"
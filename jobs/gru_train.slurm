#!/bin/bash
#SBATCH -J gru_sweep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --array=1-12     # 12 combinations (2×3×2), run 3 at a time
#SBATCH -o /user_data/cicid/Multifirefly-Project/multiff_analysis/logs/gru_sweep-%A_%a.out
#SBATCH -e /user_data/cicid/Multifirefly-Project/multiff_analysis/logs/gru_sweep-%A_%a.err

set -euo pipefail
cd /user_data/cicid/Multifirefly-Project

# ---------------------
# Activate conda environment
# ---------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiff_clean

# ---------------------
# Define hyperparameter lists
# ---------------------
declare -a LRS=(1e-3 3e-4)
declare -a NUM_OBS_FF_LIST=(6 7 8)
declare -a MAX_IN_MEMORY_TIME_LIST=(2 1)

# total combinations = 2 × 3 × 2 = 12
IDX=$((SLURM_ARRAY_TASK_ID - 1))

LR_IDX=$((IDX / (${#NUM_OBS_FF_LIST[@]} * ${#MAX_IN_MEMORY_TIME_LIST[@]})))
OBS_IDX=$(((IDX / ${#MAX_IN_MEMORY_TIME_LIST[@]}) % ${#NUM_OBS_FF_LIST[@]}))
MEM_IDX=$((IDX % ${#MAX_IN_MEMORY_TIME_LIST[@]}))

LR=${LRS[$LR_IDX]}
NUM_OBS_FF=${NUM_OBS_FF_LIST[$OBS_IDX]}
MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME_LIST[$MEM_IDX]}

# ---------------------
# Create a unique folder name
# ---------------------
BASE_DIR="multiff_analysis/RL_models/GRU_stored_models/all_agents"
RUN_NAME="lr${LR}_numobs${NUM_OBS_FF}_mem${MAX_IN_MEMORY_TIME}_job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OVERALL_FOLDER="${BASE_DIR}/${RUN_NAME}"

mkdir -p "$OVERALL_FOLDER"

# ---------------------
# Run training
# ---------------------
echo "[jobinfo] Starting GRU training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}"

srun python -u multiff_analysis/multiff_code/notebooks/reinforcement_learning/GRU_training.py \
  --overall-folder "$OVERALL_FOLDER" \
  --max-in-memory-time "$MAX_IN_MEMORY_TIME" \
  --lr "$LR" \
  --num-obs-ff "$NUM_OBS_FF"

echo "[jobinfo] Finished GRU training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}"

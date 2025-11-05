#!/bin/bash
#SBATCH -J gru_sweep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --array=1-8 
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
declare -a LRS=(3e-4)
declare -a NUM_OBS_FF_LIST=(1 2 3 4)
declare -a MAX_IN_MEMORY_TIME_LIST=(2)
declare -a IDENTITY_SLOT_STRATEGY_LIST=(rank_keep drop_fill)

# total combinations = ${#LRS[@]} × ${#NUM_OBS_FF_LIST[@]} × ${#MAX_IN_MEMORY_TIME_LIST[@]} × ${#IDENTITY_SLOT_STRATEGY_LIST[@]}
IDX=$((SLURM_ARRAY_TASK_ID - 1))

LR_IDX=$((IDX / (${#NUM_OBS_FF_LIST[@]} * ${#MAX_IN_MEMORY_TIME_LIST[@]} * ${#IDENTITY_SLOT_STRATEGY_LIST[@]})))
OBS_IDX=$(((IDX / (${#MAX_IN_MEMORY_TIME_LIST[@]} * ${#IDENTITY_SLOT_STRATEGY_LIST[@]})) % ${#NUM_OBS_FF_LIST[@]}))
MEM_IDX=$(((IDX / ${#IDENTITY_SLOT_STRATEGY_LIST[@]}) % ${#MAX_IN_MEMORY_TIME_LIST[@]}))
STRAT_IDX=$((IDX % ${#IDENTITY_SLOT_STRATEGY_LIST[@]}))

LR=${LRS[$LR_IDX]}
NUM_OBS_FF=${NUM_OBS_FF_LIST[$OBS_IDX]}
MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME_LIST[$MEM_IDX]}
IDENTITY_SLOT_STRATEGY=${IDENTITY_SLOT_STRATEGY_LIST[$STRAT_IDX]}

# ---------------------
# Create a unique folder name
# ---------------------
BASE_DIR="multiff_analysis/RL_models/GRU_stored_models/all_agents"
# Sanitize decimal points in memory time for folder names (e.g., 1.5 -> 1p5)
SAFE_MAX_IN_MEMORY_TIME="${MAX_IN_MEMORY_TIME//./p}"
RUN_NAME="lr${LR}_numobs${NUM_OBS_FF}_mem${SAFE_MAX_IN_MEMORY_TIME}_strat${IDENTITY_SLOT_STRATEGY}_job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OVERALL_FOLDER="${BASE_DIR}/${RUN_NAME}"

mkdir -p "$OVERALL_FOLDER"

# ---------------------
# Run training
# ---------------------
echo "[jobinfo] Starting GRU training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}, STRAT=${IDENTITY_SLOT_STRATEGY}"

srun python -u multiff_analysis/scripts/gru_script.py \
  --overall-folder "$OVERALL_FOLDER" \
  --max-in-memory-time "$MAX_IN_MEMORY_TIME" \
  --lr "$LR" \
  --num-obs-ff "$NUM_OBS_FF" \
  --identity-slot-strategy "$IDENTITY_SLOT_STRATEGY"

echo "[jobinfo] Finished GRU training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}"
echo "[jobinfo] Finished GRU training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}, STRAT=${IDENTITY_SLOT_STRATEGY}"

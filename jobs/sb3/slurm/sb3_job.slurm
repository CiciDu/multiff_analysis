#!/bin/bash
#SBATCH -J sb3_sweep
#SBATCH --partition=gpu,xaqq
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH -t 24:00:00
#SBATCH --array=1-40%5
#SBATCH -o /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_stdout/sb3_sweep-%A_%a.out
#SBATCH -e /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_stdout/sb3_sweep-%A_%a.err
#SBATCH --exclude=mind-1-29

set -euo pipefail
cd /user_data/cicid/Multifirefly-Project

# activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiff_clean


# ---------------------
# Define hyperparameter lists
# ---------------------
declare -a LRS=(3e-4)
declare -a NUM_OBS_FF_LIST=(1 2 3 4 5)
declare -a MAX_IN_MEMORY_TIME_LIST=(2 3)
declare -a IDENTITY_SLOT_STRATEGY_LIST=(rank_keep_visible_only rank_keep_visible_and_memory drop_fill_visible_only drop_fill_visible_and_memory)
N_ENVS=1

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
BASE_DIR="multiff_analysis/RL_models/sb3_stored_models/all_agents"
SAFE_MAX_IN_MEMORY_TIME="${MAX_IN_MEMORY_TIME//./p}"
RUN_NAME="ff${NUM_OBS_FF}_mem${SAFE_MAX_IN_MEMORY_TIME}_${IDENTITY_SLOT_STRATEGY}"
OVERALL_FOLDER="${BASE_DIR}/${RUN_NAME}"

mkdir -p "$OVERALL_FOLDER"

# Ensure logs directory exists and define error log path for tee
LOG_DIR="/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_stdout"
ERR_FILE="$LOG_DIR/sb3_sweep-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mkdir -p "$LOG_DIR"

# ---------------------
# Run training
# ---------------------
echo "[jobinfo] Starting sb3 training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}, STRAT=${IDENTITY_SLOT_STRATEGY}, N_ENVS=${N_ENVS}"
srun python -u multiff_analysis/jobs/sb3/scripts/sb3_script.py \
  --overall-folder "$OVERALL_FOLDER" \
  --max-in-memory-time "$MAX_IN_MEMORY_TIME" \
  --lr "$LR" \
  --num-obs-ff "$NUM_OBS_FF" \
  --n-envs "$N_ENVS" \
  --no-curriculum-training \
  --identity-slot-strategy "$IDENTITY_SLOT_STRATEGY" 2> >(tee -a "$ERR_FILE" >&2)
  

echo "[jobinfo] Finished sb3 training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}, STRAT=${IDENTITY_SLOT_STRATEGY}, N_ENVS=${N_ENVS}"


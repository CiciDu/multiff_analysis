#!/bin/bash
#SBATCH -J sb3_sweep
#SBATCH --partition=gpu,xaqq
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH -t 24:00:00
#SBATCH --array=0-23
#SBATCH -o /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/sb3/sb3-%A_%a.out
#SBATCH -e /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/sb3/sb3-%A_%a.err

set -euo pipefail
cd /user_data/cicid/Multifirefly-Project

# activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiff_clean

RANDOM_SEED=12345
N_ENVS=1

# ---------------------
# Define hyperparameter lists
# ---------------------
declare -a LRS=(3e-4)
declare -a NUM_OBS_FF_LIST=(1 2 3 4 5 6 7)
declare -a MAX_IN_MEMORY_TIME_LIST=(2 3)
declare -a IDENTITY_SLOT_STRATEGY_LIST=(rank_keep_visible_only rank_keep_visible_and_memory drop_fill_visible_only drop_fill_visible_and_memory)
declare -a BASE_SEEDS=(0 1 2 3)
NUM_REPEATS=5

N_LR=${#LRS[@]}
N_OBS=${#NUM_OBS_FF_LIST[@]}
N_MEM=${#MAX_IN_MEMORY_TIME_LIST[@]}
N_STRAT=${#IDENTITY_SLOT_STRATEGY_LIST[@]}
NUM_LEVELS=$((N_LR * N_OBS * N_MEM * N_STRAT))
NUM_BASE_SEEDS=${#BASE_SEEDS[@]}
COMBOS_PER_LEVEL=$((NUM_BASE_SEEDS * NUM_REPEATS))
TOTAL_COMB=$((NUM_LEVELS * COMBOS_PER_LEVEL))

echo "[info] NUM_LEVELS=${NUM_LEVELS} COMBOS_PER_LEVEL=${COMBOS_PER_LEVEL} TOTAL_COMB=${TOTAL_COMB}"

# Global flat index: submit_sb3.sh may set SLURM_ARRAY_OFFSET when splitting the sweep
# into multiple array jobs (cluster MaxArraySize is typically 1001).
ARRAY_OFFSET=${SLURM_ARRAY_OFFSET:-0}
TASK_ID=$((ARRAY_OFFSET + SLURM_ARRAY_TASK_ID))
echo "[info] ARRAY_OFFSET=${ARRAY_OFFSET} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} TASK_ID=${TASK_ID}"

if (( TASK_ID >= TOTAL_COMB )); then
  echo "[warning] TASK_ID exceeds TOTAL_COMB — exiting safely"
  exit 0
fi

# ---------------------
# Deterministic shuffle (same pattern as sb3_noise_job.slurm)
# ---------------------
IDX=$(python - <<EOF
import numpy as np
rng = np.random.default_rng(${RANDOM_SEED})
perm = rng.permutation(${TOTAL_COMB})
print(int(perm[${TASK_ID}]))
EOF
)

# ---------------------
# Map flat index → hyperparams + seed/repeat
# ---------------------
_i=$IDX

LEVEL_IDX=$((_i / COMBOS_PER_LEVEL))
REST=$((_i % COMBOS_PER_LEVEL))
BASE_SEED_IDX=$((REST / NUM_REPEATS))
REPEAT_IDX=$((REST % NUM_REPEATS))

LR_IDX=$((LEVEL_IDX / (N_OBS * N_MEM * N_STRAT)))
OBS_IDX=$(((LEVEL_IDX / (N_MEM * N_STRAT)) % N_OBS))
MEM_IDX=$(((LEVEL_IDX / N_STRAT) % N_MEM))
STRAT_IDX=$((LEVEL_IDX % N_STRAT))

LR=${LRS[$LR_IDX]}
NUM_OBS_FF=${NUM_OBS_FF_LIST[$OBS_IDX]}
MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME_LIST[$MEM_IDX]}
IDENTITY_SLOT_STRATEGY=${IDENTITY_SLOT_STRATEGY_LIST[$STRAT_IDX]}
BASE_SEED=${BASE_SEEDS[$BASE_SEED_IDX]}

# ---------------------
# Create a unique folder name
# ---------------------
BASE_DIR="RL_models/sb3_stored_models/all_agents/agents_without_noise"
SAFE_MAX_IN_MEMORY_TIME="${MAX_IN_MEMORY_TIME//./p}"
RUN_NAME="ff${NUM_OBS_FF}_mem${SAFE_MAX_IN_MEMORY_TIME}_${IDENTITY_SLOT_STRATEGY}"
AGENT_RUN_NAME="seed${BASE_SEED}_rep${REPEAT_IDX}"
OVERALL_FOLDER="${BASE_DIR}/${RUN_NAME}/${AGENT_RUN_NAME}"

mkdir -p "$OVERALL_FOLDER"

# Ensure logs directory exists and define error log path for tee
LOG_DIR="/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_stdout"
ERR_FILE="$LOG_DIR/sb3-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mkdir -p "$LOG_DIR"

# ---------------------
# Run training
# ---------------------
echo "[jobinfo] Starting sb3 training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}, STRAT=${IDENTITY_SLOT_STRATEGY}, base_seed=${BASE_SEED} rep=${REPEAT_IDX}, N_ENVS=${N_ENVS}"
srun python -u multiff_analysis/jobs/sb3/scripts/sb3_script.py \
  --overall-folder "$OVERALL_FOLDER" \
  --base-seed "$BASE_SEED" \
  --max-in-memory-time "$MAX_IN_MEMORY_TIME" \
  --lr "$LR" \
  --num-obs-ff "$NUM_OBS_FF" \
  --n-envs "$N_ENVS" \
  --identity-slot-strategy "$IDENTITY_SLOT_STRATEGY" \
  2> >(tee -a "$ERR_FILE" >&2)
# --no-curriculum-training


echo "[jobinfo] Finished sb3 training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}, STRAT=${IDENTITY_SLOT_STRATEGY}, base_seed=${BASE_SEED} rep=${REPEAT_IDX}, N_ENVS=${N_ENVS}"


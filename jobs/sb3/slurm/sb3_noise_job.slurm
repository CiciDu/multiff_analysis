#!/bin/bash
#SBATCH -J sb3_noise
#SBATCH --partition=gpu,xaqq
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH -t 24:00:00
# Submit via submit_sb3_noise.sh for full sweep (chunks ≤1000 tasks). Direct sbatch: tiny test array only.
#SBATCH --array=0-0%10
#SBATCH -o /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/sb3_noise/sb3_noise-%A_%a.out
#SBATCH -e /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/sb3_noise/sb3_noise-%A_%a.err


set -euo pipefail
cd /user_data/cicid/Multifirefly-Project

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiff_clean

RANDOM_SEED=12345
N_ENVS=1

# ---------------------
# Hyperparameter lists
# ---------------------
# declare -a MEM_R_LIST=(0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)

# flip the below next time
declare -a MEM_R_LIST=(0.0 3.5 4.0)
declare -a MEM_TH_LIST=(0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
declare -a BASE_SEEDS=(0 1 2 3 4 5 6 7)
NUM_REPEATS=5

# Discrete levels correspond to the (mem_r, mem_th) cartesian product.
LEVELS=($(seq 0 $(( ${#MEM_R_LIST[@]} * ${#MEM_TH_LIST[@]} - 1 ))))
NUM_LEVELS=${#LEVELS[@]}
NUM_BASE_SEEDS=${#BASE_SEEDS[@]}

# ---------------------
# Fixed hyperparameters
# ---------------------
LR=3e-4
NUM_OBS_FF=6
MAX_IN_MEMORY_TIME=3
IDENTITY_SLOT_STRATEGY="drop_fill_visible_only"
PERC_R=0.0
PERC_TH=0.0
V_NOISE_STD=0.0
W_NOISE_STD=0.0

# ---------------------
# Compute total combinations
# ---------------------
N_MEM_R=${#MEM_R_LIST[@]}
N_MEM_TH=${#MEM_TH_LIST[@]}
NUM_BASE_SEEDS=${#BASE_SEEDS[@]}

# Treat the (mem_r, mem_th) cartesian product as discrete "levels".
NUM_LEVELS=$((N_MEM_R * N_MEM_TH))
COMBOS_PER_LEVEL=$((NUM_BASE_SEEDS * NUM_REPEATS))
TOTAL_COMB=$((NUM_LEVELS * COMBOS_PER_LEVEL))

echo "[info] TOTAL_COMB=${TOTAL_COMB}"

# Global flat index when submit_sb3_noise.sh splits the array (SLURM_ARRAY_OFFSET per chunk).
ARRAY_OFFSET=${SLURM_ARRAY_OFFSET:-0}
TASK_ID=$((ARRAY_OFFSET + SLURM_ARRAY_TASK_ID))
echo "[info] ARRAY_OFFSET=${ARRAY_OFFSET} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} TASK_ID=${TASK_ID}"

if (( TASK_ID >= TOTAL_COMB )); then
  echo "[warning] TASK_ID exceeds TOTAL_COMB — exiting safely"
  exit 0
fi

# ---------------------
# Deterministic shuffle
# ---------------------
IDX=$(python - <<EOF
import numpy as np
rng = np.random.default_rng(${RANDOM_SEED})
perm = rng.permutation(${TOTAL_COMB})
print(int(perm[${TASK_ID}]))
EOF
)

# ---------------------
# Map flat index
# ---------------------
_i=$IDX

LEVEL_IDX=$((_i / COMBOS_PER_LEVEL))
REST=$((_i % COMBOS_PER_LEVEL))
BASE_SEED_IDX=$((REST / NUM_REPEATS))
REPEAT_IDX=$((REST % NUM_REPEATS))

MEMR_TH_IDX=${LEVEL_IDX}
MEMR_IDX=$((MEMR_TH_IDX / N_MEM_TH))
MEMTH_IDX=$((MEMR_TH_IDX % N_MEM_TH))

BASE_SEED=${BASE_SEEDS[$BASE_SEED_IDX]}
MEM_R=${MEM_R_LIST[$MEMR_IDX]}
MEM_TH=${MEM_TH_LIST[$MEMTH_IDX]}

echo "[jobinfo] base_seed=${BASE_SEED} rep=${REPEAT_IDX} mem_r=${MEM_R} mem_th=${MEM_TH}"

# ---------------------
# Output folder
# ---------------------
BASE_DIR="RL_models/sb3_stored_models/all_agents/agents_with_noise"

SAFE_MAX_IN_MEMORY_TIME="${MAX_IN_MEMORY_TIME//./p}"
SAFE_MR="${MEM_R//./p}"
SAFE_MT="${MEM_TH//./p}"

SUB_RUN_NAME="ff${NUM_OBS_FF}_mem${SAFE_MAX_IN_MEMORY_TIME}_${IDENTITY_SLOT_STRATEGY}"
AGENT_RUN_NAME="mr${SAFE_MR}_mt${SAFE_MT}_seed${BASE_SEED}_rep${REPEAT_IDX}"

OVERALL_FOLDER="${BASE_DIR}/${SUB_RUN_NAME}/${AGENT_RUN_NAME}"

mkdir -p "$OVERALL_FOLDER"

# ---------------------
# Run training
# ---------------------
srun python -u multiff_analysis/jobs/sb3/scripts/sb3_script.py \
  --overall-folder "$OVERALL_FOLDER" \
  --base-seed "$BASE_SEED" \
  --max-in-memory-time "$MAX_IN_MEMORY_TIME" \
  --lr "$LR" \
  --num-obs-ff "$NUM_OBS_FF" \
  --n-envs "$N_ENVS" \
  --identity-slot-strategy "$IDENTITY_SLOT_STRATEGY" \
  --v-noise-std "$V_NOISE_STD" \
  --w-noise-std "$W_NOISE_STD" \
  --obs-perc-r "$PERC_R" \
  --obs-perc-th "$PERC_TH" \
  --obs-mem-r "$MEM_R" \
  --obs-mem-th "$MEM_TH"
  # --no-curriculum-training \

echo "[jobinfo] Finished base_seed=${BASE_SEED} rep=${REPEAT_IDX} mem_r=${MEM_R} mem_th=${MEM_TH}"
#!/bin/bash
#SBATCH -J sb3_cond
#SBATCH --partition=gpu,xaqq
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH -t 24:00:00
#SBATCH --array=0-23
#SBATCH -o /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/cond/sb3_cond-%A_%a.out
#SBATCH -e /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/cond/sb3_cond-%A_%a.err

set -euo pipefail
cd /user_data/cicid/Multifirefly-Project

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiff_clean

# ---------------------------------------
# Choose which parameter to sweep
# ---------------------------------------
: "${SWEEP_PARAM:?Must set SWEEP_PARAM=perc_r|perc_th|mem_r|mem_th}"

# 6 discrete levels, base seeds, and N repeats per (level, base_seed)
LEVELS=(0.5 1.0 1.5 2.0 2.5 3.0)
BASE_SEEDS=(100 101 102 103 104 105 106 107 108 109 110 111 112 113 114)
NUM_REPEATS=5
NUM_LEVELS=${#LEVELS[@]}
NUM_BASE_SEEDS=${#BASE_SEEDS[@]}

COMBOS_PER_LEVEL=$((NUM_BASE_SEEDS * NUM_REPEATS))
LEVEL_IDX=$((SLURM_ARRAY_TASK_ID / COMBOS_PER_LEVEL))
REST=$((SLURM_ARRAY_TASK_ID % COMBOS_PER_LEVEL))
BASE_SEED_IDX=$((REST / NUM_REPEATS))
REPEAT_IDX=$((REST % NUM_REPEATS))
SWEEP_VALUE=${LEVELS[$LEVEL_IDX]}
BASE_SEED=${BASE_SEEDS[$BASE_SEED_IDX]}

# ---------------------------------------
# Fix all other params to zero
# ---------------------------------------

PERC_R=0.0
PERC_TH=0.0
MEM_R=0.0
MEM_TH=0.0

case "$SWEEP_PARAM" in
    perc_r)
        PERC_R=$SWEEP_VALUE
        ;;
    perc_th)
        PERC_TH=$SWEEP_VALUE
        ;;
    mem_r)
        MEM_R=$SWEEP_VALUE
        ;;
    mem_th)
        MEM_TH=$SWEEP_VALUE
        ;;
    *)
        echo "Invalid SWEEP_PARAM"
        exit 1
        ;;
esac

echo "[info] Sweeping $SWEEP_PARAM = $SWEEP_VALUE (base_seed=$BASE_SEED, repeat=$REPEAT_IDX)"
echo "[info] perc_r=$PERC_R perc_th=$PERC_TH mem_r=$MEM_R mem_th=$MEM_TH"

# ---------------------------------------
# Fixed hyperparameters
# ---------------------------------------

LR=3e-4
NUM_OBS_FF=5
MAX_IN_MEMORY_TIME=2.5
IDENTITY_SLOT_STRATEGY="drop_fill_visible_only"
V_NOISE_STD=0.0
W_NOISE_STD=0.0
N_ENVS=1

# ---------------------------------------
# Output folder
# ---------------------------------------
SAFE_MAX_IN_MEMORY_TIME="${MAX_IN_MEMORY_TIME//./p}"

BASE_DIR="RL_models/sb3_stored_models/all_agents/sb3_conditional"
SUB_RUN_NAME="ff${NUM_OBS_FF}_mem${SAFE_MAX_IN_MEMORY_TIME}_${IDENTITY_SLOT_STRATEGY}"
RUN_NAME="${SWEEP_PARAM}_${SWEEP_VALUE//./p}_seed${BASE_SEED}_rep${REPEAT_IDX}"
OVERALL_FOLDER="${BASE_DIR}/${SUB_RUN_NAME}/${RUN_NAME}"

mkdir -p "$OVERALL_FOLDER"

# ---------------------------------------
# Run training
# ---------------------------------------

srun python -u multiff_analysis/jobs/sb3/scripts/sb3_script.py \
  --overall-folder "$OVERALL_FOLDER" \
  --base-seed "$BASE_SEED" \
  --obs-perc-r "$PERC_R" \
  --obs-perc-th "$PERC_TH" \
  --obs-mem-r "$MEM_R" \
  --obs-mem-th "$MEM_TH" \
  --lr "$LR" \
  --num-obs-ff "$NUM_OBS_FF" \
  --max-in-memory-time "$MAX_IN_MEMORY_TIME" \
  --identity-slot-strategy "$IDENTITY_SLOT_STRATEGY" \
  --v-noise-std "$V_NOISE_STD" \
  --w-noise-std "$W_NOISE_STD" \
  --n-envs "$N_ENVS"
  #--no-curriculum-training
  
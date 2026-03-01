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

# 8 discrete levels, 3 seeds per level
LEVELS=(0.5 1.0 1.5 2.0 2.5 3.0)
SEEDS=(0 1 2 3 4 5 6 7)
NUM_LEVELS=${#LEVELS[@]}
NUM_SEEDS=${#SEEDS[@]}

LEVEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
SWEEP_VALUE=${LEVELS[$LEVEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

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

echo "[info] Sweeping $SWEEP_PARAM = $SWEEP_VALUE (seed=$SEED)"
echo "[info] perc_r=$PERC_R perc_th=$PERC_TH mem_r=$MEM_R mem_th=$MEM_TH"

# ---------------------------------------
# Fixed hyperparameters
# ---------------------------------------

LR=3e-4
NUM_OBS_FF=4
MAX_IN_MEMORY_TIME=3
IDENTITY_SLOT_STRATEGY="drop_fill_visible_only"
V_NOISE_STD=0.0
W_NOISE_STD=0.0
N_ENVS=1

# ---------------------------------------
# Output folder
# ---------------------------------------
SAFE_MAX_IN_MEMORY_TIME="${MAX_IN_MEMORY_TIME//./p}"

BASE_DIR="multiff_analysis/RL_models/sb3_stored_models/all_agents/sb3_conditional"
SUB_RUN_NAME="ff${NUM_OBS_FF}_mem${SAFE_MAX_IN_MEMORY_TIME}_${IDENTITY_SLOT_STRATEGY}"
RUN_NAME="${SWEEP_PARAM}_${SWEEP_VALUE//./p}_seed${SEED}"
OVERALL_FOLDER="${BASE_DIR}/${SUB_RUN_NAME}/${RUN_NAME}"

mkdir -p "$OVERALL_FOLDER"

# ---------------------------------------
# Run training
# ---------------------------------------

srun python -u multiff_analysis/jobs/sb3/scripts/sb3_script.py \
  --overall-folder "$OVERALL_FOLDER" \
  --seed "$SEED" \
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
  --n-envs "$N_ENVS" \
  --no-curriculum-training
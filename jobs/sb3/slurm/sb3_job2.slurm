#!/bin/bash
#SBATCH -J sb3_sweep
#SBATCH --partition=gpu,xaqq
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH -t 24:00:00
#SBATCH --array=1-300%10
#SBATCH -o /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_stdout/sb3_sweep-%A_%a.out
#SBATCH -e /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_stdout/sb3_sweep-%A_%a.err
#SBATCH --exclude=mind-1-29

set -euo pipefail
cd /user_data/cicid/Multifirefly-Project

# activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiff_clean

N_SAMPLES=300
RANDOM_SEED=12345

# ---------------------
# Define hyperparameter lists
# ---------------------
declare -a V_NOISE_STDS=(0.0)
declare -a W_NOISE_STDS=(0.0)
declare -a PERC_R_LIST=(0.0 0.05 0.1 0.2)
declare -a PERC_TH_LIST=(0.0 0.05 0.1 0.2)
declare -a MEM_R_LIST=(0.0 0.05 0.1 0.2)
declare -a MEM_TH_LIST=(0.0 0.05 0.1 0.2)

N_ENVS=1

declare -a LRS=(3e-4)
declare -a NUM_OBS_FF_LIST=(4)
declare -a MAX_IN_MEMORY_TIME_LIST=(2)
declare -a IDENTITY_SLOT_STRATEGY_LIST=("drop_fill_visible_only")

# Compute total combinations across all lists (dims with singletons don't expand the search)
TOTAL_COMB=$(( \
  ${#LRS[@]} * \
  ${#NUM_OBS_FF_LIST[@]} * \
  ${#MAX_IN_MEMORY_TIME_LIST[@]} * \
  ${#IDENTITY_SLOT_STRATEGY_LIST[@]} * \
  ${#V_NOISE_STDS[@]} * \
  ${#W_NOISE_STDS[@]} * \
  ${#PERC_R_LIST[@]} * \
  ${#PERC_TH_LIST[@]} * \
  ${#MEM_R_LIST[@]} * \
  ${#MEM_TH_LIST[@]} \
))

TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))  # 0-based

if (( TASK_ID >= TOTAL_COMB )); then
  echo "TASK_ID exceeds TOTAL_COMB"
  exit 0
fi


# Deterministic pseudo-random mapping from task_id -> flat index
IDX=$(python - <<EOF
import numpy as np
rng = np.random.default_rng(${RANDOM_SEED})
perm = rng.permutation(${TOTAL_COMB})
print(int(perm[${TASK_ID}]))
EOF
)


# Map flat index to multi-dim indices (least significant dimension first)
_i=$IDX
LR_IDX=$((_i % ${#LRS[@]}));                      _i=$((_i / ${#LRS[@]}))
OBS_IDX=$((_i % ${#NUM_OBS_FF_LIST[@]}));         _i=$((_i / ${#NUM_OBS_FF_LIST[@]}))
MEM_IDX=$((_i % ${#MAX_IN_MEMORY_TIME_LIST[@]})); _i=$((_i / ${#MAX_IN_MEMORY_TIME_LIST[@]}))
STRAT_IDX=$((_i % ${#IDENTITY_SLOT_STRATEGY_LIST[@]})); _i=$((_i / ${#IDENTITY_SLOT_STRATEGY_LIST[@]}))
VNOISE_IDX=$((_i % ${#V_NOISE_STDS[@]}));         _i=$((_i / ${#V_NOISE_STDS[@]}))
WNOISE_IDX=$((_i % ${#W_NOISE_STDS[@]}));         _i=$((_i / ${#W_NOISE_STDS[@]}))
PERCR_IDX=$((_i % ${#PERC_R_LIST[@]}));           _i=$((_i / ${#PERC_R_LIST[@]}))
PERCTH_IDX=$((_i % ${#PERC_TH_LIST[@]}));         _i=$((_i / ${#PERC_TH_LIST[@]}))
MEMR_IDX=$((_i % ${#MEM_R_LIST[@]}));             _i=$((_i / ${#MEM_R_LIST[@]}))
MEMTH_IDX=$((_i % ${#MEM_TH_LIST[@]}))

LR=${LRS[$LR_IDX]}
NUM_OBS_FF=${NUM_OBS_FF_LIST[$OBS_IDX]}
MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME_LIST[$MEM_IDX]}
IDENTITY_SLOT_STRATEGY=${IDENTITY_SLOT_STRATEGY_LIST[$STRAT_IDX]}
V_NOISE_STD=${V_NOISE_STDS[$VNOISE_IDX]}
W_NOISE_STD=${W_NOISE_STDS[$WNOISE_IDX]}
PERC_R=${PERC_R_LIST[$PERCR_IDX]}
PERC_TH=${PERC_TH_LIST[$PERCTH_IDX]}
MEM_R=${MEM_R_LIST[$MEMR_IDX]}
MEM_TH=${MEM_TH_LIST[$MEMTH_IDX]}

# ---------------------
# Create a unique folder name
# ---------------------
BASE_DIR="multiff_analysis/RL_models/sb3_stored_models/all_agents/agents_with_noise"
SAFE_MAX_IN_MEMORY_TIME="${MAX_IN_MEMORY_TIME//./p}"
SAFE_VN="${V_NOISE_STD//./p}"
SAFE_WN="${W_NOISE_STD//./p}"
SAFE_PR="${PERC_R//./p}"
SAFE_PT="${PERC_TH//./p}"
SAFE_MR="${MEM_R//./p}"
SAFE_MT="${MEM_TH//./p}"

sanitize_zero() {
    [[ "$1" == "0p0" ]] && echo "0" || echo "$1"
}

SAFE_VN=$(sanitize_zero "$SAFE_VN")
SAFE_WN=$(sanitize_zero "$SAFE_WN")
SAFE_PR=$(sanitize_zero "$SAFE_PR")
SAFE_PT=$(sanitize_zero "$SAFE_PT")
SAFE_MR=$(sanitize_zero "$SAFE_MR")
SAFE_MT=$(sanitize_zero "$SAFE_MT")

SUB_RUN_NAME="ff${NUM_OBS_FF}_mem${SAFE_MAX_IN_MEMORY_TIME}_${IDENTITY_SLOT_STRATEGY}"
AGENT_RUN_NAME="vn${SAFE_VN}_wn${SAFE_WN}_pr${SAFE_PR}_pt${SAFE_PT}_mr${SAFE_MR}_mt${SAFE_MT}"
OVERALL_FOLDER="${BASE_DIR}/${SUB_RUN_NAME}/${AGENT_RUN_NAME}"

mkdir -p "$OVERALL_FOLDER"

# Ensure logs directory exists and define error log path for tee
LOG_DIR="/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_stdout"
ERR_FILE="$LOG_DIR/sb3_sweep-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mkdir -p "$LOG_DIR"

# ---------------------
# Run training
# ---------------------
echo "[jobinfo] Starting sb3 training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}, STRAT=${IDENTITY_SLOT_STRATEGY}, N_ENVS=${N_ENVS}, v_noise=${V_NOISE_STD}, w_noise=${W_NOISE_STD}, perc_r=${PERC_R}, perc_th=${PERC_TH}, mem_r=${MEM_R}, mem_th=${MEM_TH}"
srun --gres=gpu:1 python -u multiff_analysis/jobs/sb3/scripts/sb3_script.py \
  --overall-folder "$OVERALL_FOLDER" \
  --max-in-memory-time "$MAX_IN_MEMORY_TIME" \
  --lr "$LR" \
  --num-obs-ff "$NUM_OBS_FF" \
  --n-envs "$N_ENVS" \
  --no-curriculum-training \
  --identity-slot-strategy "$IDENTITY_SLOT_STRATEGY" \
  --v-noise-std "$V_NOISE_STD" \
  --w-noise-std "$W_NOISE_STD" \
  --obs-perc-r "$PERC_R" \
  --obs-perc-th "$PERC_TH" \
  --obs-mem-r "$MEM_R" \
  --obs-mem-th "$MEM_TH" \
  2> >(tee -a "$ERR_FILE" >&2)
  

echo "[jobinfo] Finished sb3 training for LR=${LR}, NUM_OBS_FF=${NUM_OBS_FF}, MAX_IN_MEMORY_TIME=${MAX_IN_MEMORY_TIME}, STRAT=${IDENTITY_SLOT_STRATEGY}, N_ENVS=${N_ENVS}"


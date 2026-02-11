#!/bin/bash
#SBATCH -J rppo_cost_sweep
#SBATCH --partition=xaqq,gpu
#SBATCH --exclude=mind-1-29,mind-1-15
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=5G
#SBATCH -t 24:00:00
#SBATCH --array=1-18

set -euo pipefail

# ---------------------
# Setup and environment
# ---------------------
cd /user_data/cicid/Multifirefly-Project

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiff_clean

python - <<'PY'
import importlib, sys, subprocess
missing = []
for m in ('stable_baselines3', 'sb3_contrib'):
    try:
        importlib.import_module(m)
    except ImportError:
        missing.append(m)
if missing:
    print('[setup] Installing rppo deps:', ', '.join(missing))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'stable-baselines3==2.4.0', 'sb3-contrib==2.4.0'])
else:
    print('[setup] rppo deps already present')
PY

# ---------------------
# Define sweep values
# ---------------------
BASE_RUN="/user_data/cicid/Multifirefly-Project/multiff_analysis/RL_models/rppo_stored_models/all_agents/ff4_mem2_drop_fill_lr3e-4_nenvs1_job194758_20"

declare -a DV_LIST=(0.5 0.7)
declare -a JK_LIST=(0.5 0.7 0.9)
declare -a CPS_LIST=(6 7 8)
declare -a DW_LIST=(0)
declare -a W_LIST=(0)

N_DV=${#DV_LIST[@]}
N_DW=${#DW_LIST[@]}
N_W=${#W_LIST[@]}
N_JK=${#JK_LIST[@]}
N_CPS=${#CPS_LIST[@]}

TOTAL=$((N_DV * N_DW * N_W * N_JK * N_CPS))
EXPECTED_COUNT=${SLURM_ARRAY_TASK_COUNT:-$TOTAL}
if [[ $TOTAL -ne $EXPECTED_COUNT ]]; then
  echo "[warn] TOTAL combos ($TOTAL) != SBATCH --array size ($EXPECTED_COUNT). Adjust --array or lists." >&2
fi

IDX=$((SLURM_ARRAY_TASK_ID - 1))
DV_IDX=$(( IDX / (N_DW * N_W * N_JK * N_CPS) ))
DW_IDX=$(( (IDX / (N_W * N_JK * N_CPS)) % N_DW ))
W_IDX=$(( (IDX / (N_JK * N_CPS)) % N_W ))
JK_IDX=$(( (IDX / N_CPS) % N_JK ))
CPS_IDX=$(( IDX % N_CPS ))

DV=${DV_LIST[$DV_IDX]}
DW=${DW_LIST[$DW_IDX]}
W=${W_LIST[$W_IDX]}
JERK=${JK_LIST[$JK_IDX]}
CPS=${CPS_LIST[$CPS_IDX]}

echo "[jobinfo] Sweep combo: dv=${DV}, dw=${DW}, w=${W}, jerk=${JERK}, cps=${CPS}"

# ---------------------
# Dynamic log filenames
# ---------------------
# ---------------------
# Dynamic log filenames
# ---------------------
LOG_DIR="/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rppo/logs/run_stdout"
mkdir -p "$LOG_DIR"

fmt_cost() {
  local s
  printf -v s "%.6f" "$1"
  shopt -s extglob             # enable extended globbing
  s="${s%%+([0])}"             # remove trailing zeros
  s="${s%%.}"                  # remove trailing dot if left
  s="${s//./p}"                # replace decimal point with 'p'
  shopt -u extglob             # disable extended globbing
  echo "$s"
}

COST_TAG="dv$(fmt_cost "$DV")_jerk$(fmt_cost "$JERK")_stop$(fmt_cost "$CPS")"

# Include both array job id and unique job id in filenames
OUT_FILE="${LOG_DIR}/cost_sweep-${COST_TAG}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERR_FILE="${LOG_DIR}/cost_sweep-${COST_TAG}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

echo "[jobinfo] Writing logs to:"
echo "  stdout -> $OUT_FILE"
echo "  stderr -> $ERR_FILE"

# ---------------------
# Train agent
# ---------------------
INIT_FROM="${BASE_RUN}"
MODEL_FOLDER_NAME="${BASE_RUN}/${COST_TAG}"

srun python -u multiff_analysis/jobs/rppo/scripts/rppo_script.py \
  --overall-folder "$MODEL_FOLDER_NAME" \
  --init-from "$INIT_FROM" \
  --model-folder-name "$COST_TAG" \
  --n-envs 1 \
  --no-curriculum-training \
  --dv-cost-factor "$DV" \
  --dw-cost-factor "$DW" \
  --w-cost-factor "$W" \
  --jerk-cost-factor "$JERK" \
  --cost-per-stop "$CPS" \
  > "$OUT_FILE" 2> "$ERR_FILE"

echo "[jobinfo] Completed combo: dv=${DV}, dw=${DW}, w=${W}, jerk=${JERK}, cps=${CPS}"

#!/bin/bash
#SBATCH --job-name=decode
#SBATCH --output=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/run_stdout/decode_%A_%a.out
#SBATCH --error=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/run_stdout/decode_%A_%a.err
#SBATCH --array=0-4
#SBATCH --partition=cpu,xaqq
#SBATCH --cpus-per-task=5
#SBATCH --mem=5G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

# ==============================
#  Robust SLURM job: decode
# ==============================

set -euo pipefail
# trap 'echo "[SLURM] Error on line $LINENO: $BASH_COMMAND" >&2' ERR
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/run_stdout

# --- Project setup ---
cd /user_data/cicid/Multifirefly-Project || { echo "[SLURM] Failed to cd into project"; exit 1; }

# --- Conda activation (robust) ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "[SLURM] conda.sh not found" >&2
    exit 1
fi

conda activate multiff_clean || { echo "[SLURM] Conda activation failed"; exit 1; }

echo "[SLURM] Using conda env: $(which python)"
echo "[SLURM] Job $SLURM_JOB_ID | Array index $SLURM_ARRAY_TASK_ID"
echo "[SLURM] Using $SLURM_CPUS_PER_TASK CPU cores"

# --- Model configs ---
# Allow MODELS to be provided by caller as space- or comma-delimited string
if [ -n "${MODELS:-}" ]; then
    MODELS_SANITIZED="$(echo "$MODELS" | tr ',' ' ' | xargs)"
    read -r -a MODELS <<< "$MODELS_SANITIZED"
else
    MODELS=("svm" "logreg")
fi
# Enable cumulative mode by setting CUMULATIVE=1 in environment before sbatch
CUMULATIVE="${CUMULATIVE:-0}"

for MODEL in "${MODELS[@]}"; do
    case "$MODEL" in
        svm) KWARGS='{"C":2.0,"gamma":0.05}' ;;
        *)   KWARGS='{}' ;;
    esac

    echo "[SLURM] Running model $MODEL for comparison index $SLURM_ARRAY_TASK_ID"
    # Build args and optionally append --cumulative
    PY_ARGS=(-u multiff_analysis/jobs/decode/scripts/decode_script.py \
        --comparisons multiff_analysis/configs/comparisons.json \
        --idx "$SLURM_ARRAY_TASK_ID" \
        --model "$MODEL" \
        --model_kwargs "$KWARGS" \
        --n_jobs "$SLURM_CPUS_PER_TASK" \
        --n_perm 1000 \
        --do_testing \
        --tune \
        --exists_ok \
        --iterate-all)
    if [ "$CUMULATIVE" = "1" ]; then
        PY_ARGS+=(--cumulative)
        echo "[SLURM] Cumulative mode enabled"
    fi
    python "${PY_ARGS[@]}"
done

echo "[SLURM] Job $SLURM_JOB_ID completed for index $SLURM_ARRAY_TASK_ID"

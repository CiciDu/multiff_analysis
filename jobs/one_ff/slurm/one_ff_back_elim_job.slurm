#!/bin/bash
#SBATCH --job-name=one_ff_back_elim
#SBATCH --output=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff/logs/back_elim/one_ff_back_elim_%A_%a.out
#SBATCH --error=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff/logs/back_elim/one_ff_back_elim_%A_%a.err
#SBATCH --partition=cpu,xaqq
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

set -e
set -o pipefail

# ------------------------------
# Paths
# ------------------------------
PROJECT_ROOT=/user_data/cicid/Multifirefly-Project
SCRIPT_PATH=$PROJECT_ROOT/multiff_analysis/jobs/one_ff/scripts/one_ff_back_elim_script.py

cd "$PROJECT_ROOT"

echo "[SLURM][START] Job started at $(date)" >&2
echo "[SLURM][DEBUG] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2

# ------------------------------
# Conda setup - FAST PATH (bypass conda activate)
# ------------------------------
# Direct path to multiff_clean environment's Python (much faster than conda activate)
MULTIFF_PYTHON="$HOME/miniconda3/envs/multiff_clean/bin/python"

echo "[SLURM][DEBUG] Checking for multiff_clean Python at: $MULTIFF_PYTHON" >&2
if [[ ! -f "$MULTIFF_PYTHON" ]]; then
  echo "[SLURM][ERROR] multiff_clean Python not found at $MULTIFF_PYTHON" >&2
  echo "[SLURM][ERROR] Falling back to conda activate..." >&2
  
  # Fallback to traditional conda activate
  CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
  if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    conda activate multiff_clean || {
      echo '[SLURM][ERROR] Conda activation failed' >&2
      exit 1
    }
    MULTIFF_PYTHON="python"
  else
    echo '[SLURM][ERROR] conda.sh not found' >&2
    exit 1
  fi
fi

echo "[SLURM][CONDA] Environment ready at $(date)" >&2
echo "[SLURM][DEBUG] Python path: $MULTIFF_PYTHON" >&2
echo "[SLURM][DEBUG] Python version: $($MULTIFF_PYTHON --version 2>&1)" >&2

# ------------------------------
# Now enable nounset
# ------------------------------
set -u

# ------------------------------
# Unit index
# ------------------------------
UNIT_IDX=${SLURM_ARRAY_TASK_ID}

echo "[SLURM] Running one-FF backward elimination" >&2
echo "[SLURM] Unit index: ${UNIT_IDX}" >&2
echo "[SLURM] Host: $(hostname)" >&2
echo "[SLURM] Start time: $(date)" >&2

# Force flush to log files
exec 1>&1 2>&2

# ------------------------------
# Run backward elimination
# ------------------------------
echo "[SLURM][DEBUG] About to run Python script: $SCRIPT_PATH" >&2
echo "[SLURM][DEBUG] With unit_idx: $UNIT_IDX" >&2

stdbuf -oL -eL "$MULTIFF_PYTHON" -u "$SCRIPT_PATH" \
    --unit_idx "${UNIT_IDX}"

echo "[SLURM] Done" >&2
echo "[SLURM] End time: $(date)" >&2

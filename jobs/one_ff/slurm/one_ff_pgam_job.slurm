#!/usr/bin/env bash
#SBATCH --job-name=one_ff_pgam
#SBATCH --output=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff/logs/pgam/one_ff_pgam_%A_%a.out
#SBATCH --error=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff/logs/pgam/one_ff_pgam_%A_%a.err
#SBATCH --partition=cpu,xaqq
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

# --------------------------------------------------
# Shell safety
# --------------------------------------------------
set -e
set -o pipefail

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT='/user_data/cicid/Multifirefly-Project'
SCRIPT_PATH="$PROJECT_ROOT/multiff_analysis/jobs/one_ff/scripts/one_ff_pgam_script.py"

cd "$PROJECT_ROOT"

# Early status print (helps diagnose delays)
echo "[SLURM][START] Job started at $(date)" >&2
echo "[SLURM][DEBUG] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2

# --------------------------------------------------
# Conda setup - FAST PATH (bypass conda activate)
# --------------------------------------------------
# Direct path to pgam environment's Python (much faster than conda activate)
PGAM_PYTHON="$HOME/miniconda3/envs/pgam/bin/python"

echo "[SLURM][DEBUG] Checking for pgam Python at: $PGAM_PYTHON" >&2
if [[ ! -f "$PGAM_PYTHON" ]]; then
  echo "[SLURM][ERROR] pgam Python not found at $PGAM_PYTHON" >&2
  echo "[SLURM][ERROR] Falling back to conda activate..." >&2
  
  # Fallback to traditional conda activate
  CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
  if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    conda activate pgam || {
      echo '[SLURM][ERROR] Conda activation failed' >&2
      exit 1
    }
    PGAM_PYTHON="python"
  else
    echo '[SLURM][ERROR] conda.sh not found' >&2
    exit 1
  fi
fi

echo "[SLURM][CONDA] Environment ready at $(date)" >&2
echo "[SLURM][DEBUG] Python path: $PGAM_PYTHON" >&2
echo "[SLURM][DEBUG] Python version: $($PGAM_PYTHON --version 2>&1)" >&2

# --------------------------------------------------
# Now enable nounset
# --------------------------------------------------
set -u

# --------------------------------------------------
# Array index
# --------------------------------------------------
UNIT_IDX="${SLURM_ARRAY_TASK_ID}"

echo '[SLURM] Running one-FF PGAM'
echo "[SLURM] Unit index: $UNIT_IDX"
echo "[SLURM] Host: $(hostname)"
echo "[SLURM] Start time: $(date)"

# Force flush to log files
exec 1>&1 2>&2

# --------------------------------------------------
# Run PGAM
# --------------------------------------------------
echo "[SLURM][DEBUG] About to run Python script: $SCRIPT_PATH" >&2
echo "[SLURM][DEBUG] With unit_idx: $UNIT_IDX" >&2

stdbuf -oL -eL "$PGAM_PYTHON" -u "$SCRIPT_PATH" \
  --unit_idx "$UNIT_IDX"

echo '[SLURM] Done'
echo "[SLURM] End time: $(date)"
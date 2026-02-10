#!/usr/bin/env bash
#SBATCH --job-name=one_ff_my_gam
#SBATCH --output=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff/logs/my_gam/one_ff_my_gam_%A_%a.out
#SBATCH --error=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff/logs/my_gam/one_ff_my_gam_%A_%a.err
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
SCRIPT_PATH="$PROJECT_ROOT/multiff_analysis/jobs/one_ff/scripts/one_ff_my_gam_script.py"

cd "$PROJECT_ROOT"

# --------------------------------------------------
# Conda setup
# --------------------------------------------------
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

if [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
else
  echo '[SLURM][ERROR] conda.sh not found' >&2
  exit 1
fi

conda activate multiff_clean || {
  echo '[SLURM][ERROR] Conda activation failed' >&2
  exit 1
}

# --------------------------------------------------
# Now enable nounset
# --------------------------------------------------
set -u

# --------------------------------------------------
# Array index
# --------------------------------------------------
UNIT_IDX="${SLURM_ARRAY_TASK_ID}"

echo '[SLURM] Running one-FF MY GAM'
echo "[SLURM] Unit index: $UNIT_IDX"
echo "[SLURM] Host: $(hostname)"
echo "[SLURM] Start time: $(date)"

# --------------------------------------------------
# Run MY GAM
# --------------------------------------------------
python -u "$SCRIPT_PATH" \
  --unit_idx "$UNIT_IDX"

echo '[SLURM] Done'
echo "[SLURM] End time: $(date)"
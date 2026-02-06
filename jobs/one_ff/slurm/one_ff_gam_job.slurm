#!/bin/bash
#SBATCH --job-name=one_ff_gam
#SBATCH --output=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff/logs/run_stdout/one_ff_%A_%a.out
#SBATCH --error=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff/logs/run_stdout/one_ff_%A_%a.err
#SBATCH --partition=cpu,xaqq
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ------------------------------
# Paths
# ------------------------------
PROJECT_ROOT=/user_data/cicid/Multifirefly-Project
SCRIPT_PATH=$PROJECT_ROOT/multiff_analysis/jobs/one_ff/scripts/one_ff_gam_script.py

cd "$PROJECT_ROOT"

# ------------------------------
# Conda
# ------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "[SLURM][ERROR] conda.sh not found" >&2
    exit 1
fi

conda activate pgam || {
    echo "[SLURM][ERROR] Conda activation failed"
    exit 1
}

# ------------------------------
# Unit index
# ------------------------------
UNIT_IDX=${SLURM_ARRAY_TASK_ID}

echo "[SLURM] Running one-FF PGAM"
echo "[SLURM] Unit index: ${UNIT_IDX}"
echo "[SLURM] Host: $(hostname)"
echo "[SLURM] Start time: $(date)"

# ------------------------------
# Run PGAM
# ------------------------------
python -u "$SCRIPT_PATH" \
    --unit_idx "${UNIT_IDX}"

echo "[SLURM] Done"
echo "[SLURM] End time: $(date)"

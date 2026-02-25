#!/usr/bin/env bash
#SBATCH --job-name=one_ff_decoding
#SBATCH --output=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff_decoding/logs/decoding/one_ff_decoding_%j.out
#SBATCH --error=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/one_ff_decoding/logs/decoding/one_ff_decoding_%j.err
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
SCRIPT_PATH="$PROJECT_ROOT/multiff_analysis/jobs/one_ff_decoding/scripts/one_ff_decoding_script.py"

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

echo '[SLURM] Running One-FF Decoding'
echo "[SLURM] Host: $(hostname)"
echo "[SLURM] Start time: $(date)"

# --------------------------------------------------
# Run One-FF Decoding
# --------------------------------------------------
# save_dir is automatic: output_root/session_N (default: all_monkey_data/one_ff_data/decoding/session_0)
python -u "$SCRIPT_PATH" --session_num 0

echo '[SLURM] Done'
echo "[SLURM] End time: $(date)"
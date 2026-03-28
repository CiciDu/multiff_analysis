#!/bin/bash
set -euo pipefail

# Pre-create SLURM log dir before submission (SLURM writes logs itself)
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_stdout
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_curriculum
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_summary

ARRAY_MAX=1000
MAX_CONCURRENT=10

echo "[SUBMIT] Submitting array: 0-$ARRAY_MAX (max $MAX_CONCURRENT concurrent)"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch --array=1-"$ARRAY_MAX"%$MAX_CONCURRENT "$SCRIPT_DIR/sb3_noise_job.slurm" "$@"



#!/bin/bash
set -euo pipefail

# Pre-create SLURM log dir before submission (SLURM writes logs itself)
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rppo/logs/run_stdout
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rppo/logs/run_curriculum
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rppo/logs/run_summary

# Submit the job; pass through any extra sbatch args
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$SCRIPT_DIR/../slurm/rppo_job.slurm" "$@"



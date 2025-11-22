#!/bin/bash
set -euo pipefail

# Pre-create SLURM log dirs
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rppo/logs/run_stdout
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rppo/logs/run_curriculum
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rppo/logs/run_summary

# Submit cost sweep job
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$SCRIPT_DIR/../slurm/rppo_cost_sweep.slurm" "$@"



#!/bin/bash
set -euo pipefail

# Make sure logs exist
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rl_collect/logs/run_stdout
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rl_collect/logs/run_stderr

echo "[info] Splitting agent list into batches..."
python /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rl_collect/scripts/split_agent_list_into_batches.py

# Submit SBATCH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$SCRIPT_DIR/../slurm/rl_collect_job.slurm" "$@"

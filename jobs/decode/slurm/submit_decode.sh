#!/bin/bash
set -euo pipefail

# Pre-create SLURM log dirs before submission (SLURM writes logs itself)
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/run_stdout

# Submit the job; pass through any extra sbatch args
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$SCRIPT_DIR/../slurm/decode_job.slurm" "$@"





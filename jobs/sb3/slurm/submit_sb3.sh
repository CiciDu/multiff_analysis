#!/bin/bash
set -euo pipefail

# Pre-create SLURM log dir before submission (SLURM writes logs itself)
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_stdout
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_curriculum
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/run_summary

# Maximum number of concurrent jobs with this name
MAX_CONCURRENT=3
JOB_NAME="sb3_job"

# Wait if we're at the limit
while true; do
  RUNNING=$(squeue -u "$USER" -n "$JOB_NAME" -h -t PENDING,RUNNING -r | wc -l)
  if [ "$RUNNING" -lt "$MAX_CONCURRENT" ]; then
    break
  fi
  echo "[SUBMIT] $RUNNING jobs running/pending (max: $MAX_CONCURRENT). Waiting 30s..."
  sleep 30
done

# Submit the job; pass through any extra sbatch args
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$SCRIPT_DIR/../slurm/sb3_job.slurm" "$@"



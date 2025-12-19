#!/bin/bash
set -euo pipefail

# Pre-create SLURM log dirs before submission (SLURM writes logs itself)
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/run_stdout

# Parse optional --models flag and forward MODELS via --export (space- or comma-delimited)
MODELS="${MODELS:-}"
FORWARD_ARGS=()
i=0
while [ $i -lt $# ]; do
  arg="${!i}"
  # Bash array indexing for "$@" is tricky; use shift loop
  break
done
# Shift-loop parse to safely capture values
while (( "$#" )); do
  if [ "$1" = "--models" ] && [ -n "${2-}" ]; then
    MODELS="$2"
    shift 2
    continue
  fi
  FORWARD_ARGS+=("$1")
  shift
done

# Default models if none provided
if [ -z "${MODELS:-}" ]; then
  MODELS="svm logreg"
fi

# Submit the job; pass through remaining sbatch args and export MODELS if provided
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${MODELS:-}" ]; then
  sbatch --export=ALL,MODELS="${MODELS}" "$SCRIPT_DIR/../slurm/decode_job.slurm" "${FORWARD_ARGS[@]}"
else
  sbatch --export=ALL "$SCRIPT_DIR/../slurm/decode_job.slurm" "${FORWARD_ARGS[@]}"
fi





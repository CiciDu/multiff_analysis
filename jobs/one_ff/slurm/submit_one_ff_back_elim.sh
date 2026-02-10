#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/user_data/cicid/Multifirefly-Project
JOB_DIR=$PROJECT_ROOT/multiff_analysis/jobs/one_ff
SLURM_SCRIPT=$JOB_DIR/slurm/one_ff_back_elim_job.slurm

LOG_DIR=$JOB_DIR/logs/back_elim
mkdir -p "$LOG_DIR"

usage() {
  cat <<EOF
Usage:
  submit_one_ff_back_elim.sh [options]

Options (handled by this script):
  --units N        Number of units (default: 128)
  --max-parallel N Max concurrent jobs (default: 4)

All other arguments are forwarded to sbatch.

Examples:
  submit_one_ff_back_elim.sh
  submit_one_ff_back_elim.sh --units 96
  submit_one_ff_back_elim.sh --units 128 --max-parallel 8
  submit_one_ff_back_elim.sh -p cpu --time=24:00:00
EOF
}

# ------------------------------
# Defaults
# ------------------------------
N_UNITS=128
MAX_PARALLEL=3
FORWARD_ARGS=()

# ------------------------------
# Parse args
# ------------------------------
while (( "$#" )); do
  case "$1" in
    --units)
      [ -z "${2:-}" ] && { echo "--units requires an argument"; exit 1; }
      N_UNITS="$2"
      shift 2
      ;;
    --max-parallel)
      [ -z "${2:-}" ] && { echo "--max-parallel requires an argument"; exit 1; }
      MAX_PARALLEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

ARRAY_MAX=$((N_UNITS - 1))

echo "[SUBMIT] One-FF Backward Elimination"
echo "[SUBMIT] Units: $N_UNITS (0-$ARRAY_MAX)"
echo "[SUBMIT] Max concurrent: $MAX_PARALLEL"
echo "[SUBMIT] Slurm script: $SLURM_SCRIPT"

# ------------------------------
# Submit job
# ------------------------------
sbatch \
  --array=0-"$ARRAY_MAX"%$MAX_PARALLEL \
  ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"} \
  "$SLURM_SCRIPT"

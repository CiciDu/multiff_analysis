#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/user_data/cicid/Multifirefly-Project
JOB_DIR=$PROJECT_ROOT/multiff_analysis/jobs/glm
CONFIG_DIR=$PROJECT_ROOT/multiff_analysis/jobs/data_configs

mkdir -p "$JOB_DIR/logs/run_stdout"

usage() {
  cat <<EOF
Usage:
  submit_glm.sh --monkey MONKEY_NAME [sbatch args]

Example:
  submit_glm.sh --monkey Bruno
  submit_glm.sh --monkey Schro -p cpu --time=12:00:00
EOF
}

# ------------------------------
# Parse args
# ------------------------------
MONKEY_NAME=""
FORWARD_ARGS=()

while (( "$#" )); do
  case "$1" in
    --monkey)
      [ -z "${2:-}" ] && { echo "--monkey requires an argument"; exit 1; }
      MONKEY_NAME="$2"
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

if [ -z "$MONKEY_NAME" ]; then
  echo "[SUBMIT] Error: --monkey is required" >&2
  exit 1
fi

RAW_PATH_FILE=$CONFIG_DIR/raw_data_paths_${MONKEY_NAME}.txt

if [ ! -f "$RAW_PATH_FILE" ]; then
  echo "[SUBMIT] Missing config: $RAW_PATH_FILE" >&2
  exit 1
fi

# ------------------------------
# Determine array size
# ------------------------------
NUM_JOBS=$(wc -l < "$RAW_PATH_FILE")

if [ "$NUM_JOBS" -eq 0 ]; then
  echo "[SUBMIT] No sessions listed in $RAW_PATH_FILE" >&2
  exit 1
fi

ARRAY_MAX=$((NUM_JOBS - 1))
MAX_CONCURRENT=5

echo "[SUBMIT] Monkey: $MONKEY_NAME"
echo "[SUBMIT] Sessions: $NUM_JOBS"
echo "[SUBMIT] Submitting array: 0-$ARRAY_MAX (max $MAX_CONCURRENT concurrent)"

# ------------------------------
# Submit job
# ------------------------------
sbatch \
  --array=0-"$ARRAY_MAX"%$MAX_CONCURRENT \
  "${FORWARD_ARGS[@]}" \
  "$JOB_DIR/slurm/glm_job.slurm" \
  "$MONKEY_NAME"

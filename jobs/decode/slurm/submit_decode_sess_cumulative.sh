#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/user_data/cicid/Multifirefly-Project"
DEFAULT_MONKEY_DIR="$PROJECT_ROOT/all_monkey_data/raw_monkey_data/monkey_Bruno"
MONKEY_DIR="${MONKEY_DIR:-$DEFAULT_MONKEY_DIR}"

if [ ! -d "$MONKEY_DIR" ]; then
  echo "[submit] Monkey dir not found: $MONKEY_DIR" >&2
  exit 1
fi

# Prepare sessions file (cumulative-specific)
LOG_DIR="$PROJECT_ROOT/multiff_analysis/jobs/decode/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
SESS_FILE="$LOG_DIR/sessions_cumulative_${TS}.txt"

find "$MONKEY_DIR" -maxdepth 1 -mindepth 1 -type d -name 'data_*' -printf '%f\n' | sort > "$SESS_FILE"
NUM_SESS="$(wc -l < "$SESS_FILE" | tr -d ' ')"
echo "[submit] Sessions listed: $NUM_SESS -> $SESS_FILE"
ARRAY_SPEC="0-$((NUM_SESS-1))"
echo "[submit] Array spec: $ARRAY_SPEC"

echo "[submit] Submitting decode_sess.slurm with CUMULATIVE=1 and array=$ARRAY_SPEC"
sbatch --array="$ARRAY_SPEC" --export=ALL,MONKEY_DIR="$MONKEY_DIR",SESSIONS_FILE="$SESS_FILE",CUMULATIVE=1,KEYS_CSV="${KEYS_CSV:-}",MODELS_CSV="${MODELS_CSV:-}" "$PROJECT_ROOT/multiff_analysis/jobs/decode/slurm/decode_sess.slurm"

echo "[submit] Done."



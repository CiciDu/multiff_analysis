#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/user_data/cicid/Multifirefly-Project"
DEFAULT_MONKEY_DIR="$PROJECT_ROOT/all_monkey_data/raw_monkey_data/monkey_Bruno"
MONKEY_DIR="${MONKEY_DIR:-$DEFAULT_MONKEY_DIR}"

mkdir -p "$PROJECT_ROOT/multiff_analysis/jobs/decode/logs/run_stdout"

# ---------------------------------------------------------
# Parse flags: --models and --cumulative
# ---------------------------------------------------------
MODELS="${MODELS:-}"
CUMULATIVE=0

ARGS=("$@")
PARSED_ARGS=()

i=0
while [ $i -lt ${#ARGS[@]} ]; do
  case "${ARGS[$i]}" in

    --models)
      if [ $((i+1)) -lt ${#ARGS[@]} ]; then
        MODELS="${ARGS[$((i+1))]}"
        i=$((i+2))
        continue
      fi
      ;;

    --cumulative)
      CUMULATIVE=1
      i=$((i+1))
      continue
      ;;

    *)
      PARSED_ARGS+=("${ARGS[$i]}")
      i=$((i+1))
      ;;
  esac
done

set -- "${PARSED_ARGS[@]}"

# Default models
if [ -z "$MODELS" ]; then
  MODELS="svm logreg rf logreg_elasticnet"
fi

echo "[submit] Models: $MODELS"
echo "[submit] Cumulative mode: $CUMULATIVE"

# ---------------------------------------------------------
# Validate monkey directory
# ---------------------------------------------------------
if [ ! -d "$MONKEY_DIR" ]; then
  echo "[submit] Monkey dir not found: $MONKEY_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------
# Build session list
# ---------------------------------------------------------
LOG_DIR="$PROJECT_ROOT/multiff_analysis/jobs/decode/logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
SESS_FILE="$LOG_DIR/sessions_${TS}.txt"

find "$MONKEY_DIR" -maxdepth 1 -mindepth 1 -type d -name 'data_*' -printf '%f\n' \
  | sort > "$SESS_FILE"

NUM_SESS="$(wc -l < "$SESS_FILE" | tr -d ' ')"
ARRAY_SPEC="0-$((NUM_SESS-1))"

echo "[submit] Sessions listed: $NUM_SESS"
echo "[submit] Array spec: $ARRAY_SPEC"

# ---------------------------------------------------------
# Submit job
# ---------------------------------------------------------
echo "[submit] Submitting decode_sess.slurm"
sbatch \
  --array="$ARRAY_SPEC" \
  --export=ALL,MONKEY_DIR="$MONKEY_DIR",SESSIONS_FILE="$SESS_FILE",CUMULATIVE="$CUMULATIVE",KEYS_CSV="${KEYS_CSV:-}",MODELS="$MODELS" \
  "$PROJECT_ROOT/multiff_analysis/jobs/decode/slurm/decode_sess.slurm"

echo "[submit] Done."

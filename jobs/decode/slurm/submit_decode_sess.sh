#!/bin/bash
set -euo pipefail

# Usage:
#   MONKEY_DIR=/path/to/monkey sbatch jobs/slurm_wrappers/submit_decode_sess.sh
# or:
#   jobs/slurm_wrappers/submit_decode_sess.sh /path/to/monkey
#
# If not provided, defaults to Bruno's raw data directory.

# --- Resolve project-relative paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --- Logs ---
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/run_stdout
WRAP_LOG_DIR="/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/run_stdout"
WRAP_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
WRAP_LOG="$WRAP_LOG_DIR/submit_decode_sess_${WRAP_STAMP}.log"
# Redirect all output of this wrapper to a log file (and also show on console)
exec > >(tee -a "$WRAP_LOG") 2>&1
echo "[wrap] Logging wrapper output to: $WRAP_LOG"

# --- Input monkey directory ---
DEFAULT_MONKEY_DIR="/user_data/cicid/Multifirefly-Project/all_monkey_data/raw_monkey_data/monkey_Bruno"
# If first arg looks like an sbatch flag ("-..."), treat all args as sbatch args
if [[ "${1-}" =~ ^- ]]; then
    MONKEY_DIR="${MONKEY_DIR:-$DEFAULT_MONKEY_DIR}"
    EXTRA_ARGS=("$@")
else
    MONKEY_DIR="${MONKEY_DIR:-${1:-$DEFAULT_MONKEY_DIR}}"
    EXTRA_ARGS=("${@:2}")
fi

if [ ! -d "$MONKEY_DIR" ]; then
    echo "[wrap] Monkey directory not found: $MONKEY_DIR" >&2
    exit 1
fi

# --- Models (keep consistent with decode_sess.slurm) ---
MODELS=("svm" "logreg" "logreg_elasticnet" "rf" "mlp")
MODELS_CSV="$(IFS=,; echo "${MODELS[*]}")"

# --- Discover candidate sessions using a combined progress file (pending means any model remaining) ---
SESSIONS_STR="$(MONKEY_DIR="$MONKEY_DIR" KEYS_CSV="${KEYS_CSV:-}" python - <<'PY'
import os, sys, json, hashlib
from pathlib import Path

def list_sessions(monkey_dir: Path):
    return sorted([p.name for p in monkey_dir.iterdir() if p.is_dir() and p.name.startswith('data_')])

monkey_dir = Path(os.environ.get('MONKEY_DIR', ''))
if not monkey_dir.exists():
    print('', end=''); sys.exit(0)
all_sessions = list_sessions(monkey_dir)
retry_monkey_dir = Path(str(monkey_dir).replace('/raw_monkey_data/', '/retry_decoder/'))
retry_monkey_dir.mkdir(parents=True, exist_ok=True)

keys_csv = os.environ.get('KEYS_CSV', '').strip()
if keys_csv:
    keys_sorted = ','.join(sorted([k.strip() for k in keys_csv.split(',') if k.strip()]))
    keys_hash = hashlib.md5(keys_sorted.encode('utf-8')).hexdigest()[:8]
    progress_path = retry_monkey_dir / f"_decoding_progress_all_k{keys_hash}.json"
else:
    progress_path = retry_monkey_dir / f"_decoding_progress_all.json"

if progress_path.exists():
    try:
        with open(progress_path, 'r') as f:
            saved = json.load(f)
        pend = saved.get('pending', [])
        selected = [p for p in pend if p in all_sessions]
    except Exception:
        selected = all_sessions
else:
    selected = all_sessions
print("\n".join(selected))
PY
)"

if [ -z "$SESSIONS_STR" ]; then
    echo "[wrap] No sessions found under: $MONKEY_DIR" >&2
    exit 0
fi

mapfile -t SESSIONS <<< "$SESSIONS_STR"
NUM_SESS=${#SESSIONS[@]}
if [ "$NUM_SESS" -eq 0 ]; then
    echo "[wrap] No sessions to process." >&2
    exit 0
fi

ARRAY_SPEC="0-$((NUM_SESS-1))"
echo "[wrap] Monkey: $(basename "$MONKEY_DIR") | sessions=$NUM_SESS | array=$ARRAY_SPEC"

# Optional: Show first few sessions
printf '[wrap] First sessions: %s\n' "$(printf '%s ' "${SESSIONS[@]:0:5}")"

# Write sessions to a shared file (avoid commas in --export parsing)
SESS_LIST_DIR="/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/session_logs"
mkdir -p "$SESS_LIST_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SESSIONS_FILE="$SESS_LIST_DIR/sessions_${STAMP}.txt"
printf "%s\n" "${SESSIONS[@]}" > "$SESSIONS_FILE"
echo "[wrap] Wrote session list to: $SESSIONS_FILE"

# --- Submit SLURM job with tailored array and exported MONKEY_DIR ---
sbatch --array="$ARRAY_SPEC" --export=ALL,MONKEY_DIR="$MONKEY_DIR",SESSIONS_FILE="$SESSIONS_FILE",KEYS_CSV="${KEYS_CSV:-}" "$SCRIPT_DIR/../slurm/decode_sess.slurm" "${EXTRA_ARGS[@]}"



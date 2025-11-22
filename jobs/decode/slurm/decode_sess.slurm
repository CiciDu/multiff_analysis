#!/bin/bash
#SBATCH --job-name=decode_sess
#SBATCH --output=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/run_stdout/decode_sess_%A_%a.out
#SBATCH --error=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/run_stdout/decode_sess_%A_%a.err
# NOTE: Set this range high enough; out-of-range indices will no-op safely below.
#SBATCH --cpus-per-task=11
#SBATCH --partition=cpu,xaqq
#SBATCH --mem=5G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decode/logs/run_stdout

# --- Project setup ---
cd /user_data/cicid/Multifirefly-Project || { echo "[SLURM] Failed to cd into project"; exit 1; }

# --- Conda activation (robust) ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "[SLURM] conda.sh not found" >&2
    exit 1
fi

conda activate multiff_clean || { echo "[SLURM] Conda activation failed"; exit 1; }

# Ensure worker prints from joblib/multiprocessing reach Slurm logs
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

echo "[SLURM] Using conda env: $(which python)"
echo "[SLURM] Job $SLURM_JOB_ID | Array index $SLURM_ARRAY_TASK_ID"
echo "[SLURM] Using $SLURM_CPUS_PER_TASK CPU cores"

# --- Data discovery: enumerate sessions under a monkey directory ---
# Change this to target a different monkey if needed.
MONKEY_DIR="${MONKEY_DIR:-/user_data/cicid/Multifirefly-Project/all_monkey_data/raw_monkey_data/monkey_Bruno}"
# Enable cumulative mode by setting CUMULATIVE=1 in environment before sbatch
CUMULATIVE="${CUMULATIVE:-0}"
# Models to run for each selected session
# MODELS=("svm" "logreg" "logreg_elasticnet" "rf")
MODELS=("svm" "logreg")
if [ ! -d "$MONKEY_DIR" ]; then
    echo "[SLURM] Monkey directory not found: $MONKEY_DIR" >&2
    exit 1
fi

# Wrapper provides SESSIONS_FILE (newline-delimited). Require it.
if [ -z "${SESSIONS_FILE:-}" ] || [ ! -f "${SESSIONS_FILE:-/nonexistent}" ]; then
    echo "[SLURM] SESSIONS_FILE not provided or missing. Use the wrapper to submit." >&2
    exit 1
fi
mapfile -t SESSIONS < "$SESSIONS_FILE"
NUM_SESS=${#SESSIONS[@]}

if [ "$NUM_SESS" -eq 0 ]; then
    echo "[SLURM] No sessions (data_*) found under: $MONKEY_DIR" >&2
    exit 0
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SESS" ]; then
    echo "[SLURM] Array index $SLURM_ARRAY_TASK_ID out of range (sessions=$NUM_SESS). Exiting." >&2
    exit 0
fi

SESSION_NAME="${SESSIONS[$SLURM_ARRAY_TASK_ID]}"
RAW_DATA="$MONKEY_DIR/$SESSION_NAME"
echo "[SLURM] Selected session: $SESSION_NAME ($RAW_DATA)"

# Build models CSV for progress bookkeeping
MODELS_CSV="$(IFS=,; echo "${MODELS[*]}")"

# --- Optional: reset combined progress (clear all 'done' and per-model done) ---
if [ "${RESET_PROGRESS:-0}" = "1" ]; then
    echo "[SLURM] RESET_PROGRESS=1; resetting progress file (clear done/per_model_done)"
    MONKEY_DIR="$MONKEY_DIR" KEYS_CSV="${KEYS_CSV:-}" MODELS_CSV="$MODELS_CSV" CUMULATIVE="${CUMULATIVE:-0}" python - <<'PY'
import os, json, hashlib
from pathlib import Path
from datetime import datetime

monkey_dir = Path(os.environ.get('MONKEY_DIR', ''))
keys_csv = os.environ.get('KEYS_CSV', '').strip()
models = [m.strip() for m in os.environ.get('MODELS_CSV', '').split(',') if m.strip()]

cumulative = os.environ.get('CUMULATIVE', '0') == '1'
replace_target = '/retry_decoder_cumulative/' if cumulative else '/retry_decoder/'
retry_monkey_dir = Path(str(monkey_dir).replace('/raw_monkey_data/', replace_target))
retry_monkey_dir.mkdir(parents=True, exist_ok=True)

if keys_csv:
    keys_sorted = ','.join(sorted([k.strip() for k in keys_csv.split(',') if k.strip()]))
    keys_hash = hashlib.md5(keys_sorted.encode('utf-8')).hexdigest()[:8]
    progress_filename = f"_decoding_progress_all_k{keys_hash}.json"
else:
    progress_filename = f"_decoding_progress_all.json"
progress_path = retry_monkey_dir / progress_filename

all_sessions = []
if monkey_dir.exists():
    all_sessions = sorted([p.name for p in monkey_dir.iterdir() if p.is_dir() and p.name.startswith('data_')])

progress = {
    'monkey': monkey_dir.name if monkey_dir else None,
    'all': all_sessions,
    'done': [],
    'pending': all_sessions,
    'last_updated': datetime.now().isoformat(timespec='seconds'),
    'keys': [k.strip() for k in keys_csv.split(',') if k.strip()] or None,
    'models': models,
    'per_model_done': {m: [] for m in models} if models else {},
}

with open(progress_path, 'w') as f:
    json.dump(progress, f, indent=2)

print(f"[SLURM] Progress reset: {progress_path} | done_all=0 | pending_all={len(progress['pending'])}")
PY
fi

# --- Combined progress guard: skip whole session if already done for all models ---
SESSION_DONE_ALL="$(MONKEY_DIR="$MONKEY_DIR" SESSION_NAME="$SESSION_NAME" KEYS_CSV="${KEYS_CSV:-}" MODELS_CSV="$MODELS_CSV" CUMULATIVE="${CUMULATIVE:-0}" python - <<'PY'
import os, json, hashlib
from pathlib import Path

def list_sessions(monkey_dir: Path):
    return sorted([p.name for p in monkey_dir.iterdir() if p.is_dir() and p.name.startswith('data_')])

monkey_dir = Path(os.environ.get('MONKEY_DIR', ''))
session = os.environ.get('SESSION_NAME', '')
keys_csv = os.environ.get('KEYS_CSV', '').strip()
models = [m.strip() for m in os.environ.get('MODELS_CSV', '').split(',') if m.strip()]
cumulative = os.environ.get('CUMULATIVE', '0') == '1'
replace_target = '/retry_decoder_cumulative/' if cumulative else '/retry_decoder/'
retry_monkey_dir = Path(str(monkey_dir).replace('/raw_monkey_data/', replace_target))
retry_monkey_dir.mkdir(parents=True, exist_ok=True)
if keys_csv:
    keys_sorted = ','.join(sorted([k.strip() for k in keys_csv.split(',') if k.strip()]))
    keys_hash = hashlib.md5(keys_sorted.encode('utf-8')).hexdigest()[:8]
    progress_filename = f"_decoding_progress_all_k{keys_hash}.json"
else:
    progress_filename = f"_decoding_progress_all.json"
progress_path = retry_monkey_dir / progress_filename

done_all = False
if progress_path.exists():
    try:
        with open(progress_path, 'r') as f:
            saved = json.load(f)
        done_set = set(saved.get('done', []))
        done_all = session in done_set
    except Exception:
        pass
print("1" if done_all else "0", end='')
PY
)"
if [ "$SESSION_DONE_ALL" = "1" ]; then
    echo "[SLURM] Skipping session=$SESSION_NAME (already done for all models per combined progress)"
    exit 0
fi

# --- Model configs: run all models for this session ---
for MODEL in "${MODELS[@]}"; do
    echo "[SLURM] Running model=$MODEL for session=$SESSION_NAME"
    # Build python args, forwarding keys if provided
    PY_ARGS=(-u multiff_analysis/jobs/decode/scripts/decode_script.py
        --comparisons multiff_analysis/configs/comparisons.json
        --model "$MODEL"
        --model_kwargs '{}'
        --n_jobs "$SLURM_CPUS_PER_TASK"
        --do_testing
        --perm_search "grid"
        --exists_ok
        --raw_data "$RAW_DATA"
        --n_perm 1000
        # --tune
        )
    if [ "$CUMULATIVE" = "1" ]; then
        PY_ARGS+=(--cumulative)
        echo "[SLURM] Cumulative mode enabled"
    fi
    if [ -n "${KEYS_CSV:-}" ]; then
        IFS=',' read -r -a KEYS_ARR <<< "$KEYS_CSV"
        if [ "${#KEYS_ARR[@]}" -gt 0 ]; then
            PY_ARGS+=(--keys "${KEYS_ARR[@]}")
        fi
    fi
    python "${PY_ARGS[@]}"

    # On success, update combined progress file: mark this model done; mark session done if all models done
    MONKEY_DIR="$MONKEY_DIR" SESSION_NAME="$SESSION_NAME" MODEL="$MODEL" KEYS_CSV="${KEYS_CSV:-}" MODELS_CSV="$MODELS_CSV" CUMULATIVE="${CUMULATIVE:-0}" python - <<'PY'
import os, json, hashlib
from pathlib import Path
from datetime import datetime

def list_sessions(monkey_dir: Path):
    return sorted([p.name for p in monkey_dir.iterdir() if p.is_dir() and p.name.startswith('data_')])

monkey_dir = Path(os.environ.get('MONKEY_DIR', ''))
session = os.environ.get('SESSION_NAME', '')
model = os.environ.get('MODEL', '')
keys_csv = os.environ.get('KEYS_CSV', '').strip()
models = [m.strip() for m in os.environ.get('MODELS_CSV', '').split(',') if m.strip()]

cumulative = os.environ.get('CUMULATIVE', '0') == '1'
replace_target = '/retry_decoder_cumulative/' if cumulative else '/retry_decoder/'
retry_monkey_dir = Path(str(monkey_dir).replace('/raw_monkey_data/', replace_target))
retry_monkey_dir.mkdir(parents=True, exist_ok=True)
if keys_csv:
    keys_sorted = ','.join(sorted([k.strip() for k in keys_csv.split(',') if k.strip()]))
    keys_hash = hashlib.md5(keys_sorted.encode('utf-8')).hexdigest()[:8]
    progress_filename = f"_decoding_progress_all_k{keys_hash}.json"
else:
    progress_filename = f"_decoding_progress_all.json"
progress_path = retry_monkey_dir / progress_filename

all_sessions = list_sessions(monkey_dir) if monkey_dir.exists() else []
progress = {
    'monkey': monkey_dir.name if monkey_dir else None,
    'all': all_sessions,
    'done': [],               # sessions completed for all models
    'pending': [],            # sessions not yet completed for all models
    'last_updated': None,
    'keys': [k.strip() for k in keys_csv.split(',') if k.strip()] or None,
    'models': models,
    'per_model_done': {m: [] for m in models} if models else {},
}

if progress_path.exists():
    try:
        with open(progress_path, 'r') as f:
            saved = json.load(f)
        # merge/normalize existing fields
        saved_all = saved.get('all', [])
        saved_done = saved.get('done', [])
        saved_pmd = saved.get('per_model_done', {})
        # keep only sessions that still exist
        if all_sessions:
            saved_done = [d for d in saved_done if d in all_sessions]
            for m, lst in saved_pmd.items():
                saved_pmd[m] = [d for d in lst if d in all_sessions]
        progress['done'] = saved_done
        # per_model_done needs to include all models keys
        if models:
            base = {m: [] for m in models}
            base.update({k: v for k, v in saved_pmd.items() if isinstance(v, list)})
            progress['per_model_done'] = base
        else:
            progress['per_model_done'] = saved_pmd
    except Exception:
        pass

# Mark this model as done for the session
if session:
    if model not in progress['per_model_done']:
        progress['per_model_done'][model] = []
    if session not in progress['per_model_done'][model]:
        progress['per_model_done'][model].append(session)

# If all models are done for this session, mark session done
if session and progress.get('models'):
    if all(session in progress['per_model_done'].get(m, []) for m in progress['models']):
        if session not in progress['done']:
            progress['done'].append(session)

# Recompute pending
progress['pending'] = [s for s in all_sessions if s not in progress['done']] if all_sessions else []
progress['last_updated'] = datetime.now().isoformat(timespec='seconds')

with open(progress_path, 'w') as f:
    json.dump(progress, f, indent=2)

print(f"[SLURM] Progress updated: {progress_path} | done_all={len(progress['done'])} | pending_all={len(progress['pending'])}")
PY
done

echo "[SLURM] Job $SLURM_JOB_ID completed for session $SESSION_NAME (index $SLURM_ARRAY_TASK_ID)"



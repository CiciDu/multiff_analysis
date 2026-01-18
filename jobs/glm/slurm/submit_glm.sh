#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/user_data/cicid/Multifirefly-Project
JOB_DIR=$PROJECT_ROOT/multiff_analysis/jobs/glm
RAW_PATH_FILE=$JOB_DIR/raw_data_paths.txt

mkdir -p "$JOB_DIR/logs/run_stdout"

usage() {
  cat <<EOF
Usage:
  submit_glm.sh --monkey MONKEY_NAME [sbatch args]

Example:
  submit_glm.sh --monkey monkey_Bruno
  submit_glm.sh --monkey monkey_Schro -p cpu --time=12:00:00
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

echo "[SUBMIT] Monkey: $MONKEY_NAME"

# ------------------------------
# Generate raw data path list (PYTHON, canonical)
# ------------------------------
echo "[SUBMIT] Generating raw data folder list via Python..."

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "[SUBMIT] conda.sh not found" >&2
  exit 1
fi
conda activate multiff_clean || { echo "[SUBMIT] Conda activation failed"; exit 1; }

python <<EOF
import os
from pathlib import Path
import sys

# Ensure project root
project_root = Path('$PROJECT_ROOT')
os.chdir(project_root)
sys.path.insert(0, str(project_root / 'multiff_analysis/multiff_code/methods'))

from data_wrangling import combine_info_utils

raw_data_dir_name = 'all_monkey_data/raw_monkey_data'
monkey_name = '$MONKEY_NAME'

sessions_df = combine_info_utils.make_sessions_df_for_one_monkey(
    raw_data_dir_name, monkey_name
)

paths = []
for _, row in sessions_df.iterrows():
    path = os.path.join(
        raw_data_dir_name,
        row['monkey_name'],
        row['data_name']
    )
    paths.append(os.path.abspath(path))

out_file = Path('$RAW_PATH_FILE')
out_file.write_text('\n'.join(paths) + '\n')

print(f'[SUBMIT] Wrote {len(paths)} raw data paths to {out_file}')
EOF

NUM_JOBS=$(wc -l < "$RAW_PATH_FILE")

if [ "$NUM_JOBS" -eq 0 ]; then
    echo "[SUBMIT] No sessions found for $MONKEY_NAME" >&2
    exit 1
fi

ARRAY_MAX=$((NUM_JOBS - 1))

echo "[SUBMIT] Submitting SLURM array: 0-$ARRAY_MAX"

# ------------------------------
# Submit job
# ------------------------------
MAX_CONCURRENT=5

sbatch \
    --array=0-"$ARRAY_MAX"%$MAX_CONCURRENT \
    "$JOB_DIR/slurm/glm_job.slurm" \
    "${FORWARD_ARGS[@]}"
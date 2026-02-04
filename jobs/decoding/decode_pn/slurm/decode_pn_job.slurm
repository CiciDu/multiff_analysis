#!/bin/bash
#SBATCH --job-name=decode_pn
#SBATCH --output=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decoding/decode_pn/logs/run_stdout/decode_pn_%A_%a.out
#SBATCH --error=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decoding/decode_pn/logs/run_stdout/decode_pn_%A_%a.err
#SBATCH --partition=cpu,xaqq
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ------------------------------
# Inputs
# ------------------------------
: "${1:?Usage: sbatch event_decoding_job.slurm MONKEY_NAME}"

MONKEY_NAME="$1"
shift

PROJECT_ROOT=/user_data/cicid/Multifirefly-Project
CONFIG_DIR=$PROJECT_ROOT/multiff_analysis/jobs/data_configs
RAW_PATH_FILE=$CONFIG_DIR/raw_data_paths_${MONKEY_NAME}.txt

# ------------------------------
# Validate raw data config
# ------------------------------
if [ ! -f "$RAW_PATH_FILE" ]; then
    echo "[SLURM][ERROR] Missing raw data config:"
    echo "  $RAW_PATH_FILE"
    echo "Did you generate it first?"
    exit 1
fi

if [ ! -s "$RAW_PATH_FILE" ]; then
    echo "[SLURM][ERROR] Raw data config is empty:"
    echo "  $RAW_PATH_FILE"
    exit 1
fi

# ------------------------------
# Paths
# ------------------------------
PROJECT_ROOT=/user_data/cicid/Multifirefly-Project
DATA_CONFIG_DIR=$PROJECT_ROOT/multiff_analysis/jobs/data_configs
RAW_PATH_FILE=$DATA_CONFIG_DIR/raw_data_paths_${MONKEY_NAME}.txt

cd "$PROJECT_ROOT"

# ------------------------------
# Conda
# ------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "[SLURM] conda.sh not found" >&2
    exit 1
fi
conda activate multiff_clean || { echo "[SLURM] Conda activation failed"; exit 1; }

# ------------------------------
# Determine mode
# ------------------------------
RAW_DATA_FOLDER_PATH="${RAW_DATA_FOLDER_PATH:-}"

if [ -n "$RAW_DATA_FOLDER_PATH" ]; then
    # ==============================
    # Single-job mode
    # ==============================
    echo "[SLURM] Single-session mode"
    echo "[SLURM] Monkey: $MONKEY_NAME"
    echo "[SLURM] Processing: $RAW_DATA_FOLDER_PATH"

else
    # ==============================
    # Array-job mode
    # ==============================
    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

    if [ ! -f "$RAW_PATH_FILE" ]; then
        echo "[SLURM] RAW_PATH_FILE not found: $RAW_PATH_FILE" >&2
        exit 1
    fi

    RAW_DATA_FOLDER_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RAW_PATH_FILE")

    if [ -z "$RAW_DATA_FOLDER_PATH" ]; then
        echo "[SLURM] No raw data path for index $SLURM_ARRAY_TASK_ID" >&2
        exit 1
    fi


    echo "[SLURM] Monkey: $MONKEY_NAME"
    echo "[SLURM] Processing: $RAW_DATA_FOLDER_PATH"
fi

# ------------------------------
# Run decode_pn
# ------------------------------
python -u multiff_analysis/jobs/decoding/decode_pn/scripts/decode_pn_script.py \
    --raw_data_folder_path "$RAW_DATA_FOLDER_PATH" \

echo "[SLURM] Done"

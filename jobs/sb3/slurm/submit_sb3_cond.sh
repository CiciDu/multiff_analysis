#!/bin/bash
set -euo pipefail

# Usage examples:
#   ./submit_sb3_cond.sh perc_r
#   ./submit_sb3_cond.sh mem_r mem_th
#   ./submit_sb3_cond.sh all

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <perc_r|perc_th|mem_r|mem_th|all> [additional params...]"
    exit 1
fi

PARAMS=(perc_r perc_th mem_r mem_th)

PROJECT_ROOT="/user_data/cicid/Multifirefly-Project"
LOG_DIR="$PROJECT_ROOT/multiff_analysis/jobs/sb3/logs/cond"

# -------------------------------------------------
# Ensure log directory exists BEFORE sbatch runs
# -------------------------------------------------
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_MAX=500
MAX_CONCURRENT=10

submit_one () {
    local P="$1"
    echo "[submit] SWEEP_PARAM=$P"
    SWEEP_PARAM="$P" sbatch --array=0-"$ARRAY_MAX"%$MAX_CONCURRENT "$SCRIPT_DIR/sb3_cond_job.slurm"
}

# -------------------------------------------------
# Handle arguments
# -------------------------------------------------

if [[ "$1" == "all" ]]; then
    for p in "${PARAMS[@]}"; do
        submit_one "$p"
    done
    exit 0
fi

# Otherwise: allow multiple specific params
for ARG in "$@"; do
    case "$ARG" in
        perc_r|perc_th|mem_r|mem_th)
            submit_one "$ARG"
            ;;
        *)
            echo "Invalid argument: $ARG"
            echo "Valid options: perc_r perc_th mem_r mem_th all"
            exit 1
            ;;
    esac
done
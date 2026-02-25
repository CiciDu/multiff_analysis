#!/bin/bash
set -euo pipefail

# Usage:
#   ./submit_sb3_cond.sh perc_r
#   ./submit_sb3_cond.sh all

ARG="${1:?Usage: $0 <perc_r|perc_th|mem_r|mem_th|all>}"

PARAMS=(perc_r perc_th mem_r mem_th)

PROJECT_ROOT="/user_data/cicid/Multifirefly-Project"
LOG_DIR="$PROJECT_ROOT/multiff_analysis/jobs/sb3/logs/cond"

# -------------------------------------------------
# Ensure log directory exists BEFORE sbatch runs
# -------------------------------------------------
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_MAX=100
MAX_CONCURRENT=3

submit_one () {
    local P="$1"
    echo "[submit] SWEEP_PARAM=$P"
    SWEEP_PARAM="$P" sbatch --array=0-"$ARRAY_MAX"%$MAX_CONCURRENT "$SCRIPT_DIR/sb3_cond_job.slurm"
}

if [[ "$ARG" == "all" ]]; then
    for p in "${PARAMS[@]}"; do
        submit_one "$p"
    done
else
    case "$ARG" in
        perc_r|perc_th|mem_r|mem_th)
            submit_one "$ARG"
            ;;
        *)
            echo "Invalid argument: $ARG"
            echo "Usage: $0 <perc_r|perc_th|mem_r|mem_th|all>"
            exit 1
            ;;
    esac
fi
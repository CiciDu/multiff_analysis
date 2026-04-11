#!/bin/bash
set -euo pipefail


# Usage examples:
#   ./submit_sb3_cond.sh perc_r
#   ./submit_sb3_cond.sh mem_r mem_th
#   ./submit_sb3_cond.sh all

# Pre-create SLURM log dir before submission (SLURM writes logs itself)
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/sb3_cond


if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <perc_r|perc_th|mem_r|mem_th|all> [additional params...]"
    exit 1
fi

PARAMS=(perc_r perc_th mem_r mem_th)

PROJECT_ROOT="/user_data/cicid/Multifirefly-Project"
LOG_DIR="$PROJECT_ROOT/multiff_analysis/jobs/sb3/logs/cond"

mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same as submit_sb3.sh: at most ARRAY_MAX tasks per array job, % throttling
ARRAY_MAX=1000
MAX_CONCURRENT=10

# Must stay in sync with sb3_cond_job.slurm: |LEVELS|×|BASE_SEEDS|×NUM_REPEATS
N_LEVELS=7
N_BASE_SEEDS=15
NUM_REPEATS=5
TOTAL_COMB=$((N_LEVELS * N_BASE_SEEDS * NUM_REPEATS))

submit_one () {
    local P="$1"
    local offset=0
    while (( offset < TOTAL_COMB )); do
        local remaining=$((TOTAL_COMB - offset))
        local chunk last
        if (( remaining > ARRAY_MAX )); then
            chunk=$ARRAY_MAX
        else
            chunk=$remaining
        fi
        last=$((chunk - 1))

        echo "[submit] SWEEP_PARAM=$P SLURM_ARRAY_OFFSET=${offset} array: 0-${last} (${chunk} tasks, max ${MAX_CONCURRENT} concurrent)"
        sbatch \
            --array=0-"${last}"%"${MAX_CONCURRENT}" \
            --export=ALL,SWEEP_PARAM="${P}",SLURM_ARRAY_OFFSET="${offset}" \
            "$SCRIPT_DIR/sb3_cond_job.slurm"
        offset=$((offset + chunk))
    done
}

if [[ "$1" == "all" ]]; then
    for p in "${PARAMS[@]}"; do
        submit_one "$p"
    done
    exit 0
fi

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

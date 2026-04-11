#!/bin/bash
set -euo pipefail

# Pre-create SLURM log dir before submission (SLURM writes logs itself)
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/sb3

# Same knobs as submit_sb3.sh: at most ARRAY_MAX tasks per array job, % throttling
ARRAY_MAX=1000
MAX_CONCURRENT=10

# Must stay in sync with sb3_job.slurm: NUM_LEVELS × |BASE_SEEDS| × NUM_REPEATS
# NUM_LEVELS = |LRS|×|NUM_OBS_FF|×|MAX_IN_MEMORY|×|STRAT| = 1×7×2×4 = 56
N_LEVELS=56
N_BASE_SEEDS=4
NUM_REPEATS=5
TOTAL_COMB=$((N_LEVELS * N_BASE_SEEDS * NUM_REPEATS))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_FILE="$SCRIPT_DIR/sb3_job.slurm"

offset=0
while (( offset < TOTAL_COMB )); do
  remaining=$((TOTAL_COMB - offset))
  if (( remaining > ARRAY_MAX )); then
    chunk=$ARRAY_MAX
  else
    chunk=$remaining
  fi
  last=$((chunk - 1))

  echo "[SUBMIT] SLURM_ARRAY_OFFSET=${offset} array: 0-${last} (${chunk} tasks, max ${MAX_CONCURRENT} concurrent); global TASK_ID ${offset}-$((offset + last)) / $((TOTAL_COMB - 1))"
  sbatch \
    --array=0-"${last}"%"${MAX_CONCURRENT}" \
    --export=ALL,SLURM_ARRAY_OFFSET="${offset}" \
    "$SLURM_FILE" "$@"

  offset=$((offset + chunk))
done

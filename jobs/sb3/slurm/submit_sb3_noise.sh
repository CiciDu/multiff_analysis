#!/bin/bash
set -euo pipefail

# Pre-create SLURM log dir before submission (SLURM writes logs itself)
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/sb3/logs/sb3_noise


# Same as submit_sb3.sh: at most ARRAY_MAX tasks per array job, % throttling
ARRAY_MAX=1000
MAX_CONCURRENT=10

# Must stay in sync with sb3_noise_job.slurm: |MEM_R_LIST|×|MEM_TH_LIST|×|BASE_SEEDS|×NUM_REPEATS
N_MEM_R=3
N_MEM_TH=9
N_BASE_SEEDS=8
NUM_REPEATS=5
TOTAL_COMB=$((N_MEM_R * N_MEM_TH * N_BASE_SEEDS * NUM_REPEATS))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_FILE="$SCRIPT_DIR/sb3_noise_job.slurm"

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

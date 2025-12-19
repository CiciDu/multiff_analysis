#!/bin/bash
#SBATCH -J rl_collect_batches
#SBATCH --partition=gpu,xaqq
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH -t 24:00:00
#SBATCH --array=1-10
#SBATCH -o /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rl_collect/logs/run_stdout/rl_collect_batch-%A_%a.out
#SBATCH -e /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/rl_collect/logs/run_stderr/rl_collect_batch-%A_%a.err

set -euo pipefail

cd /user_data/cicid/Multifirefly-Project

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiff_clean

# Batch json path
BATCH_DIR="multiff_analysis/RL_models/meta/directory_of_agents/batches"
AGENT_JSON="${BATCH_DIR}/dir_batch_${SLURM_ARRAY_TASK_ID}.json"

echo "[jobinfo] Processing batch: $AGENT_JSON"

srun python multiff_analysis/jobs/rl_collect/scripts/rl_collect_script.py \
    --agent-directory "$AGENT_JSON" \
    --log-dir multiff_analysis/RL_models/meta/pipeline_logs \
    --num-datasets-to-collect 3 \
    --num-steps-per-dataset 9000 \
    >> "$SLURM_STDOUT" 2>> "$SLURM_STDERR"

echo "[jobinfo] Finished batch ${SLURM_ARRAY_TASK_ID}"

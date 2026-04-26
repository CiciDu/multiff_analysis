#!/usr/bin/env bash
#SBATCH --job-name=agent_retry
#SBATCH --output=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/agent_data/logs/agent_retry/agent_retry_%A_%a.out
#SBATCH --error=/user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/agent_data/logs/agent_retry/agent_retry_%A_%a.err
#SBATCH --partition=cpu,xaqq
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=mind-1-29,mind-1-15

# --------------------------------------------------
# Shell safety
# --------------------------------------------------
set -e
set -o pipefail

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT='/user_data/cicid/Multifirefly-Project'
SCRIPT_PATH="$PROJECT_ROOT/multiff_analysis/jobs/agent_data/scripts/agent_retry_script.py"

cd "$PROJECT_ROOT"

# Early status print (helps diagnose delays)
echo "[SLURM][START] Job started at $(date), Array task: ${SLURM_ARRAY_TASK_ID:-N/A}" >&2

# --------------------------------------------------
# Conda setup
# --------------------------------------------------
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

if [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
else
  echo '[SLURM][ERROR] conda.sh not found' >&2
  exit 1
fi

conda activate multiff_clean || {
  echo '[SLURM][ERROR] Conda activation failed' >&2
  exit 1
}

echo "[SLURM][CONDA] Environment activated at $(date)" >&2

# --------------------------------------------------
# Now enable nounset
# --------------------------------------------------
set -u

# --------------------------------------------------
# Determine agent folder from manifest
# --------------------------------------------------
# SLURM_ARRAY_OFFSET is set by submit_agent_retry.sh when chunking (keeps array IDs within MaxArraySize).
SLURM_ARRAY_OFFSET="${SLURM_ARRAY_OFFSET:-0}"
AGENT_INDEX=$((SLURM_ARRAY_OFFSET + SLURM_ARRAY_TASK_ID))

# Use MANIFEST_PATH from environment if provided, otherwise use default
MANIFEST_PATH="${MANIFEST_PATH:-$PROJECT_ROOT/RL_models/sb3_stored_models/all_agents/agent_folder_manifest.json}"
AGENT_FOLDER=$(python -c "import json; manifest = json.load(open('$MANIFEST_PATH')); print(manifest[$AGENT_INDEX] if $AGENT_INDEX < len(manifest) else 'INDEX_OUT_OF_RANGE')")

if [[ "$AGENT_FOLDER" == "INDEX_OUT_OF_RANGE" ]]; then
  echo "[SLURM][ERROR] AGENT_INDEX=$AGENT_INDEX (task=$SLURM_ARRAY_TASK_ID offset=$SLURM_ARRAY_OFFSET) is out of range for manifest: $MANIFEST_PATH" >&2
  exit 1
fi

echo '[SLURM] Running Agent Retry Job'
echo "[SLURM] Host: $(hostname)"
echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Array task ID: $SLURM_ARRAY_TASK_ID (offset $SLURM_ARRAY_OFFSET)"
echo "[SLURM] Agent index: $AGENT_INDEX"
echo "[SLURM] Manifest path: $MANIFEST_PATH"
echo "[SLURM] Agent path: $AGENT_FOLDER"
echo "[SLURM] Start time: $(date)"
echo "[SLURM] Command: python -u $SCRIPT_PATH --agent-path \"$AGENT_FOLDER\""

# Force flush to log files
exec 1>&1 2>&2

# --------------------------------------------------
# Run agent retry analysis
# --------------------------------------------------
stdbuf -oL -eL python -u "$SCRIPT_PATH" --agent-path "$AGENT_FOLDER"

echo '[SLURM] Done'
echo "[SLURM] End time: $(date)"
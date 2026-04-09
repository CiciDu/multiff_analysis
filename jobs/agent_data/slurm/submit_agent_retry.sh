#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] Failed at line $LINENO" >&2' ERR

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT='/user_data/cicid/Multifirefly-Project'
RL_AGENT_DIR="$PROJECT_ROOT/RL_models/sb3_stored_models/all_agents/agents_with_noise"


JOB_DIR="$PROJECT_ROOT/multiff_analysis/jobs/agent_data"
SLURM_SCRIPT="$JOB_DIR/slurm/agent_retry_job.slurm"
SCRIPT_DIR="$JOB_DIR/scripts"

MANIFEST_SCRIPT="$SCRIPT_DIR/build_agent_manifest.py"
BUILD_MANIFEST_ARGS="--overall-folder $RL_AGENT_DIR"
MANIFEST_PATH="$RL_AGENT_DIR/agent_folder_manifest.json"

LOG_DIR="$JOB_DIR/logs"
mkdir -p "$LOG_DIR"

# --------------------------------------------------
# Usage
# --------------------------------------------------
usage() {
  cat <<EOF
Usage:
  submit_agent_retry.sh [options]

Options (handled by this script):
  --max-parallel N  Max concurrent jobs (default: 20)

All other arguments are forwarded directly to sbatch.

Examples:
  submit_agent_retry.sh
  submit_agent_retry.sh --max-parallel 8
  submit_agent_retry.sh -p cpu --time=24:00:00
EOF
}

# --------------------------------------------------
# Defaults
# --------------------------------------------------
MAX_PARALLEL=20

# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
FORWARD_ARGS=()
while (( "$#" )); do
  case "$1" in
    --max-parallel)
      [ -z "${2:-}" ] && { echo "--max-parallel requires an argument"; exit 1; }
      MAX_PARALLEL="$2"
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

# --------------------------------------------------
# Preflight checks
# --------------------------------------------------
command -v sbatch >/dev/null 2>&1 || {
  echo '[ERROR] sbatch not found in PATH' >&2
  exit 1
}

[[ -r "$SLURM_SCRIPT" ]] || {
  echo "[ERROR] Slurm script not found: $SLURM_SCRIPT" >&2
  exit 1
}

[[ -r "$MANIFEST_SCRIPT" ]] || {
  echo "[ERROR] Manifest script not found: $MANIFEST_SCRIPT" >&2
  exit 1
}

# --------------------------------------------------
# Build agent manifest
# --------------------------------------------------
echo '[INFO] Building agent manifest...'

cd "$PROJECT_ROOT" || { echo "[ERROR] Failed to cd to $PROJECT_ROOT" >&2; exit 1; }
python "$MANIFEST_SCRIPT" $BUILD_MANIFEST_ARGS

[[ -f "$MANIFEST_PATH" ]] || {
  echo "[ERROR] Manifest file was not created: $MANIFEST_PATH"
  exit 1
}

# --------------------------------------------------
# Determine number of agents
# --------------------------------------------------
NUM_AGENTS=$(python - <<EOF
import json
with open("$MANIFEST_PATH") as f:
    data = json.load(f)
print(len(data))
EOF
)

echo "[INFO] Found $NUM_AGENTS agents"

if [[ "$NUM_AGENTS" -eq 0 ]]; then
  echo "[ERROR] No agents found in manifest"
  exit 1
fi

if [[ "$NUM_AGENTS" -gt 999 ]]; then
  echo "[WARN] Number of agents ($NUM_AGENTS) exceeds maximum supported (999); capping to 999"
  NUM_AGENTS=999
fi

ARRAY_RANGE="0-$((NUM_AGENTS-1))"

echo "[INFO] Job array range: $ARRAY_RANGE"
echo "[INFO] Max concurrent jobs: $MAX_PARALLEL"
echo "[INFO] Each job will run: python agent_retry_script.py --agent-path <AGENT_FOLDER>"
echo "[INFO] Agent indices: $(seq -s ', ' 0 $((NUM_AGENTS-1)))"

# --------------------------------------------------
# Submit job array
# --------------------------------------------------
echo '[SUBMIT] Running sbatch:'
echo "sbatch --array=\"$ARRAY_RANGE%$MAX_PARALLEL\" ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"} \"$SLURM_SCRIPT\""

sbatch \
  --array="${ARRAY_RANGE}%${MAX_PARALLEL}" \
  --export=ALL,MANIFEST_PATH="$MANIFEST_PATH" \
  ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"} \
  "$SLURM_SCRIPT"
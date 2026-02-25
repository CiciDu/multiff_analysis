#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# Error trap (no silent failures)
# --------------------------------------------------
trap 'echo "[ERROR] Failed at line $LINENO" >&2' ERR

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT='/user_data/cicid/Multifirefly-Project'
JOB_DIR="$PROJECT_ROOT/multiff_analysis/jobs/one_ff_decoding"
SLURM_SCRIPT="$JOB_DIR/slurm/one_ff_decoding_job.slurm"

LOG_DIR="$JOB_DIR/logs/decoding"
mkdir -p "$LOG_DIR"

# --------------------------------------------------
# Usage
# --------------------------------------------------
usage() {
  cat <<EOF
Usage:
  submit_one_ff_decoding.sh [options]

All arguments are forwarded directly to sbatch.
EOF
}

# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
FORWARD_ARGS=()
while (( "$#" )); do
  case "$1" in
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
# Preflight checks (important)
# --------------------------------------------------
command -v sbatch >/dev/null 2>&1 || {
  echo '[ERROR] sbatch not found in PATH' >&2
  exit 1
}

[[ -r "$SLURM_SCRIPT" ]] || {
  echo "[ERROR] Slurm script not found or unreadable: $SLURM_SCRIPT" >&2
  exit 1
}

# Detect Windows CRLF line endings (Slurm will reject them)
if file "$SLURM_SCRIPT" | grep -q CRLF; then
  echo "[ERROR] Slurm script has Windows (CRLF) line endings:" >&2
  echo "        $SLURM_SCRIPT" >&2
  echo "        Fix with: dos2unix $SLURM_SCRIPT" >&2
  exit 1
fi

# --------------------------------------------------
# Info
# --------------------------------------------------
echo '[SUBMIT] One-FF Decoding'
echo "[SUBMIT] Slurm script: $SLURM_SCRIPT"

# --------------------------------------------------
# Submit job (echo first so nothing is hidden)
# --------------------------------------------------
echo '[SUBMIT] Running sbatch:'
echo sbatch "${FORWARD_ARGS[@]}" "$SLURM_SCRIPT"

sbatch \
  "${FORWARD_ARGS[@]}" \
  "$SLURM_SCRIPT"
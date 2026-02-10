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
JOB_DIR="$PROJECT_ROOT/multiff_analysis/jobs/one_ff"
SLURM_SCRIPT="$JOB_DIR/slurm/one_ff_my_gam_job.slurm"

LOG_DIR="$JOB_DIR/logs/my_gam"
mkdir -p "$LOG_DIR"

# --------------------------------------------------
# Usage
# --------------------------------------------------
usage() {
  cat <<EOF
Usage:
  submit_one_ff_my_gam.sh [options]

Options handled by this script:
  --units N         Number of units (default: 128)
  --max-parallel N  Max concurrent jobs (default: 3)

All other arguments are forwarded directly to sbatch.
EOF
}

# --------------------------------------------------
# Defaults
# --------------------------------------------------
N_UNITS=128
MAX_PARALLEL=3
FORWARD_ARGS=()

# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
while (( "$#" )); do
  case "$1" in
    --units)
      [[ -z "${2:-}" ]] && { echo '[ERROR] --units requires an argument' >&2; exit 1; }
      N_UNITS="$2"
      shift 2
      ;;
    --max-parallel)
      [[ -z "${2:-}" ]] && { echo '[ERROR] --max-parallel requires an argument' >&2; exit 1; }
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
# Validate inputs
# --------------------------------------------------
[[ "$N_UNITS" =~ ^[0-9]+$ ]] || { echo '[ERROR] --units must be an integer' >&2; exit 1; }
[[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || { echo '[ERROR] --max-parallel must be an integer' >&2; exit 1; }

ARRAY_MAX=$((N_UNITS - 1))
(( ARRAY_MAX >= 0 )) || { echo '[ERROR] --units must be >= 1' >&2; exit 1; }

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
echo '[SUBMIT] One-FF GAM MAP Fit'
echo "[SUBMIT] Units: $N_UNITS (0-$ARRAY_MAX)"
echo "[SUBMIT] Max concurrent: $MAX_PARALLEL"
echo "[SUBMIT] Slurm script: $SLURM_SCRIPT"

# --------------------------------------------------
# Submit job (echo first so nothing is hidden)
# --------------------------------------------------
echo '[SUBMIT] Running sbatch:'
echo sbatch --array=0-"$ARRAY_MAX"%"$MAX_PARALLEL" "${FORWARD_ARGS[@]}" "$SLURM_SCRIPT"

sbatch \
  --array=0-"$ARRAY_MAX"%"$MAX_PARALLEL" \
  "${FORWARD_ARGS[@]}" \
  "$SLURM_SCRIPT"
#!/bin/bash
set -euo pipefail

# Pre-create SLURM log dirs before submission (SLURM writes logs itself)
mkdir -p /user_data/cicid/Multifirefly-Project/multiff_analysis/jobs/decoding/cond_decoding/logs/run_stdout

usage() {
  cat <<'EOF'
Usage: submit_cond_decode.sh [--models "svm logreg"] [--cumulative] [sbatch args...]

Options:
  --models "LIST"     Space- or comma-delimited model list to try (e.g., "svm,logreg").
  --cumulative        Enable cumulative mode (exports CUMULATIVE=1).
  -h, --help          Show this help and exit.

All other arguments are forwarded directly to sbatch (e.g., -p cpu --array=0-9).
Note: If --models is omitted, the SLURM job script's defaults are used.
EOF
}

# Parse flags and collect sbatch args
MODELS="${MODELS:-}"
CUMULATIVE_FLAG=0
FORWARD_ARGS=()
while (( "$#" )); do
  case "${1}" in
    -h|--help)
      usage
      exit 0
      ;;
    --models)
      if [ -n "${2-}" ]; then
        MODELS="$2"
        shift 2
        continue
      else
        echo "Error: --models requires a value" >&2
        exit 1
      fi
      ;;
    --cumulative)
      CUMULATIVE_FLAG=1
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_FILE="$SCRIPT_DIR/../slurm/cond_decoding_job.slurm"

if [ ! -f "$JOB_FILE" ]; then
  echo "Error: SLURM job file not found at $JOB_FILE" >&2
  exit 1
fi

# Build --export line
EXPORT_VARS="ALL"
if [ -n "${MODELS:-}" ]; then
  EXPORT_VARS="${EXPORT_VARS},MODELS=${MODELS}"
fi
if [ "$CUMULATIVE_FLAG" = "1" ]; then
  EXPORT_VARS="${EXPORT_VARS},CUMULATIVE=1"
fi

sbatch --export="$EXPORT_VARS" "$JOB_FILE" "${FORWARD_ARGS[@]}"
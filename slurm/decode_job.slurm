#!/bin/bash
#SBATCH --job-name=decode
#SBATCH --output=/user_data/cicid/Multifirefly-Project/multiff_analysis/logs_decode/decode_%A_%a.out
#SBATCH --error=/user_data/cicid/Multifirefly-Project/multiff_analysis/logs_decode/decode_%A_%a.err
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd /user_data/cicid/Multifirefly-Project || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multiff_clean

mkdir -p multiff_analysis/logs_decode
export PYTHONUNBUFFERED=1

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID not set."
    exit 1
fi

N_JOBS=${SLURM_CPUS_PER_TASK}
RUN_ID=$(date +%Y%m%d_%H%M)
echo "[SLURM] Job $SLURM_JOB_ID | Array index $SLURM_ARRAY_TASK_ID"
echo "[SLURM] Using $N_JOBS CPU cores"

MODELS=("svm" "logreg" "logreg_elasticnet" "rf" "mlp")

for MODEL in "${MODELS[@]}"; do
    case "$MODEL" in
        svm) KWARGS='{"C":2.0,"gamma":0.05}' ;;
        logreg|logreg_elasticnet|rf|mlp) KWARGS='{}' ;;
        *) KWARGS='{}' ;;
    esac

    echo "[SLURM] Running model $MODEL for comparison index $SLURM_ARRAY_TASK_ID"
    python -u multiff_analysis/scripts/decode_script.py \
        --comparisons multiff_analysis/configs/comparisons.json \
        --idx $SLURM_ARRAY_TASK_ID \
        --model "$MODEL" \
        --model_kwargs "$KWARGS" \
        --n_jobs $N_JOBS \
        --n_perm 1000 \
        --do_testing \
        --tune
done

echo "[SLURM] Job $SLURM_JOB_ID completed for index $SLURM_ARRAY_TASK_ID"

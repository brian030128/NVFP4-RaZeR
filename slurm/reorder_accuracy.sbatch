#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=reorder_accuracy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL=qwen27b
source slurm/task_reorder_env.sh
OUT="$PWD/results/task_reorder/accuracy_20260920"
"$PYTHON" tests/test_reorder_accuracy.py
"$PYTHON" -m py_compile run_reorder_accuracy.py run_taskfit_diagnosis.py
EXTRA=()
if [[ "${ACCURACY_SMOKE:-0}" == 1 ]]; then
    EXTRA+=(--smoke)
    DESTINATION="$OUT/smoke"
else
    DESTINATION="$OUT/shards/$SLURM_ARRAY_TASK_ID"
fi
"$PYTHON" run_reorder_accuracy.py --root /work/u4320956/task_reorder/pilot_20260919/qwen27b \
    --out "$DESTINATION" --plan "$OUT/plan.json" --shard "$SLURM_ARRAY_TASK_ID" "${EXTRA[@]}"

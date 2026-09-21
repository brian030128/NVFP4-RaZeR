#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=llama_diagnosis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --array=0-3%4
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL=llama8b
source slurm/task_reorder_env.sh
EXTRA=()
if [[ "${DIAGNOSIS_SMOKE:-0}" == 1 ]]; then EXTRA+=(--smoke); fi
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
    "$PYTHON" run_llama_taskfit_diagnosis.py "${EXTRA[@]}"
else
    LAYERS=(0 0 15 31)
    "$PYTHON" run_llama_depth_diagnosis.py --layer "${LAYERS[$SLURM_ARRAY_TASK_ID]}" "${EXTRA[@]}"
fi

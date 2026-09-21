#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=llama_acc_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL=llama8b
source slurm/task_reorder_env.sh
UV="$PWD/.joblib/bootstrap/bin/uv"
export UV_CACHE_DIR="$PWD/.joblib/uv-cache"
"$UV" pip freeze --python "$PYTHON" > "$TMPDIR/accuracy_constraints.txt"
"$UV" pip install --python "$PYTHON" --constraint "$TMPDIR/accuracy_constraints.txt" 'lm_eval==0.4.5'
"$PYTHON" -c 'import lm_eval_compat; import lm_eval; import torch, transformers; from lm_eval.models.huggingface import HFLM; print("Accuracy imports ready", torch.__version__, transformers.__version__)'

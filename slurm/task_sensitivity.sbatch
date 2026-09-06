#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=task_sensitivity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
uv pip install --python "$PYTHON" --target "$HF_HOME/joblib" loguru
export PYTHONPATH="$PWD:$HF_HOME/joblib:${PYTHONPATH:-}"
"$PYTHON" tests/test_task_sensitivity.py
MODEL=${1:?model name required}
shift
"$PYTHON" analyze_task_sensitivity.py --model "$MODEL" \
  --out "results/task_sensitivity/$MODEL" "$@"

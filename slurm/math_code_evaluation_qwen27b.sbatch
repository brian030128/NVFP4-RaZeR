#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=math_code_eval_27b
#SBATCH --array=0-2%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:H100:2
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
export UV_CACHE_DIR="$HF_HOME/uv-cache"
export TMPDIR="$HF_HOME/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
uv pip install --python "$PYTHON" --target "$HF_HOME/joblib" --no-deps \
  'transformers==5.16.1' huggingface_hub tokenizers safetensors hf-xet loguru
export PYTHONPATH="$PWD:$HF_HOME/joblib:${PYTHONPATH:-}"
GROUPS_TO_RUN=(c4 wiki_adaptive wiki_fixed)
GROUP="${GROUPS_TO_RUN[$SLURM_ARRAY_TASK_ID]}"
"$PYTHON" run_math_code_evaluation.py --group "$GROUP" \
  --calibration "results/math_code_adaptive/calibration_${1:?calibration job ID}_qwen27b" \
  --out "results/math_code_adaptive/evaluation_${SLURM_ARRAY_JOB_ID}_qwen27b_$GROUP"

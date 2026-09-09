#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=calibration_qwen27b
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
"$PYTHON" tests/test_calibration_sensitivity.py
"$PYTHON" tests/test_causal_four_over_six.py
case "$SLURM_ARRAY_TASK_ID" in
  0) DOMAINS=(c4); SUFFIX=c4 ;;
  1) DOMAINS=(wiki); SUFFIX=wiki ;;
  2) DOMAINS=(literature science government); SUFFIX=transfer ;;
esac
"$PYTHON" run_calibration_sensitivity.py \
  --prepared "results/calibration_sensitivity/prepared_${1:?preparation job ID}_qwen27b" \
  --domains "${DOMAINS[@]}" \
  --out "results/calibration_sensitivity/model_${SLURM_ARRAY_JOB_ID}_qwen27b_$SUFFIX"

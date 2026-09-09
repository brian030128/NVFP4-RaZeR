#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=cap_sweep
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=03:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python

"$PYTHON" tests/test_cap_sweep.py
"$PYTHON" tests/test_causal_four_over_six.py

MODELS=(olmo1b pythia14b qwen4b llama8b)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
"$PYTHON" run_cap_sweep.py --model "$MODEL" \
  --stage-root /home/u4320956/NVFP4-RaZeR \
  --out "results/cap_sweep/model_${SLURM_ARRAY_JOB_ID}_${MODEL}"

echo "=== done $(date -Iseconds) ==="

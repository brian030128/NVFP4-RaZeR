#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=baseline_protocol_audit
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --mem=100G
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
MODELS=(qwen4b llama8b qwen4b llama8b)
LENGTHS=(512 512 2048 2048)
/home/u4320956/beam_engine-v3/.venv/bin/python run_baseline_protocol_audit.py \
  --model "${MODELS[$SLURM_ARRAY_TASK_ID]}" --length "${LENGTHS[$SLURM_ARRAY_TASK_ID]}" \
  --out "results/baseline_protocol_audit/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" "$@"

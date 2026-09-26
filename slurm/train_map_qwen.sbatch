#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --job-name=train_map_qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=190G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Usage: sbatch slurm/train_map_qwen.sbatch OUT [run_train_map.py args...]
# Qwen's batched forward is not identical to batch 1, so each optimizer step
# accumulates 8 single-sequence passes (same 16 steps/epoch as Llama at batch 8).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/qwen27b/bin/python MODEL=qwen27b
source slurm/task_reorder_env.sh

OUT="${1:?out}"; shift
"$PYTHON" run_train_map.py --model qwen27b --gpus 1 --batch 1 --accum 8 --eval-batch 1 --out "$OUT" "$@"

#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=qbatchdiag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Usage: sbatch slurm/qwen_batch_diag.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/qwen27b/bin/python MODEL=qwen27b
source slurm/task_reorder_env.sh

"$PYTHON" diagnose_qwen_batching.py

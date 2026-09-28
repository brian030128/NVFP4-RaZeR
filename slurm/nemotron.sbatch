#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=nemotron
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=190G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Usage: sbatch slurm/nemotron.sbatch SCRIPT.py [args...]
# NVIDIA-Nemotron-Nano-9B-v2 jobs. Model weights go to a job-local Hub cache in /tmp
# ($HOME has no room); datasets keep using the shared HF_HOME.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/qwen27b/bin/python MODEL=qwen27b
source slurm/task_reorder_env.sh
export HF_HUB_CACHE="/tmp/$USER/reorder_$SLURM_JOB_ID/hub"
mkdir -p "$HF_HUB_CACHE"
trap 'rm -rf "/tmp/$USER/reorder_$SLURM_JOB_ID"' EXIT

"$PYTHON" "$@"

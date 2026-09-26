#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=e0m3_headroom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Usage: sbatch slurm/e0m3_headroom.sbatch OUT_JSON [analyze_e0m3_headroom.py args...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/llama8b/bin/python MODEL=llama8b
source slurm/task_reorder_env.sh

OUT="${1:?out}"; shift
"$PYTHON" analyze_e0m3_headroom.py --out "$OUT" "$@"

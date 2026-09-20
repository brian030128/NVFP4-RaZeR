#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=llama_fresh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL=llama8b
source slurm/task_reorder_env.sh
ROOT=/work/u4320956/task_reorder/transfer_20260920/llama8b
"$PYTHON" run_llama_fresh_candidate.py "$ROOT" "$ROOT/${1:?layouts}" "$ROOT/${2:?out}"

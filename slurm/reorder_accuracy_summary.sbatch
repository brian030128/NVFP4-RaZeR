#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=accuracy_summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL=qwen27b
source slurm/task_reorder_env.sh
"$PYTHON" tests/test_reorder_accuracy.py
"$PYTHON" summarize_reorder_accuracy.py results/task_reorder/accuracy_20260920

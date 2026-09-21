#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=reorder_capacity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Bracket how much any legal reordering could be worth, before searching for one.
# CPU-only; this cluster has no CPU-only allocation so one GPU is left idle.
#
#   sbatch slurm/reorder_capacity.sbatch SCORES_DIR OUT_DIR
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL="${REORDER_MODEL:-llama8b}"
source slurm/task_reorder_env.sh
"$PYTHON" run_reorder_capacity.py \
  --scores "${1:?Usage: sbatch ... SCORES_DIR OUT_DIR}" \
  --out "${2:?Usage: sbatch ... SCORES_DIR OUT_DIR}" \
  --threads "${SLURM_CPUS_PER_TASK:-8}"

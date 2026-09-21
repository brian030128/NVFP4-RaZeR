#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=reorder_objective_summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Aggregate ablation reports on a worker. Walking a result tree and loading
# every report is not login-node work, even though it needs no GPU.
#
#   sbatch slurm/reorder_objective_summary.sbatch ROOT [OUT_JSON]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL="${REORDER_MODEL:-llama8b}"
source slurm/task_reorder_env.sh

ROOT="${1:?Usage: sbatch ... ROOT [OUT_JSON]}"
OUT="${2:-}"
if [ -n "$OUT" ]; then
  "$PYTHON" summarize_reorder_objective_ablation.py --root "$ROOT" --out "$OUT"
else
  "$PYTHON" summarize_reorder_objective_ablation.py --root "$ROOT"
fi

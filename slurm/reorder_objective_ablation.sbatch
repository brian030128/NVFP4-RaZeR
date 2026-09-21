#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=reorder_objective_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
# Objective ablation for the 256x64 reordering search. CPU-only search; this
# cluster has no CPU-only allocation, so one GPU is requested and left idle.
#
#   sbatch --array=0-20 slurm/reorder_objective_ablation.sbatch SCORES_ROOT OUT_ROOT
#
# Array index = matrix * 7 + variant, over the three final-MLP projections
# (000 gate, 001 up, 002 down) and the seven objective variants. Indices 0-14
# under the old five-variant list are NOT the same runs; use the full range.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL="${REORDER_MODEL:-llama8b}"
source slurm/task_reorder_env.sh

SCORES_ROOT="${1:?Usage: sbatch ... SCORES_ROOT OUT_ROOT}"
OUT_ROOT="${2:?Usage: sbatch ... SCORES_ROOT OUT_ROOT}"
VARIANTS=(ce_kl kl ce shrunk placebo placebo_kl placebo_ce)
MATRICES=(000 001 002)

INDEX="${SLURM_ARRAY_TASK_ID:?Submit with --array}"
MATRIX="${MATRICES[$((INDEX / ${#VARIANTS[@]}))]}"
VARIANT="${VARIANTS[$((INDEX % ${#VARIANTS[@]}))]}"

echo "matrix=$MATRIX variant=$VARIANT scores=$SCORES_ROOT/$MATRIX"
"$PYTHON" run_reorder_objective_ablation.py \
  --scores "$SCORES_ROOT/$MATRIX" \
  --out "$OUT_ROOT/$MATRIX/$VARIANT" \
  --variant "$VARIANT" \
  --threads "${SLURM_CPUS_PER_TASK:-8}"

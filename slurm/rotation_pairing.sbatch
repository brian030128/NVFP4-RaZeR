#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=rotation_pairing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Pair every rotation arm of job 404773 on its own 64 development documents.
#
#   sbatch slurm/rotation_pairing.sbatch REPORT_JSON [OUT_JSON]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL="${REORDER_MODEL:-llama8b}"
source slurm/task_reorder_env.sh
OUT="${2:-}"
if [ -n "$OUT" ]; then
  "$PYTHON" analyze_rotation_pairing.py --report "${1:?}" --out "$OUT"
else
  "$PYTHON" analyze_rotation_pairing.py --report "${1:?}"
fi

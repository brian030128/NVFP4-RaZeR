#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=kl_candidate_replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Finite-loss replay of the step-1 KL-grouped candidate on development data.
#
#   sbatch slurm/kl_candidate_replay.sbatch LAYOUT_PT OUT_DIR [DOCUMENTS]
#
# Pass a small DOCUMENTS first as a smoke run: it exercises the capture, the
# bitwise suffix audit and the weight installation on a few documents before
# spending a full pass over all 192.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL=llama8b
source slurm/task_reorder_env.sh

LAYOUT="${1:?Usage: sbatch ... LAYOUT_PT OUT_DIR [DOCUMENTS]}"
OUT="${2:?Usage: sbatch ... LAYOUT_PT OUT_DIR [DOCUMENTS]}"
DOCUMENTS="${3:-}"

if [ -n "$DOCUMENTS" ]; then
  "$PYTHON" run_kl_candidate_replay.py --layout "$LAYOUT" --out "$OUT" --documents "$DOCUMENTS"
else
  "$PYTHON" run_kl_candidate_replay.py --layout "$LAYOUT" --out "$OUT"
fi

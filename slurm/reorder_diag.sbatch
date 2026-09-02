#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=reorder_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Diagnostics only: for EVERY linear layer of a model, decompose the E0M3 tag grid into row
# effects, column effects and residual. row+col is the fraction of the signal any row/column
# reordering scheme can systematically exploit; the residual is what only a per-scale-block
# choice can reach. No search runs, so this is cheap.
#
#   sbatch slurm/reorder_diag.sbatch llama-3.1-8b
#   CLIP=base sbatch slurm/reorder_diag.sbatch llama-3.1-8b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-llama-3.1-8b}
CLIP=${CLIP:-a1}

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/DiRotQ/.venv/bin/python}

# This account has no HF token; the models that matter are already in the legacy /work cache, so
# read it strictly offline -- no download, no new blobs on the quota'd filesystem.
export HF_HOME="/work/$USER/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_DIR="results/reorder"
mkdir -p "$OUT_DIR" slurm/logs

echo "=== diag $MODEL clip=$CLIP start $(date -Iseconds) on $(hostname) ==="

"$PYTHON" run_reorder_sim.py \
    --model_name "$MODEL" \
    --layer_stride 1 \
    --projections q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --diagnostics_only \
    --clip "$CLIP" \
    --threads 12 \
    --out "$OUT_DIR/diag_${MODEL}_${CLIP}.csv" 2>&1 | tee "$OUT_DIR/diag_${MODEL}_${CLIP}.log"

echo "=== done $(date -Iseconds) ==="

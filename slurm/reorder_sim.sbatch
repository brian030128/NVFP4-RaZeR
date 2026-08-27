#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=reorder_sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# CPU-only sweep: how much of the 1x16 E0M3 gain a row/column reordering gives back to a coarse
# type block, on real weights. The GPU is requested only because this cluster rations cores at
# 12 per GPU and has no CPU-only partition; it is left idle on purpose.
#
#   sbatch slurm/reorder_sim.sbatch llama-2-7b
#   sbatch slurm/reorder_sim.sbatch llama-3.1-8b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-llama-2-7b}
STRIDE=${2:-8}

# --- environment ---
source /etc/profile.d/lmod.sh 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/DiRotQ/.venv/bin/python}

# --- HF cache on node-local scratch, never /home or /work ---
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"

OUT_DIR="results/reorder"
mkdir -p "$OUT_DIR" slurm/logs

echo "=== $MODEL  stride=$STRIDE  start $(date -Iseconds)  on $(hostname) ==="

"$PYTHON" run_reorder_sim.py \
    --model_name "$MODEL" \
    --layer_stride "$STRIDE" \
    --projections q_proj,v_proj,o_proj,up_proj,down_proj \
    --type_blocks 8x64,32x128 \
    --rules argmin,h1.5 \
    --clip heade0 \
    --threads 12 \
    --out "$OUT_DIR/${MODEL}.csv" 2>&1 | tee -a "$OUT_DIR/${MODEL}.log"

echo "=== done $(date -Iseconds) ==="

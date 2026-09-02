#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=rotorder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Reordering before a BLOCK-DIAGONAL Hadamard. Against QuaRot's full-dimension rotation a
# permutation is absorbed and useless; against a rotate_size-wide chunked Hadamard it decides what
# each chunk gets to mix. CPU-only; the GPU is requested because cores are rationed 12 per GPU.
#
#   sbatch slurm/rotorder_sim.sbatch                  # synthetic first
#   sbatch slurm/rotorder_sim.sbatch llama-3.1-8b 8

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-}
STRIDE=${2:-8}
ROTS=${ROTS:-16,64,128}
MIN_GAIN=${MIN_GAIN:-0.1}
CLIP=${CLIP:-headx}
TAG=${TAG:-${MODEL:-synthetic}}

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/DiRotQ/.venv/bin/python}
export PYTHONUNBUFFERED=1

export HF_HOME="/work/$USER/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_DIR="results/rotorder"
mkdir -p "$OUT_DIR" slurm/logs

if [[ -z "$MODEL" ]]; then
    "$PYTHON" run_rotorder_sim.py --threads 12 --rotate_sizes "$ROTS" --min_gain "$MIN_GAIN" --clip "$CLIP" \
        --out "$OUT_DIR/${TAG}.csv" 2>&1 | tee "$OUT_DIR/${TAG}.log"
else
    "$PYTHON" run_rotorder_sim.py --model_name "$MODEL" --layer_stride "$STRIDE" --threads 12 \
        --rotate_sizes "$ROTS" --min_gain "$MIN_GAIN" --clip "$CLIP" \
        --out "$OUT_DIR/${TAG}.csv" 2>&1 | tee "$OUT_DIR/${TAG}.log"
fi

echo "=== done $(date -Iseconds) ==="

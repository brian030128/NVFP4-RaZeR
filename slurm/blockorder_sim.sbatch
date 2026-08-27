#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=blockorder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Column reordering that changes which elements share a 16-element scale block -- the lever the
# chunk-of-16 approach in reorder.py excluded by construction. CPU-only; the GPU is requested
# because cores are rationed at 12 per GPU and there is no CPU-only partition.
#
#   sbatch slurm/blockorder_sim.sbatch                 # synthetic controls first
#   sbatch slurm/blockorder_sim.sbatch llama-3.1-8b 8

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-}
STRIDE=${2:-8}
export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/DiRotQ/.venv/bin/python}
export PYTHONUNBUFFERED=1

export HF_HOME="/work/$USER/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_DIR="results/blockorder"
mkdir -p "$OUT_DIR" slurm/logs

if [[ -z "$MODEL" ]]; then
    echo "=== synthetic controls $(date -Iseconds) ==="
    "$PYTHON" run_blockorder_sim.py --threads 12 \
        --out "$OUT_DIR/synthetic.csv" 2>&1 | tee "$OUT_DIR/synthetic.log"
else
    echo "=== $MODEL stride=$STRIDE $(date -Iseconds) on $(hostname) ==="
    "$PYTHON" run_blockorder_sim.py --model_name "$MODEL" --layer_stride "$STRIDE" \
        --threads 12 --out "$OUT_DIR/${MODEL}.csv" 2>&1 | tee "$OUT_DIR/${MODEL}.log"
fi

echo "=== done $(date -Iseconds) ==="

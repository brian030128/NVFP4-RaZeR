#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=e0m3_choice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Which blocks want E0M3, how the importance-weighted criterion changes that, and whether the
# choice tracks outliers. Needs a GPU for the calibration forward passes that estimate E[x_j^2].
#
#   sbatch slurm/e0m3_choice.sbatch llama-3.1-8b-local
#   sbatch slurm/e0m3_choice.sbatch qwen3-4b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-llama-3.1-8b-local}
STRIDE=${2:-8}

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

LIBDIR="$PWD/.joblib"
mkdir -p "$LIBDIR"
[[ -d "$LIBDIR/loguru" ]] || uv pip install --quiet --target "$LIBDIR" loguru
export PYTHONPATH="$LIBDIR:${PYTHONPATH:-}"

export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT

OUT_DIR="results/e0m3_choice"
mkdir -p "$OUT_DIR" slurm/logs

echo "=== $MODEL stride=$STRIDE $(date -Iseconds) on $(hostname) ==="
"$PYTHON" analyze_e0m3_choice.py --model_name "$MODEL" --layer_stride "$STRIDE" \
    --out "$OUT_DIR/${MODEL}.csv" 2>&1 | tee "$OUT_DIR/${MODEL}.log"
echo "=== done $(date -Iseconds) ==="

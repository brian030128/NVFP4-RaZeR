#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=lambda_pred
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Per-model statistics that might PREDICT which election rule a model needs, so the rule does not
# have to be swept per model. See analyze_lambda_predictor.py for the hypothesis: the election uses
# the diagonal surrogate of S = E[x x^T], its error is bounded by the off-diagonal mass of S, and
# `h<lambda>` is a margin against exactly that error -- so more off-diagonal mass should need a
# larger lambda.
#
#   sbatch slurm/lambda_pred.sbatch qwen3-4b
#   MODELS="a b c" sbatch slurm/lambda_pred.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODELS=${MODELS:-${1:-qwen3-4b}}
STRIDE=${STRIDE:-8}

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

LIBDIR="$PWD/.joblib"
mkdir -p "$LIBDIR"
export PYTHONPATH="$LIBDIR:${PYTHONPATH:-}"

# gated models load from a local snapshot path; the hub is only needed for ungated weights and the
# calibration text, both of which land on node-local scratch
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT

OUT_DIR="results/lambda"
mkdir -p "$OUT_DIR" slurm/logs

for M in $MODELS; do
    echo "############ $M  $(date -Iseconds)"
    "$PYTHON" analyze_lambda_predictor.py \
        --model_name "$M" --layer_stride "$STRIDE" \
        --out "$OUT_DIR/${M}.json" 2>&1 | tee -a "$OUT_DIR/${M}.log"
    echo "############ exit=$? $M"
done

echo "=== done $(date -Iseconds) ==="

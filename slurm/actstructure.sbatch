#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=actstructure
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Do ACTIVATIONS have the per-channel structure that weights lack? Runs the same two-way
# decomposition on real calibration activations that run_blockorder_sim.py runs on weights, so the
# col_share numbers are directly comparable. Needs a GPU for the forward passes.
#
#   sbatch slurm/actstructure.sbatch llama-3.1-8b-local

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-llama-3.1-8b-local}

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

LIBDIR="$PWD/.joblib"
mkdir -p "$LIBDIR"
[[ -d "$LIBDIR/loguru" ]] || uv pip install --quiet --target "$LIBDIR" loguru
export PYTHONPATH="$LIBDIR:${PYTHONPATH:-}"

# The gated model is loaded from a local snapshot path (see model2path.json), so the hub is only
# needed for the ungated wikitext calibration set, which lands on node-local scratch.
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT

OUT_DIR="results/blockorder"
mkdir -p "$OUT_DIR" slurm/logs

echo "=== $MODEL $(date -Iseconds) on $(hostname) ==="
"$PYTHON" run_actstructure_sim.py --model_name "$MODEL" --nsamples 4 --layer_stride 8 \
    --out "$OUT_DIR/act_${MODEL}.csv" 2>&1 | tee "$OUT_DIR/act_${MODEL}.log"
echo "=== done $(date -Iseconds) ==="

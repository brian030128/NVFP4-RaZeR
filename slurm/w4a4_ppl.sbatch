#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=w4a4_ppl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Pick E0M3 on the DIAGONAL-HESSIAN objective instead of raw weight MSE.
#
# Every tag grid measured in this study so far used plain weight MSE, which CLAUDE.md's own
# analysis says does not predict layer output error: for Y = X W^T the output error is
# tr(dW S dW^T) with S = E[x x^T], and the measured spread of diag(S) is max/min ~4.1e3. Weighting
# the per-block selection loss by E[x_j^2] turns the criterion into the GPTQ/OBQ diagonal estimate
# of the LAYER OUTPUT error, which is what perplexity charges for.
#
# `run_ppl.py` has no importance plumbing at all; `run_ppl_sweep.py` does -- it collects importance
# on the unquantized model before any weight is overwritten, and only *_hess configs consume it.
# So this uses the sweep entry point.
#
#   sbatch slurm/hess_ppl.sbatch llama-3.1-8b-local
#   CONFIGS="..." sbatch slurm/hess_ppl.sbatch qwen3-4b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-llama-3.1-8b-local}
DATASETS=${DATASETS:-wikitext,c4}
TB=${TB:-8x64}
CALIB=${CALIB:-4}

# The hess variants are the point; the two controls bracket them.
#   dom            -- alpha search only, E0M3 off (the "never worse" floor)
#   h1.5           -- today's MSE-based election
#   hess_h1.5      -- same election, importance-weighted criterion
#   hess_coclcol.. -- + reordering, with the grid the election actually uses
CONFIGS=${CONFIGS:-"nvfp4,\
nvfp4_4over6,\
mix_4_6_clipbase_hess_e2m1/nvfp4_4over6@${TB},\
mix_4_6_clipheade0_hess_m1/nvfp4_4over6@${TB},\
mix_4_6_clipheade0_hess_coclcol_m1/nvfp4_4over6@${TB},\
mix_4_6_clipheade0_hess_m1@${TB},\
mix_4_6_clipheade0_hess_coclcol_m1@${TB}"}

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

LIBDIR="$PWD/.joblib"
mkdir -p "$LIBDIR"
[[ -d "$LIBDIR/loguru" ]] || uv pip install --quiet --target "$LIBDIR" loguru
export PYTHONPATH="$LIBDIR:${PYTHONPATH:-}"

# gated models load from a local snapshot path (model2path.json), so the hub is only needed for the
# ungated eval/calibration data, which lands on node-local scratch
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT

OUT_DIR="results/w4a4"
mkdir -p "$OUT_DIR" slurm/logs

echo "=== $MODEL  $TB  $DATASETS  calib=$CALIB  $(date -Iseconds) on $(hostname) ==="
echo "configs: $CONFIGS"

"$PYTHON" run_ppl_sweep.py --calib_source "${CALIB_SRC:-wikitext}" \
    --model_name "$MODEL" \
    --datasets "$DATASETS" \
    --sweep w4a4 \
    --configs "$CONFIGS" \
    --calib_batches "$CALIB" \
    --output "$OUT_DIR/${MODEL}${CALIB_TAG:-}.json" 2>&1 | tee -a "$OUT_DIR/${MODEL}.log"

echo "=== done $(date -Iseconds) ==="

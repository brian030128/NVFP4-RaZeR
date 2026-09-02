#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=actperm_ppl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# W4A4 with a fixed calibration-derived ACTIVATION channel permutation. This is the one axis where
# the tag grid has real structure to group: col_share 0.169 against 0.004 for weights, and a
# reorder lift over the cell-shuffle control of +0.081 against +0.003.
#
#   sbatch slurm/actperm_ppl.sbatch llama-3.1-8b-local

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-llama-3.1-8b-local}
ATB=${ATB:-16x64}
WTB=${WTB:-8x64}

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
LIBDIR="$PWD/.joblib"; mkdir -p "$LIBDIR"
[[ -d "$LIBDIR/loguru" ]] || uv pip install --quiet --target "$LIBDIR" loguru
export PYTHONPATH="$LIBDIR:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"; mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT

OUT="results/actperm/${MODEL}.json"
mkdir -p "$(dirname "$OUT")" slurm/logs
LOG="results/actperm/${MODEL}.log"

echo "=== $MODEL  W$WTB A$ATB  $(date -Iseconds) on $(hostname) ===" | tee -a "$LOG"

run () {  # $1 = extra flags, for the with/without permutation pair
    "$PYTHON" run_actperm_ppl.py --model_name "$MODEL" \
        --w_type_block "$WTB" --a_type_block "$ATB" \
        --output "$OUT" $1 2>&1 | tee -a "$LOG"
}

# A4 baseline first (nvfp4_4over6 activations, no type block), then MixFP4 activations with and
# without the channel permutation. The pair is what isolates the permutation's contribution.
"$PYTHON" run_actperm_ppl.py --model_name "$MODEL" \
    --w_dtype mix_4_6_clipheadx_h1.5 --a_dtype nvfp4_4over6 \
    --w_type_block "$WTB" --a_type_block "$ATB" --output "$OUT" 2>&1 | tee -a "$LOG"
run ""
run "--act_perm"

echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

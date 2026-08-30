#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=peakveto
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# The peakedness veto on ACTIVATIONS, decided per tile at runtime -- no search, no permutation, no
# calibration. `pv<tau>` refuses E0M3 for a tile whose blocks average max/rms above tau.
#
# Grounding: the E0M3 gain has rank correlation -0.59 with block max/rms on both models measured,
# and the E0M3-preferring population sits at max/rms ~2.0 (a Gaussian 16-block) against ~2.33 for
# E2M1. So E0M3 is the "no outlier in this block" grid, and the veto is that rule made explicit.
#
# Both statistics are already available where the scale is computed, so the runtime cost is one
# extra reduction over data in registers.
#
#   sbatch slurm/peakveto_ppl.sbatch llama-3.1-8b-local
#   ATB=8x64 sbatch slurm/peakveto_ppl.sbatch qwen3-8b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-llama-3.1-8b-local}
ATB=${ATB:-16x64}
WTB=${WTB:-8x64}
W=${W:-mix_4_6_clipheade0_h1.5}
DS=${DS:-wikitext}

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
LIBDIR="$PWD/.joblib"; mkdir -p "$LIBDIR"
[[ -d "$LIBDIR/loguru" ]] || uv pip install --quiet --target "$LIBDIR" loguru
export PYTHONPATH="$LIBDIR:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"; mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT

OUT_DIR="results/peakveto/${MODEL}__A${ATB}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="results/peakveto/${MODEL}__A${ATB}.log"

echo "=== $MODEL  W$WTB A$ATB  $(date -Iseconds) on $(hostname) ===" | tee -a "$LOG"

run () {
    echo "############ a_dtype=$1  $(date -Iseconds)" | tee -a "$LOG"
    "$PYTHON" run_ppl.py --model_name "$MODEL" --datasets "$DS" --seq_len 2048 \
        --output_dir "$OUT_DIR" \
        --w_bits 4 --w_groupsize 16 --w_dtype "$W" --w_type_block "$WTB" \
        --a_bits 4 --a_groupsize 16 --a_dtype "$1" --a_type_block "$ATB" 2>&1 | tee -a "$LOG"
}

# A4 references: no type block at all, then the type block with no veto.
run nvfp4_4over6
run mix_4_6_clipheade0_e2m1        # activations forced to E2M1 -- the "never elect" floor
run mix_4_6_clipheade0_h1.5        # the type block, unvetoed

# The veto sweep. A Gaussian 16-block sits at max/rms 2.0, so thresholds bracket that.
for tau in ${TAUS:-2.0 2.1 2.2 2.4 2.8}; do
    run "mix_4_6_clipheade0_pv${tau}_h1.5"
done

echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

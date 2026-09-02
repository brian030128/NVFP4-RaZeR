#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=a1_cmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Is `nvfp4_4over6` itself the wrong baseline on some models?
#
# Every delta in this study is quoted against nvfp4_4over6, which fixes the block scale by a
# two-point search alpha in {1, 1.5}. If alpha=1.5 is already harmful on a model, then that
# reference is degraded and every "+0.5145" measured against it is partly an artefact of the
# baseline rather than of the E0M3 election.
#
# So compare, W4A16, weights only:
#   nvfp4          alpha = 1 exactly, block max on the top code -- the unmodified format
#   nvfp4_4over6   alpha in {1, 1.5}
#   nvfp4_nover6   the wider alpha family without any type block
#   ..._dom        the headx alpha set, E0M3 never elected
#
#   sbatch slurm/baseline_cmp.sbatch qwen3-4b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-qwen3-4b}
DS=${DS:-wikitext,c4}

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
LIBDIR="$PWD/.joblib"; mkdir -p "$LIBDIR"
[[ -d "$LIBDIR/loguru" ]] || uv pip install --quiet --target "$LIBDIR" loguru
export PYTHONPATH="$LIBDIR:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"; mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT

OUT_DIR="results/a1_cmp/${MODEL}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="results/a1_cmp/${MODEL}.log"
echo "=== $MODEL $(date -Iseconds) on $(hostname) ===" | tee -a "$LOG"

run () {
    echo "############ $1" | tee -a "$LOG"
    "$PYTHON" run_ppl.py --model_name "$MODEL" --datasets "$DS" --seq_len 2048 \
        --output_dir "$OUT_DIR" \
        --w_bits 4 --w_groupsize 16 --w_dtype "$1" --w_type_block 8x64 \
        --a_bits 16 --a_dtype fp16 2>&1 | tee -a "$LOG"
}

run nvfp4
run mix_4_6_clipa1_e2m1
run mix_4_6_clipa1_h1.5
run mix_4_6_clipa1_argmin
run mix_4_6_clipa1_coclcol_h1.5

echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

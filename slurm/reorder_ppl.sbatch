#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=reorder_ppl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Does the reordering's weight-MSE reduction reach PERPLEXITY?
#
# Four W4A16 points at the 8x64 weight type block, all with the `heade0` clip preset so that the
# ONLY difference between rows 2-4 is the reordering:
#
#   nvfp4_4over6                     -- the E2M1-only reference
#   mix_4_6_clipheade0_h1.5          -- the current best realizable rule, unpermuted
#   mix_4_6_clipheade0_coclcol_h1.5  -- + column-chunk reordering (free per layer for down_proj)
#   mix_4_6_clipheade0_cocl_h1.5     -- + rows and columns (upper bound, ALGORITHM.md §7)
#
# The co-clustering search runs per weight tensor inside quant_weight, so the quantization pass
# takes ~1 h for the permuted configurations. Budget accordingly.
#
#   sbatch slurm/reorder_ppl.sbatch
#
# ENVIRONMENT NOTES (both learned the hard way, see results/reorder/REPORT.md §6):
#  * This repo has no venv of its own. beam_engine-v3's has torch/transformers/accelerate/datasets;
#    it is missing only `loguru`, which is overlaid into .joblib rather than installed into another
#    project's environment.
#  * This account has no HF token, so the gated meta-llama repo cannot be contacted at all -- not
#    even for the metadata check a cached load normally makes. The model is therefore loaded from
#    its LOCAL SNAPSHOT PATH (`llama-3.1-8b-local` in model2path.json), which never touches the hub.
#    That leaves the hub reachable for the ungated eval datasets, which download to node-local /tmp.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${MODEL:-llama-3.1-8b-local}
DATASETS=${DATASETS:-wikitext}
SEQ=${SEQ:-2048}
TYPE_BLOCK=${TYPE_BLOCK:-8x64}

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# `loguru` overlay -- pure python, no deps, keeps other projects' venvs untouched
LIBDIR="$PWD/.joblib"
mkdir -p "$LIBDIR"
if [[ ! -d "$LIBDIR/loguru" ]]; then
    uv pip install --quiet --target "$LIBDIR" loguru || \
        "$PYTHON" -m pip install --quiet --target "$LIBDIR" loguru
fi
export PYTHONPATH="$LIBDIR:${PYTHONPATH:-}"

# Datasets are ungated and download to node-local scratch, never /home or /work.
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT

# run_ppl.py names its result file from the QUANT CONFIG ALONE -- not the model, not the dataset --
# and exits early if that file already exists. So the output directory has to carry the model and
# dataset, or a second model silently "completes" by skipping every config.
OUT_DIR="results/reorder/ppl/${MODEL}__$(echo "$DATASETS" | tr ',' '-')"
mkdir -p "$OUT_DIR" slurm/logs

echo "=== $MODEL  $TYPE_BLOCK  $DATASETS  start $(date -Iseconds) on $(hostname) ==="
"$PYTHON" -c "import torch, transformers, datasets, loguru, accelerate; print('env ok', torch.__version__, transformers.__version__, datasets.__version__)" || exit 1

run () {
    echo "############ $1  $(date -Iseconds)"
    "$PYTHON" run_ppl.py --model_name "$MODEL" \
        --datasets "$DATASETS" --seq_len "$SEQ" --output_dir "$OUT_DIR" \
        --w_bits 4 --w_groupsize 16 --w_dtype "$1" --w_type_block "$TYPE_BLOCK" \
        --a_bits 16 --a_dtype fp16
    echo "############ exit=$? $1  $(date -Iseconds)"
}

# CONFIGS can override the list, e.g. to rerun a single failed point:
#   CONFIGS=mix_4_6_clipheade0_cocl_h1.5 sbatch slurm/reorder_ppl.sbatch
# run_ppl.py skips a config whose output file already exists, so a rerun is cheap and idempotent.
CONFIGS=${CONFIGS:-nvfp4_4over6 mix_4_6_clipheade0_h1.5 mix_4_6_clipheade0_coclcol_h1.5 mix_4_6_clipheade0_cocl_h1.5}
for cfg in $CONFIGS; do
    run "$cfg"
done

echo "=== done $(date -Iseconds) ==="

#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=kse_obj_ppl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Does dropping CE from the election change the perplexity?
#
# The shipped rule elects a tile when max(mean CE + k SE, mean KL + k SE) < 0, so both
# objectives must be confidently negative. MIXFP4_REPORT.md argues for that conjunction -- CE
# alone can be lowered by fitting the calibration corpus's own tokens, KL alone by restoring
# agreement on positions the task does not care about -- but never measured the alternatives.
#
# This runs the report's own perplexity path (run_kse_paper.py, WikiText-2 in 141 windows and
# C4 in 256 seed-0 crops at 2048, W4A4) over four_over_six, k3 and k3_kl, so the KL-only row is
# directly comparable with the published one. k = 2 is still elected because the frozen-map
# cross-check is defined on its ranking; only k = 3 is evaluated.
#
#   sbatch slurm/kse_objective_ppl.sbatch llama8b
#   OBJECTIVES=max,kl,ce sbatch slurm/kse_objective_ppl.sbatch qwen4b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs

MODEL=${1:?llama8b | qwen4b | qwen27b}
OBJECTIVES=${OBJECTIVES:-max,kl}
KVALUES=${KVALUES:-3}

PROJECT="$PWD"
SCRATCH="/tmp/$USER/kseobj_${MODEL}_$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
export HF_HOME="$SCRATCH/hf" UV_CACHE_DIR="$SCRATCH/uv" TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
UV=/home/u4320956/.local/bin/uv
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps" loguru==0.7.3 \
  "huggingface_hub>=0.34,<1.0" || exit 1
rm -rf "$SCRATCH"/deps/{transformers,tokenizers,torch,nvidia,triton}
PYPATH="$PROJECT:$SCRATCH/deps"
if [[ "$MODEL" == "qwen27b" ]]; then
    "$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps27" --no-deps \
      transformers==5.16.1 'huggingface_hub==1.5.0' tokenizers safetensors hf-xet loguru || exit 1
    PYPATH="$PROJECT:$SCRATCH/deps27:$SCRATCH/deps"
fi

OUT="$PROJECT/results/kse_objective/job_$SLURM_JOB_ID"
mkdir -p "$OUT"

echo "=== objective ablation PPL  $MODEL  objectives=$OBJECTIVES  k=$KVALUES  $(date -Iseconds) on $(hostname) ==="

# run_kse_paper.py reads the frozen score shards, which the shipped calibration no longer has,
# so they are regenerated first. The election is then checked against the shipped frozen map
# inside run_kse_paper itself.
CALIB="$SCRATCH/calibration_$MODEL"
env PYTHONPATH="$PYPATH" "$PYTHON" run_math_code_calibration.py \
    --model "$MODEL" --out "$CALIB" --allow-source-drift \
    > "$OUT/calibration_$MODEL.log" 2>&1 || { tail -20 "$OUT/calibration_$MODEL.log"; exit 1; }
env PYTHONPATH="$PYPATH" "$PYTHON" "$PROJECT/scripts/check_map_digest.py" "$CALIB" "$MODEL" || \
    echo 'NOTE: maps.json differs from shipped; run_kse_paper checks the frozen map itself.'

# run_kse_paper defaults to the shipped calibration directory, which no longer has the score
# shards the election reads, so point it at the fresh one.
env PYTHONPATH="$PYPATH" "$PYTHON" run_kse_paper.py \
    --model "$MODEL" --out "$OUT/$MODEL" --calib "$CALIB" \
    --k-values "$KVALUES" --objectives "$OBJECTIVES" --skip-n256 \
    2>&1 | tee "$OUT/ppl_$MODEL.log"

echo "=== done $(date -Iseconds) ==="

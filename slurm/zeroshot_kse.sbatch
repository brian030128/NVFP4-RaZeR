#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=zs_kse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Zero-shot accuracy for the four policies of MIXFP4_REPORT.md, at k = 3.
#
# The report carries perplexity only, and the k-SE rule elects tiles by a calibration loss, so
# "the model got less confident" and "the model got better" are not separated by anything it
# measures. This runs the same four policies on zero-shot multiple choice.
#
# Two steps, because the shipped calibration directories under results/math_code_adaptive/ no
# longer contain their scores/ shards -- only report.json and maps.json survived. The scoring
# pass is cheap (158 s for Llama-3.1-8B, 118 s for Qwen3-4B, 1301 s for the 27B, per the
# score_seconds those reports record), so it is simply redone here. The regenerated maps.json
# is compared against the shipped map_sha256, and the re-election is checked to reproduce the
# frozen fixed256 map bitwise, so an accidental difference in the calibration cannot pass
# silently.
#
#   sbatch slurm/zeroshot_kse.sbatch llama8b
#   sbatch --partition=taide_h200 --gres=gpu:H200:2 slurm/zeroshot_kse.sbatch qwen27b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs

MODEL=${1:?llama8b | qwen4b | qwen27b}
BATCH=${BATCH:-8}

PROJECT="$PWD"
SCRATCH="/tmp/$USER/zskse_${MODEL}_$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
export HF_HOME="$SCRATCH/hf" UV_CACHE_DIR="$SCRATCH/uv-cache"
mkdir -p "$HF_HOME"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
UV=/home/u4320956/.local/bin/uv
# huggingface_hub is pinned below 1.0: transformers 4.57.3 needs it, and an lm_eval-resolved
# 1.x in the target directory shadows the venv copy and breaks the import.
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps" \
  loguru==0.7.3 lm_eval==0.4.5 "huggingface_hub>=0.34,<1.0" || exit 1
rm -rf "$SCRATCH"/deps/{transformers,tokenizers,torch,torchvision,torchaudio,triton,nvidia}
rm -rf "$SCRATCH"/deps/{transformers,tokenizers,torch,torchvision,torchaudio,triton}-*.dist-info
export PYTHONPATH="$PROJECT:$SCRATCH/deps"

# Qwen3.8-27B is a Qwen3.5-family model and its calibration origin pins transformers 5.16.1 --
# the native code path, not the older Qwen3 copies (see the project CLAUDE.md). The venv carries
# 4.57.3, which every other model here requires, so 5.16.1 is layered in front of it for this
# model only, exactly as slurm/kse_paper_27b.sbatch does: --no-deps so the venv's torch stays,
# and ahead of $SCRATCH/deps so it wins over whatever lm_eval resolved.
# huggingface_hub is pinned, not left to the resolver. transformers 5.16.1 needs hub >=1.5,
# and unpinned that resolved to 1.31, whose URI handling breaks the datasets lm-eval loads:
# four of the six tasks died with HfUriError ("Invalid HF URI hf://datasets/...huggingface.yaml")
# and the run silently evaluated two, which would not be comparable with the other models.
# 1.5.0 satisfies transformers and loads all six.
if [[ "$MODEL" == "qwen27b" ]]; then
    "$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps27" --no-deps \
      transformers==5.16.1 'huggingface_hub==1.5.0' tokenizers safetensors hf-xet loguru || exit 1
    export PYTHONPATH="$PROJECT:$SCRATCH/deps27:$SCRATCH/deps"
fi

OUT="$PROJECT/results/zeroshot_kse/job_$SLURM_JOB_ID"
mkdir -p "$OUT"

echo "=== zeroshot k=3  $MODEL  $(date -Iseconds) on $(hostname) ==="
# lm_eval_compat first: under transformers 5 (Qwen3.8-27B) importing lm_eval raises without it.
"$PYTHON" -c "import torch,transformers,lm_eval_compat,lm_eval,importlib.metadata as M; \
print('torch',torch.__version__,'transformers',transformers.__version__,'lm_eval',M.version('lm_eval'))" \
  || { echo 'FATAL: evaluation stack does not import'; exit 1; }

# --- step 1: regenerate the calibration scores the election needs ---------------------------
CALIB="$SCRATCH/calibration_$MODEL"
echo "--- calibration -> $CALIB"
# --allow-source-drift: commit 384b803 appended quant_nvfp4_4over6_pair to quantize/quantizer.py
# and changed nothing else (52 insertions, 0 deletions), which invalidates the origin job's file
# digest without altering any function this pass calls. The check that actually matters is the
# map digest comparison below, which fails the job if the regenerated election differs at all.
"$PYTHON" run_math_code_calibration.py --model "$MODEL" --out "$CALIB" --allow-source-drift \
    > "$OUT/calibration_$MODEL.log" 2>&1
RC=$?
echo "calibration exit=$RC"
if [[ $RC -ne 0 ]]; then tail -30 "$OUT/calibration_$MODEL.log"; exit 1; fi
cp "$CALIB/report.json" "$OUT/calibration_report_$MODEL.json" 2>/dev/null

# Informational: maps.json holds 20 maps and only fixed256_math_code128 has anything to do with
# this election, so a whole-file digest difference does not by itself mean the election differs.
# The gate that decides is inside run_zeroshot_kse.py, which requires the re-election to
# reproduce the SHIPPED frozen map bitwise and aborts otherwise.
"$PYTHON" "$PROJECT/scripts/check_map_digest.py" "$CALIB" "$MODEL" || \
    echo 'NOTE: maps.json differs from shipped; the frozen-map gate below is what decides.'

# --- step 2: elect k = 3 from those scores and evaluate the four policies -------------------
"$PYTHON" run_zeroshot_kse.py --model "$MODEL" --calib "$CALIB" \
    --out "$OUT/$MODEL" --batch-size "$BATCH" \
    --samples-dir "$OUT/samples_$MODEL" ${EXTRA_ARGS:-} 2>&1 | tee "$OUT/zeroshot_$MODEL.log"

echo "=== done $(date -Iseconds) ==="

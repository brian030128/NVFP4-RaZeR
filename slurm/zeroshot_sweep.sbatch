#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=zs_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=200G
#SBATCH --time=40:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Zero-shot accuracy for the MIXFP4_REPORT.md §1 panel.
#
# §1 reports perplexity only. Perplexity is a next-token loss and cannot separate a genuinely
# better model from a less overconfident one, so the same seven configurations are re-run on
# zero-shot multiple choice with lm-eval-harness. The quantization path is the one that produced
# §1: W4A4 prefill, weights 8x64 / alpha = 1, activations nvfp4_4over6 at 16x64.
#
#   sbatch slurm/zeroshot_sweep.sbatch qwen3-8b
#   PARTITION_GPU=H200 sbatch --partition=taide_h200 --gres=gpu:H200:1 \
#       slurm/zeroshot_sweep.sbatch qwen3-14b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs

MODEL=${1:?model name from model2path.json}
BATCH=${BATCH:-16}
TB=${TB:-8x64}

# The seven §1 rows, plus an unquantized reference so the accuracy cost of W4A4 itself is visible
# alongside the element-type deltas. Labels match results/w4a4/<model>_types.json exactly.
CONFIGS=${CONFIGS:-"fp16,\
nvfp4/nvfp4_4over6,\
mix_4_6_clipa1_e2m1/nvfp4_4over6@${TB},\
mix_4_6_clipa1_h10/nvfp4_4over6@${TB},\
mix_4_6_clipa1_hess_h1.5/nvfp4_4over6@${TB},\
mix_4_6_clipa1_hess_h10/nvfp4_4over6@${TB},\
mix_4_6_clipa1_hess_m1/nvfp4_4over6@${TB},\
mix_4_6_clipa1_hess_impg16_h10/nvfp4_4over6@${TB}"}

PROJECT="$PWD"
SCRATCH="/tmp/$USER/zssweep_$SLURM_JOB_ID"
export HF_HOME="$SCRATCH/hf" UV_CACHE_DIR="$SCRATCH/uv-cache"
mkdir -p "$HF_HOME"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
UV=/home/u4320956/.local/bin/uv

# huggingface_hub is pinned below 1.0: transformers 4.57.3 requires it, and an lm_eval-resolved
# 1.x in the target directory shadows the venv copy and breaks the import.
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps" \
  loguru==0.7.3 lm_eval==0.4.5 "huggingface_hub>=0.34,<1.0" || exit 1
# The venv's pinned transformers/torch must win over anything lm_eval dragged in.
rm -rf "$SCRATCH"/deps/{transformers,tokenizers,torch,torchvision,torchaudio,triton,nvidia}
rm -rf "$SCRATCH"/deps/{transformers,tokenizers,torch,torchvision,torchaudio,triton}-*.dist-info
export PYTHONPATH="$PROJECT:$SCRATCH/deps"

OUT_DIR="results/zeroshot_w4a4"
mkdir -p "$OUT_DIR"

echo "=== $MODEL  tb=$TB  bs=$BATCH  $(date -Iseconds) on $(hostname) ==="
echo "configs: $CONFIGS"
"$PYTHON" -c "import torch,transformers,lm_eval,importlib.metadata as M; \
print('torch',torch.__version__,'transformers',transformers.__version__,'lm_eval',M.version('lm_eval'))"

"$PYTHON" run_zeroshot_sweep.py \
    --model_name "$MODEL" \
    --sweep w4a4 \
    --configs "$CONFIGS" \
    --batch_size "$BATCH" \
    --calib_source wikitext \
    --calib_batches 4 \
    ${EXTRA:-} \
    --output "$OUT_DIR/${OUT_NAME:-$MODEL}.json" 2>&1 | tee -a "$OUT_DIR/${OUT_NAME:-$MODEL}.log"

echo "=== done $(date -Iseconds) ==="

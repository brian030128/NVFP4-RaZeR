#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=diffgemma_ppl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Overridable via:  sbatch --export=ALL,CONFIGS=...,MAXWIN=..,NDRAWS=..,TAG=.. slurm/diffgemma_ppl.sbatch
CONFIGS="${CONFIGS:-fp,nvfp4_4over6,mixfp4-8x64,mix_4_6-8x64}"
MAXWIN="${MAXWIN:-32}"
NDRAWS="${NDRAWS:-2}"
MAXTOK="${MAXTOK:-0}"
REVEAL="${REVEAL:-0.0}"
TAG="${TAG:-full}"

WORKDIR="/home/u4320956/NVFP4-RaZeR/.claude/worktrees/diffgemma-mse"
cd "$WORKDIR"

PY="/home/u4320956/diffgemma_venv/bin/python"     # layered venv: transformers 5.15 + cuda128 torch

set -uo pipefail

# HF cache on node-local /tmp; model cached at a STABLE path so a smoke+full pair on the same node
# reuses the 52 GB download.
export HF_HOME="/tmp/$USER/hf_ppl"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO="google/diffusiongemma-26B-A4B-it"
MODEL_DIR="/tmp/$USER/diffgemma_model"
OUT_DIR="results/diffgemma_ppl"
mkdir -p "$OUT_DIR" slurm/logs

echo "=== node $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "=== config: CONFIGS=$CONFIGS MAXWIN=$MAXWIN NDRAWS=$NDRAWS MAXTOK=$MAXTOK TAG=$TAG"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "=== $(date -Iseconds)  downloading $REPO -> $MODEL_DIR ==="
    "$PY" - "$REPO" "$MODEL_DIR" <<'PYDL'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo, local_dir=dest,
                  allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"],
                  max_workers=8)
print("snapshot ready at", dest)
PYDL
else
    echo "=== model already cached at $MODEL_DIR ==="
fi

echo "=== $(date -Iseconds)  running pseudo-PPL ($TAG) ==="
"$PY" run_diffgemma_ppl.py \
    --model_dir "$MODEL_DIR" \
    --configs "$CONFIGS" \
    --window_blocks 8 --max_windows "$MAXWIN" --n_draws "$NDRAWS" --max_tokens "$MAXTOK" \
    --reveal_frac "$REVEAL" \
    --out "$OUT_DIR/diffgemma_ppl_${TAG}.csv" \
    2>&1 | tee "$OUT_DIR/run_${TAG}_${SLURM_JOB_ID}.log"

echo "=== done $(date -Iseconds) ==="

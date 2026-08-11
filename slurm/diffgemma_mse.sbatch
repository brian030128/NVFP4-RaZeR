#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=diffgemma_mse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# --- run from the worktree so results land on /home and get committed ---
WORKDIR="/home/u4320956/NVFP4-RaZeR/.claude/worktrees/diffgemma-mse"
cd "$WORKDIR"

# --- conda env cuda128 (transformers 5.8.1 / torch 2.6 cu124 / safetensors / hf_hub) ---
# (source + activate before `set -u`: conda's shell code references unbound vars)
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate cuda128
PY="/home/u4320956/.conda/envs/cuda128/bin/python"

set -uo pipefail

# --- HF cache + model download on node-local /tmp, never /home or /work ---
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
export HF_HUB_ENABLE_HF_TRANSFER=0

export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO="google/diffusiongemma-26B-A4B-it"
MODEL_DIR="/tmp/$USER/diffgemma_$SLURM_JOB_ID"
OUT_DIR="results/diffgemma_mse"
mkdir -p "$OUT_DIR" slurm/logs

echo "=== node $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=== $(date -Iseconds)  downloading $REPO -> $MODEL_DIR (weights + json only) ==="
"$PY" - "$REPO" "$MODEL_DIR" <<'PYDL'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
p = snapshot_download(repo, local_dir=dest,
                      allow_patterns=["*.safetensors", "*.json"],
                      max_workers=8)
print("snapshot at", p)
PYDL

echo "=== $(date -Iseconds)  running weight-MSE sim (8x64 weights) ==="
"$PY" run_diffgemma_mse.py \
    --model_dir "$MODEL_DIR" \
    --type_block 8x64 \
    --out "$OUT_DIR/diffgemma_mse_8x64.csv" \
    2>&1 | tee "$OUT_DIR/run_${SLURM_JOB_ID}.log"

echo "=== done $(date -Iseconds) ==="

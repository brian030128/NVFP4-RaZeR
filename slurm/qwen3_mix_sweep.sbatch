#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=qwen3_mix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Overridable: sbatch --export=ALL,MODEL=qwen3-8b,DATASETS=wikitext+c4,LIMIT=,TAG=full slurm/qwen3_mix_sweep.sbatch
MODEL="${MODEL:?set MODEL, e.g. qwen3-8b}"
# NVFP4 4over6 vs mix_4_6 @ 8x64 (argmin) plus the two margin winners from the Llama study.
CONFIGS="${CONFIGS:-fp16,nvfp4_4over6,mix_4_6@8x64,mix_4_6_m1@8x64,mix_4_6_h1.5@8x64}"
DATASETS="${DATASETS:-wikitext+c4}"        # '+' avoids the sbatch --export comma split; converted below
LIMIT="${LIMIT:-}"                          # e.g. 8 for a smoke; empty = all windows
TAG="${TAG:-full}"

WORKDIR="/home/u4320956/NVFP4-RaZeR/.claude/worktrees/diffgemma-mse"
cd "$WORKDIR"

PY="/home/u4320956/razer_venv/bin/python"   # layered venv: transformers 4.57.3 (repo qmodules) + cuda128 torch

set -uo pipefail

export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

DS="${DATASETS//+/,}"                        # wikitext+c4 -> wikitext,c4
OUT_DIR="results/qwen3_mix_sweep"
mkdir -p "$OUT_DIR" slurm/logs
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit_samples $LIMIT"

echo "=== node $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "=== $(date -Iseconds)  MODEL=$MODEL CONFIGS=$CONFIGS DATASETS=$DS TAG=$TAG LIMIT='$LIMIT'"

"$PY" run_ppl_sweep.py \
    --model_name "$MODEL" \
    --sweep w4a16 \
    --configs "$CONFIGS" \
    --datasets "$DS" \
    --seq_len 2048 \
    $LIMIT_ARG \
    --output "$OUT_DIR/${MODEL}_${TAG}.json" \
    2>&1 | tee "$OUT_DIR/run_${MODEL}_${TAG}_${SLURM_JOB_ID}.log"

echo "=== done $(date -Iseconds) ==="

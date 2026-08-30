#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=reorder_sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# CPU-only sweep: how much of the 1x16 E0M3 gain a row/column reordering gives back to a coarse
# type block, on real weights. The GPU is requested only because this cluster rations cores at
# 12 per GPU and has no CPU-only partition; it is left idle on purpose.
#
#   sbatch slurm/reorder_sim.sbatch llama-2-7b
#   sbatch slurm/reorder_sim.sbatch llama-3.1-8b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL=${1:-llama-2-7b}
STRIDE=${2:-8}
TYPE_BLOCKS=${TYPE_BLOCKS:-8x64}
RULES=${RULES:-argmin,h1.5}
PROJ=${PROJ:-q_proj,v_proj,o_proj,up_proj,down_proj}
TAG=${TAG:-$MODEL}
AXES=${AXES:-both}
SEARCH_RULE=${SEARCH_RULE:-}

# --- environment ---
source /etc/profile.d/lmod.sh 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/DiRotQ/.venv/bin/python}

# --- weights ---
# This account has no HF token, so gated repos cannot be downloaded. The models that matter are
# already in the legacy /work cache, so read it OFFLINE: HF_HUB_OFFLINE=1 guarantees no download
# and no new blobs written to the quota'd filesystem. Set OFFLINE=0 for an ungated repo that has to
# be fetched, and it falls back to the usual node-local /tmp cache.
OFFLINE=${OFFLINE:-1}
if [[ "$OFFLINE" == "1" ]]; then
    export HF_HOME="/work/$USER/hf"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
    mkdir -p "$HF_HOME"
    trap 'rm -rf "/tmp/$USER/hf_$SLURM_JOB_ID"' EXIT
fi
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"

OUT_DIR="results/reorder"
mkdir -p "$OUT_DIR" slurm/logs

echo "=== $MODEL  stride=$STRIDE  start $(date -Iseconds)  on $(hostname) ==="

"$PYTHON" run_reorder_sim.py \
    --model_name "$MODEL" \
    --layer_stride "$STRIDE" \
    --projections "$PROJ" \
    --type_blocks "$TYPE_BLOCKS" \
    --rules "$RULES" \
    --clip "${CLIP:-heade0}" \
    --axes "$AXES" \
    --search_rule "$SEARCH_RULE" \
    --threads 12 \
    --out "$OUT_DIR/${TAG}.csv" 2>&1 | tee -a "$OUT_DIR/${TAG}.log"

echo "=== done $(date -Iseconds) ==="

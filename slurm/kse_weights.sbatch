#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=kse_weights
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs
PROJECT="$PWD"
SCRATCH="/tmp/$USER/kse_weights_$SLURM_JOB_ID"
export HF_HOME="$SCRATCH/hf" UV_CACHE_DIR="$SCRATCH/uv-cache"
mkdir -p "$HF_HOME"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
UV=/home/u4320956/.local/bin/uv
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps" loguru==0.7.3
export PYTHONPATH="$PROJECT:$SCRATCH/deps"
SEL=/home/u4320956/NVFP4-RaZeR/.claude/worktrees/tile-selection-analysis/results/kse_selection/job_337344
OUT="$PROJECT/results/kse_selection/weights_$SLURM_JOB_ID"
mkdir -p "$OUT"
FAILED=0
for MODEL in llama8b qwen4b; do
  "$PYTHON" analyze_kse_weights.py --model "$MODEL" --selection "$SEL" --out "$OUT" \
    > "$OUT/${MODEL}.log" 2>&1 || FAILED=1
done
echo "=== done failed=$FAILED $(date -Iseconds) ==="
exit "$FAILED"

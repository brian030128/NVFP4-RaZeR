#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide_h200
#SBATCH --job-name=kse_paper_27b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs
PROJECT="$PWD"
SCRATCH="/tmp/$USER/kse27b_$SLURM_JOB_ID"
export HF_HOME="$SCRATCH/hf" UV_CACHE_DIR="$SCRATCH/uv-cache"
mkdir -p "$HF_HOME"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
UV=/home/u4320956/.local/bin/uv
# Qwen3.8-27B needs the native Transformers 5.16.1 hybrid text code, exactly as the
# published fixed-256 evaluation used it. --no-deps keeps the venv's torch in place.
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps27" --no-deps \
  transformers==5.16.1 huggingface_hub tokenizers safetensors hf-xet loguru
export PYTHONPATH="$PROJECT:$SCRATCH/deps27"
OUT="$PROJECT/results/kse_paper/job_$SLURM_JOB_ID"
mkdir -p "$OUT"
"$PYTHON" -c "import transformers;print('transformers',transformers.__version__)"
"$PYTHON" run_kse_paper.py --model qwen27b --out "$OUT/qwen27b" > "$OUT/qwen27b.log" 2>&1
echo "=== done $(date -Iseconds) ==="

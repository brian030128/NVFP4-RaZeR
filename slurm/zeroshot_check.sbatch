#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide_h200
#SBATCH --job-name=zeroshot_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs
PROJECT="$PWD"
SCRATCH="/tmp/$USER/zeroshot_$SLURM_JOB_ID"
export HF_HOME="$SCRATCH/hf" UV_CACHE_DIR="$SCRATCH/uv-cache"
mkdir -p "$HF_HOME"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
UV=/home/u4320956/.local/bin/uv
# huggingface_hub is pinned below 1.0: transformers 4.57.3 requires it, and an
# lm_eval-resolved 1.x in the target directory shadows the venv copy and breaks the import.
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps" \
  loguru==0.7.3 lm_eval==0.4.5 "huggingface_hub>=0.34,<1.0"
# The venv's pinned transformers/torch must win; the calibration record asserts the version.
rm -rf "$SCRATCH"/deps/{transformers,tokenizers,torch,torchvision,torchaudio,triton,nvidia}
rm -rf "$SCRATCH"/deps/{transformers,tokenizers,torch,torchvision,torchaudio,triton}-*.dist-info
export PYTHONPATH="$PROJECT:$SCRATCH/deps"
OUT="$PROJECT/results/zeroshot_check/job_$SLURM_JOB_ID"
mkdir -p "$OUT"
"$PYTHON" -c "import transformers,lm_eval,importlib.metadata as M;print('transformers',transformers.__version__,'lm_eval',M.version('lm_eval'))"
"$PYTHON" run_zeroshot_check.py --model qwen4b --counts 256,65536 --out "$OUT/qwen4b" \
  > "$OUT/qwen4b.log" 2>&1
echo "=== done $(date -Iseconds) ==="

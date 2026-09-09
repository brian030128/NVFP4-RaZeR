#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide_h200
#SBATCH --job-name=paper_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs
PROJECT="$PWD"
SCRATCH="/tmp/$USER/paper_baseline_$SLURM_JOB_ID"
export HF_HOME="$SCRATCH/hf" UV_CACHE_DIR="$SCRATCH/uv-cache"
mkdir -p "$HF_HOME"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
UV=/home/u4320956/.local/bin/uv
# Same dependency set as the published fixed-256 paper evaluation (transformers 4.57.3).
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps" loguru==0.7.3
OUT="$PROJECT/results/paper_baseline/job_$SLURM_JOB_ID"
export PYTHONPATH="$PROJECT:$SCRATCH/deps"
"$PYTHON" prepare_paper_baseline.py --out "$OUT"
FAILED=0
for CASE in {0..5}; do
  "$PYTHON" run_fixed256_paper_eval.py --root "$OUT" --case "$CASE" \
    > "$OUT/case_${CASE}.out" 2> "$OUT/case_${CASE}.err" || FAILED=1
done
echo "=== cases done, failed=$FAILED $(date -Iseconds) ==="
exit "$FAILED"

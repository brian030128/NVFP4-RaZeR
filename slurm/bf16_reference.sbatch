#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide_h200
#SBATCH --job-name=bf16_reference
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:3
#SBATCH --mem=540G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs
PROJECT="$PWD"
SCRATCH="/tmp/$USER/bf16ref_$SLURM_JOB_ID"
export HF_HOME="$SCRATCH/hf" UV_CACHE_DIR="$SCRATCH/uv-cache"
mkdir -p "$HF_HOME"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
UV=/home/u4320956/.local/bin/uv
# Qwen3.8-27B needs the native 5.16.1 hybrid text code; the other two use the
# venv's pinned 4.57.3, exactly as their published evaluations did.
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps27" --no-deps \
  transformers==5.16.1 huggingface_hub tokenizers safetensors hf-xet loguru
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps" loguru==0.7.3
OUT="$PROJECT/results/bf16_reference/job_$SLURM_JOB_ID"
mkdir -p "$OUT"
PYTHONPATH="$PROJECT:$SCRATCH/deps27" "$PYTHON" prepare_bf16_reference.py --out "$OUT"
TASK_PIDS=()
for CASE in 0 1 2; do
  DEPS="$SCRATCH/deps"
  if (( CASE == 0 )); then DEPS="$SCRATCH/deps27"; fi
  srun --exclusive --exact --mpi=none --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gres=gpu:H200:1 --mem=170G \
    env PYTHONPATH="$PROJECT:$DEPS" "$PYTHON" run_fixed256_paper_eval.py --root "$OUT" --case "$CASE" \
    > "$OUT/case_${CASE}.out" 2> "$OUT/case_${CASE}.err" &
  TASK_PIDS+=("$!")
done
FAILED=0
for TASK_PID in "${TASK_PIDS[@]}"; do
  if ! wait "$TASK_PID"; then FAILED=1; fi
done
echo "=== done failed=$FAILED $(date -Iseconds) ==="
exit "$FAILED"

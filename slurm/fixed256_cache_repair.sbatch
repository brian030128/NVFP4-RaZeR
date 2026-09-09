#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide_h200
#SBATCH --job-name=fixed256_cache_repair
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:8
#SBATCH --mem=800G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PROJECT="$PWD"
SCRATCH="/tmp/$USER/fixed256_$SLURM_JOB_ID"
export HF_HOME="$SCRATCH/hf" UV_CACHE_DIR="$SCRATCH/uv-cache"
mkdir -p "$HF_HOME"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
UV=/home/u4320956/.local/bin/uv
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps27" --no-deps \
  transformers==5.16.1 huggingface_hub tokenizers safetensors hf-xet loguru
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps" loguru==0.7.3
OUT="$PROJECT/results/fixed256_paper_eval/job_$SLURM_JOB_ID"
export PYTHONPATH="$PROJECT:$SCRATCH/deps"
"$PYTHON" prepare_fixed256_paper_eval.py --out "$OUT"
"$PYTHON" prepare_fixed256_cache_repair.py --old "${1:?original results directory}" --new "$OUT"
TASK_PIDS=()
while read -r CASE; do
  DEPS="$SCRATCH/deps"
  if (( CASE < 11 )); then DEPS="$SCRATCH/deps27"; fi
  srun --exclusive --exact --mpi=none --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gres=gpu:H200:1 --mem=90G \
    env PYTHONPATH="$PROJECT:$DEPS" "$PYTHON" run_fixed256_paper_eval.py --root "$OUT" --case "$CASE" --repair-wiki \
    > "$OUT/repair_${CASE}.out" 2> "$OUT/repair_${CASE}.err" &
  TASK_PIDS+=("$!")
done < "$OUT/repair_cases.txt"
FAILED=0
for TASK_PID in "${TASK_PIDS[@]}"; do
  if ! wait "$TASK_PID"; then FAILED=1; fi
done
if [[ "$FAILED" == 0 ]]; then "$PYTHON" summarize_fixed256_paper_eval.py --root "$OUT"; fi
exit "$FAILED"

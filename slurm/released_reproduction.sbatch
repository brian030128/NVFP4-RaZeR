#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=released_reproduction
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:8
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PROJECT="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="/tmp/$USER/released_$SLURM_JOB_ID/hf"
export UV_CACHE_DIR="/tmp/$USER/released_$SLURM_JOB_ID/uv-cache"
SCRATCH="/tmp/$USER/released_$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$SCRATCH/source"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
GPU_TYPE=${GPU_TYPE:-H100}
OUT="$PROJECT/results/released_reproduction/job_$SLURM_JOB_ID"
mkdir -p "$OUT"
git archive e230099bf8006f571614b6c9a71cfd6556c49ef0 | tar -x -C "$SCRATCH/source"
if [[ "${REBUILD_RELEASE_ENV:-0}" == 1 ]]; then
  UV=/home/u4320956/.local/bin/uv
  export UV_PYTHON_INSTALL_DIR="$SCRATCH/python"
  "$UV" venv --python 3.10.18 --managed-python "$SCRATCH/venv"
  PYTHON="$SCRATCH/venv/bin/python"
  "$UV" pip install --python "$PYTHON" torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
  "$UV" pip install --python "$PYTHON" transformers==4.57.3 datasets==4.4.1 \
    accelerate==1.7.0 numpy==2.2.6 tokenizers==0.22.1 huggingface-hub==0.36.0 \
    safetensors==0.5.3 sentencepiece==0.2.0 loguru==0.7.3
  "$UV" pip freeze --python "$PYTHON" > "$OUT/environment.txt"
fi
/home/u4320956/.local/bin/uv pip install --python "$PYTHON" --target "$SCRATCH/deps" loguru==0.7.3
export PYTHONPATH="$SCRATCH/deps${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -c 'import loguru, torch, transformers, datasets, accelerate'
"$PYTHON" "$PROJECT/prepare_released_reproduction.py" --source "$SCRATCH/source" --out "$OUT"
TASK_PIDS=()
for CASE in {0..13}; do
  srun --exclusive --exact --mpi=none --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gres="gpu:$GPU_TYPE:1" --mem=40G \
    "$PYTHON" "$PROJECT/run_released_ppl_case.py" --source "$SCRATCH/source" --root "$OUT" --case "$CASE" \
    > "$OUT/case_${CASE}.out" 2> "$OUT/case_${CASE}.err" &
  TASK_PIDS+=("$!")
done
FAILED=0
for TASK_PID in "${TASK_PIDS[@]}"; do
  if ! wait "$TASK_PID"; then FAILED=1; fi
done
if [[ "$FAILED" == 0 ]]; then
  "$PYTHON" "$PROJECT/summarize_released_reproduction.py" --root "$OUT" ${UPDATE_REPORT_FLAG:---update-report}
fi
exit "$FAILED"

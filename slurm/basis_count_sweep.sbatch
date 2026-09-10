#!/bin/bash
# The 256-tile comparison is the wrong test of the alpha basis: E0M3 costs a
# format bit per elected tile, the alpha choice costs nothing. Sweep the count
# for every basis, re-electing from the frozen scores of job 337178.
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=basis_count
#SBATCH --array=0-5%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
SRC="${1:?source array job id}"

"$PYTHON" tests/test_basis.py

MODELS=(qwen4b qwen4b qwen4b llama8b llama8b llama8b)
BASES=(e0m3 alpha type_pure e0m3 alpha type_pure)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
BASIS="${BASES[$SLURM_ARRAY_TASK_ID]}"
echo "=== SWEEP model=$MODEL basis=$BASIS $(date -Iseconds) ==="

"$PYTHON" run_basis_count_sweep.py \
  --calibration "results/basis_ablation/${SRC}_${MODEL}_${BASIS}/calibration" \
  --out "results/basis_ablation/sweep_${SLURM_ARRAY_JOB_ID}_${MODEL}_${BASIS}"

echo "=== done model=$MODEL basis=$BASIS $(date -Iseconds) ==="

#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=math_code_completed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
"$PYTHON" tests/test_math_code_report.py
"$PYTHON" analyze_math_code_completed.py --reports \
  results/math_code_adaptive/evaluation_333786_qwen4b/report.json \
  results/math_code_adaptive/evaluation_333786_llama8b/report.json \
  --out results/math_code_adaptive/completed_small_333786

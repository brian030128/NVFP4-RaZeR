#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=math_code_curvature
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --mem=12G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
/home/u4320956/beam_engine-v3/.venv/bin/python diagnose_math_code_curvature.py --calibrations \
  results/math_code_adaptive/calibration_333779_qwen4b \
  results/math_code_adaptive/calibration_333779_llama8b \
  results/math_code_adaptive/calibration_333787_qwen27b \
  --out results/math_code_adaptive/summary_333786_333788

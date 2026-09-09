#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=calibration_check
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
"$PYTHON" -m py_compile calibration_sensitivity.py prepare_calibration_sensitivity.py \
  run_calibration_sensitivity.py summarize_calibration_sensitivity.py build_pooled_performance_tables.py \
  merge_calibration_sensitivity.py
"$PYTHON" tests/test_calibration_sensitivity_report.py
"$PYTHON" tests/test_merge_calibration_sensitivity.py

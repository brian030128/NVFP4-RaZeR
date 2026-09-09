#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=calibration_summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
export MPLCONFIGDIR="$HF_HOME/matplotlib"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
"$PYTHON" tests/test_calibration_sensitivity_report.py
"$PYTHON" tests/test_merge_calibration_sensitivity.py
"$PYTHON" merge_calibration_sensitivity.py --parts \
  "results/calibration_sensitivity/model_${2:?27B array job ID}_qwen27b_c4/report.json" \
  "results/calibration_sensitivity/model_${2}_qwen27b_wiki/report.json" \
  "results/calibration_sensitivity/model_${2}_qwen27b_transfer/report.json" \
  --out "results/calibration_sensitivity/model_${2}_qwen27b"
"$PYTHON" summarize_calibration_sensitivity.py --reports \
  "results/calibration_sensitivity/model_${1:?small model job ID}_qwen4b/report.json" \
  "results/calibration_sensitivity/model_${1}_llama8b/report.json" \
  "results/calibration_sensitivity/model_${2:?27B job ID}_qwen27b/report.json" \
  --out "results/calibration_sensitivity/summary_${1}_${2}" --update-report
"$PYTHON" build_pooled_performance_tables.py \
  --calibration-study "results/calibration_sensitivity/summary_${1}_${2}/summary.json"

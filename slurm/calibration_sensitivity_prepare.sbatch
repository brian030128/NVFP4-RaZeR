#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=calibration_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
"$PYTHON" tests/test_calibration_sensitivity.py
"$PYTHON" tests/test_causal_four_over_six.py
"$PYTHON" tests/test_wiki_frozen.py
for model in qwen4b llama8b qwen27b; do
  "$PYTHON" prepare_calibration_sensitivity.py --model "$model" \
    --out "results/calibration_sensitivity/prepared_${SLURM_JOB_ID}_$model"
done

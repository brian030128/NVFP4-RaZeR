#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=c4_frozen_summary
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
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
"$PYTHON" tests/test_c4_frozen.py
"$PYTHON" summarize_c4_frozen.py --job "${1:?job ID required}"
"$PYTHON" summarize_transfer_rule.py --c4-job "$1"

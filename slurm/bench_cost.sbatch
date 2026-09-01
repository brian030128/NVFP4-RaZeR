#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_cost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
export PYTHONUNBUFFERED=1
mkdir -p results/cost slurm/logs
for S in 2048 512 1; do
  echo "############ seq=$S"
  "$PYTHON" bench_reorder_cost.py --seq "$S"
done 2>&1 | tee results/cost/reorder_cost.log

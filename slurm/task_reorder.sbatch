#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=task_reorder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Usage: sbatch slurm/task_reorder.sbatch --scores DIR --out NEW_DIR [search options]
# CPU search requests one GPU because this cluster has no CPU-only allocation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/$USER/task_reorder_$SLURM_JOB_ID/pycache"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
"$PYTHON" run_task_reorder.py --threads 8 "$@"

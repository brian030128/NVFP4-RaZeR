#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=task_reorder_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/$USER/task_reorder_tests_$SLURM_JOB_ID/pycache"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
"$PYTHON" -m py_compile quantize/task_reorder.py run_task_reorder.py run_math_code_calibration.py
"$PYTHON" tests/test_task_reorder.py

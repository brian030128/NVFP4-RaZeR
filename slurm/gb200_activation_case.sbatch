#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_actcase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=24G
#SBATCH --time=00:03:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON=/home/u4320956/.cache/mixfp4-model-runtime/venv/bin/python
source slurm/task_reorder_env.sh
export HF_HUB_OFFLINE=1
"$PYTHON" scripts/capture_native_activation_case.py "/work/u4320956/b200/activation_case_$SLURM_JOB_ID"

#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=prepare_windows
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Freeze the published evaluation windows for the native runner. Runs in the
# ordinary environment, which has `datasets`; the native venv does not and is
# shared with the gated latency jobs, so it is left alone.
#
#   sbatch slurm/prepare_published_windows.sbatch OUT_PT
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL="${REORDER_MODEL:-llama8b}"
source slurm/task_reorder_env.sh
"$PYTHON" scripts/prepare_published_windows.py --out "${1:?Usage: sbatch ... OUT_PT}"

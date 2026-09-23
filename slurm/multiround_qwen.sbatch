#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --job-name=mr_qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H200:2
#SBATCH --mem=380G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Usage: sbatch slurm/multiround_qwen.sbatch UNIT OBJECTIVE OUT
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/qwen27b/bin/python MODEL=qwen27b
source slurm/task_reorder_env.sh
"$PYTHON" run_multiround.py --model qwen27b --unit "${1:?unit}" --objective "${2:?objective}" --out "${3:?out}" \
    --budget-hours 9

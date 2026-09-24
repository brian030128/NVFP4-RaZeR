#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --job-name=zs_maps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=190G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Usage: sbatch slurm/zeroshot_maps.sbatch MODEL OUT [--map LABEL=PATH:ROWS ...] [--skip-baseline]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
MODEL="${1:?model}"
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/$MODEL/bin/python
source slurm/task_reorder_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYTHON" run_zeroshot_maps.py --model "$MODEL" --out "${2:?out}" --samples-dir "$2/samples" "${@:3}"

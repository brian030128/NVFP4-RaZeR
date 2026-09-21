#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=reorder_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load miniconda3/26.1.1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PIP_CACHE_DIR="$PWD/.joblib/pip-cache"
export PYTHONPYCACHEPREFIX="/tmp/$USER/reorder_setup_$SLURM_JOB_ID/pycache"
mkdir -p .joblib
python -m venv .joblib/bootstrap
.joblib/bootstrap/bin/pip install 'uv==0.8.22'
UV="$PWD/.joblib/bootstrap/bin/uv"
export UV_CACHE_DIR="$PWD/.joblib/uv-cache"
for model in llama8b qwen27b; do
    "$UV" venv --python 3.11 ".joblib/$model"
    if [[ "$model" == llama8b ]]; then
        version=4.57.3
    else
        version=5.16.1
    fi
    "$UV" pip install --python ".joblib/$model/bin/python" \
        'torch==2.9.0' "transformers==$version" 'datasets==4.4.1' \
        'accelerate==1.7.0' 'numpy==2.2.6' 'scipy==1.15.3' \
        'loguru==0.7.3' sentencepiece protobuf
    ".joblib/$model/bin/python" -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name())'
done
export PYTHONPATH="$PWD"
.joblib/llama8b/bin/python tests/test_task_reorder.py

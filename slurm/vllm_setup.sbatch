#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=vllm_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Build the serving venv once, on a worker.
#
# Terminal-Bench needs the model behind an OpenAI-compatible endpoint, and nothing on this
# machine has vLLM. Unpacking those wheels is several GB of file writes -- well past what the
# login-node guard allows -- so it is a job. The venv lands on /home because /home is visible
# from every node, which is the whole reason the repo's other venvs live there.
#
#   sbatch slurm/vllm_setup.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs

VENV=${VENV:-/home/u4320956/NVFP4-RaZeR/.venv-vllm}
# The version matters. Unpinned, uv resolves vllm 0.29, which brings torch 2.13.0+cu130 --
# a CUDA *major* version this cluster's driver cannot run:
#   RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
# cu12x is fine on a 12.4 driver through minor-version compatibility (the repo's other venv
# runs torch 2.9.0+cu128), so pin a vllm whose wheels are built against cu128.
VLLM_SPEC=${VLLM_SPEC:-vllm==0.11.0}
UV=/home/u4320956/.local/bin/uv
export UV_CACHE_DIR="/tmp/$USER/uvcache_$SLURM_JOB_ID"
mkdir -p "$UV_CACHE_DIR"
trap 'rm -rf "$UV_CACHE_DIR"' EXIT

echo "=== vllm setup -> $VENV  spec=$VLLM_SPEC  $(date -Iseconds) on $(hostname) ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader
"$UV" venv --python 3.12 "$VENV"
"$UV" pip install --python "$VENV/bin/python" "$VLLM_SPEC"
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
# Fail loudly here rather than at serve time if the wheel is for the wrong CUDA major version.
"$VENV/bin/python" -c "
import torch, sys
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
sys.exit(0 if torch.cuda.is_available() else 1)
" || { echo 'FATAL: torch cannot see the GPU -- wrong CUDA build for this driver'; exit 1; }
echo "=== done $(date -Iseconds) ==="

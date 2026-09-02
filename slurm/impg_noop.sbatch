#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=impg_noop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# CPU test suites plus the diagnostics summary. Everything here is CPU-only; the GPU is requested
# solely because this cluster rations cores at 12 per GPU and has no CPU-only partition.
#
#   sbatch slurm/impg_noop.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
LIBDIR="$PWD/.joblib"; export PYTHONPATH="$LIBDIR:${PYTHONPATH:-}"
export HF_HOME="/work/$USER/hf"; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p slurm/logs

echo "############ tests/test_imp_gran.py ############"
"$PYTHON" check_impg_noop.py --model_name "${1:-llama-3.1-8b-local}"
echo "imp_gran tests exit=$?"

echo "=== done $(date -Iseconds) ==="

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
UV=/home/u4320956/.local/bin/uv
export UV_CACHE_DIR="/tmp/$USER/uvcache_$SLURM_JOB_ID"
mkdir -p "$UV_CACHE_DIR"
trap 'rm -rf "$UV_CACHE_DIR"' EXIT

echo "=== vllm setup -> $VENV  $(date -Iseconds) on $(hostname) ==="
"$UV" venv --python 3.12 "$VENV"
"$UV" pip install --python "$VENV/bin/python" vllm
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
echo "=== done $(date -Iseconds) ==="

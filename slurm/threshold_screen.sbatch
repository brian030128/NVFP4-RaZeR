#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=thresh_screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
"$PYTHON" run_threshold_screen.py --stage-root /home/u4320956/NVFP4-RaZeR \
  --out results/threshold_screen
echo "=== done $(date -Iseconds) ==="

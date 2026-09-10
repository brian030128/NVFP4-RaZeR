#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=basis_summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
"$PYTHON" summarize_basis_ablation.py --root results/basis_ablation \
  --job "${1:?array job id}" --out "results/basis_ablation/summary_${1}"
echo "=== done $(date -Iseconds) ==="

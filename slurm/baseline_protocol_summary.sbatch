#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=baseline_protocol_summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
AUDIT_JOB="${1:?audit job ID}"
shift
/home/u4320956/beam_engine-v3/.venv/bin/python summarize_baseline_protocol_audit.py --job "$AUDIT_JOB" "$@"

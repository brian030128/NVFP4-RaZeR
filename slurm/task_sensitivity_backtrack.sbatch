#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --job-name=sensitivity_backtrack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --array=0-1%2
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
SEED=$((20260906 + SLURM_ARRAY_TASK_ID))
bash slurm/task_sensitivity.sbatch llama-3.1-8b-local --policy-set sparse --skip-interventions \
  --backtrack-proposal --fit 64 --seed "$SEED" \
  --resume-scores "results/task_sensitivity_morecalib/llama-3.1-8b-local_seed$SEED/scores.pt" \
  --out "results/task_sensitivity_backtrack/llama-3.1-8b-local_seed$SEED"

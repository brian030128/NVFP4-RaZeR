#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --job-name=sensitivity_extend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --array=0-15%2
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODELS=(llama-3.2-3b llama-2-7b qwen3-8b llama-3.1-8b-ins-local llama-3.2-3b-ins llama-2-13b qwen3-14b llama-3.2-1b-ins-local)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % 8))]}
SEED=$((20260909 + SLURM_ARRAY_TASK_ID / 8))
bash slurm/task_sensitivity.sbatch "$MODEL" --policy-set sparse --skip-interventions \
  --backtrack-proposal --fit 64 --seed "$SEED" \
  --out "results/task_sensitivity_extension/${MODEL}_seed${SEED}"

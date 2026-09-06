#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --job-name=sensitivity_repeat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --array=0-1%2
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODELS=(qwen3-4b llama-3.1-8b-local)
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID:?}]}
bash slurm/task_sensitivity.sbatch "$MODEL" --policy-set sparse --skip-interventions \
  --seed 20260907 --out "results/task_sensitivity_replication/$MODEL" \
  --final-policies baseline,gradient_trust_0p1,hess_impg16_h10

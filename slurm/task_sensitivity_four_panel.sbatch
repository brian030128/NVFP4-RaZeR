#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --job-name=sensitivity_four_panel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=06:00:00
#SBATCH --mem=200G
#SBATCH --array=0-3%2
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODELS=(qwen3-4b llama-3.1-8b-local qwen3-8b llama-3.1-8b-ins-local)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
bash slurm/task_sensitivity.sbatch "$MODEL" --policy-set sparse --skip-interventions \
  --backtrack-proposal --fit 64 --seed 20260912 --weight-baseline nvfp4_4over6 \
  --out "results/task_sensitivity_four_panel/${MODEL}"

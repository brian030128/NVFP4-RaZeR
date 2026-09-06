#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --job-name=sensitivity_panel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --array=0-1%2
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# The array concurrency limit enforces two one-H100 jobs at most. Submit this
# only after earlier research jobs finish, or with afterany dependencies on them.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODELS=(qwen3-4b llama-3.1-8b-local qwen3-8b llama-3.2-1b-ins-local qwen3-14b llama-3.1-8b-ins-local)
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID:?}]}
ARGS=(--policy-set sparse --skip-interventions --out "results/task_sensitivity_sparse/$MODEL")
if [[ "$SLURM_ARRAY_TASK_ID" -lt 2 ]]; then
    ARGS+=(--resume-scores "results/task_sensitivity/$MODEL/scores.pt")
fi
bash slurm/task_sensitivity.sbatch "$MODEL" "${ARGS[@]}"

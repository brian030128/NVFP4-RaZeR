#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=qpot_rule
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Usage: sbatch slurm/potential_rule.sbatch OBJECTIVE:K:TILE_ROWS OUTROOT
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/qwen27b/bin/python MODEL=qwen27b
source slurm/task_reorder_env.sh
RULE="${1:?rule}"
"$PYTHON" run_mapped_gptq.py --model qwen27b --policy rtn_rule --rule "$RULE" --out "${2:?outroot}/rule_${RULE//:/_}"

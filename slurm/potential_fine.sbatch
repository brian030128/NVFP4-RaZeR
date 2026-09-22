#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=pot_fine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Usage: sbatch slurm/potential_fine.sbatch {both,ce,kl} OUTROOT
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/llama8b/bin/python MODEL=llama8b
source slurm/task_reorder_env.sh
RULE="${1:?both, ce or kl}"
"$PYTHON" run_mapped_gptq.py --policy rtn_fine1x16 --fine-rule "$RULE" --fine-dir "${3:-/work/u4320956/mixfp4_potential/llama8b_fine1x16_k/fine_masks}" --out "${2:?outroot}/fine1x16_$RULE"

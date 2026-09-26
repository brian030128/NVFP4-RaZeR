#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=8gpus
#SBATCH --job-name=reason_maps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=190G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# RaZeR CoT benchmark (gsm8k_llama, mmlu_cot_llama; chat template, few-shot as multi-turn)
# on Llama-3.1-8B-Instruct with fake-quantized policies, via lm-eval 0.4.9.2.
# Usage: sbatch slurm/reasoning_maps.sbatch OUT [run_zeroshot_maps.py args ...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
export REORDER_HF_HOME=$MAIN/.joblib/huggingface PYTHON=$MAIN/.joblib/llama8b/bin/python MODEL=llama8b
source slurm/task_reorder_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$MAIN/.joblib/lmeval0492_x86:$PYTHONPATH"
"$PYTHON" run_zeroshot_maps.py --model llama8b_ins --chat --out "${1:?out}" --samples-dir "$1/samples" "${@:2}"

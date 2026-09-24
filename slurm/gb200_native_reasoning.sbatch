#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_reason
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Native SM100 RaZeR CoT benchmark on Llama-3.1-8B-Instruct (scripts/native_reasoning_llama.py).
# gb200-dev caps jobs at 2 h, so long tasks are split with --shard i/N.
# Usage: sbatch slurm/gb200_native_reasoning.sbatch OUT --policy ... [--map ...] [--tasks ...] [--shard i/N]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MAIN=/home/u4320956/NVFP4-RaZeR
RUNTIME=/home/u4320956/.cache/mixfp4-model-runtime
PYTHON=$RUNTIME/venv/bin/python
source slurm/task_reorder_env.sh
export HF_HOME=$MAIN/.joblib/huggingface HF_HUB_OFFLINE=1
DEPS=$MAIN/.joblib/lmeval0492_arm
if [[ ! -d "$DEPS/lm_eval" ]]; then
  "$RUNTIME/bootstrap/bin/uv" pip install --python "$PYTHON" --target "$DEPS" \
    lm_eval==0.4.9.2 "huggingface_hub>=0.34,<1.0" langdetect immutabledict
  rm -rf "$DEPS"/{transformers,tokenizers,torch,torchvision,torchaudio,triton,nvidia,numpy}
  rm -rf "$DEPS"/{transformers,tokenizers,torch,torchvision,torchaudio,triton,numpy}-*.dist-info
fi
export PYTHONPATH="$PWD/scripts:$PWD:$DEPS:${PYTHONPATH:-}"
"$PYTHON" scripts/native_reasoning_llama.py \
  --lib "$RUNTIME/build/libmixfp4_model.so" \
  --kernel-gate "$PWD/results/task_reorder/full_model_20260920/kernel_406633" \
  --out "${1:?out}" "${@:2}"

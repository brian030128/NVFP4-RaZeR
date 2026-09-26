#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_reason_qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Native SM100 RaZeR CoT benchmark on Qwen3.8-27B (scripts/native_reasoning_llama.py --model qwen27b).
# Qwen pins transformers 5.16.1 (its calibration origin), so an ARM overlay with the same
# transformers/huggingface_hub/tokenizers/safetensors versions as .joblib/qwen27b shadows the
# runtime venv's transformers 4.57.3; lm-eval 0.4.9.2 comes from the Llama reasoning overlay.
# gb200-dev caps jobs at 2 h, so GSM8K is split with --shard i/N.
# Usage: sbatch slurm/gb200_native_reasoning_qwen.sbatch OUT --policy ... [--map ...] [--shard i/N] [--limit N]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# The 27B model needs most of the GPU; stop early if another process already holds it.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "GPU $(hostname) memory in use before start: ${USED} MiB"
(( USED < 2048 )) || { echo "GPU already occupied (${USED} MiB) on $(hostname)"; exit 3; }
MAIN=/home/u4320956/NVFP4-RaZeR
RUNTIME=/home/u4320956/.cache/mixfp4-model-runtime
PYTHON=$RUNTIME/venv/bin/python
source slurm/task_reorder_env.sh
export HF_HOME=$MAIN/.joblib/huggingface HF_HUB_OFFLINE=1
DEPS=$MAIN/.joblib/lmeval0492_arm
TF5=$MAIN/.joblib/qwen_tf5_arm
(
  flock 9
  if [[ ! -f "$TF5/.complete" ]]; then
    rm -rf "$TF5"
    "$RUNTIME/bootstrap/bin/uv" pip install --python "$PYTHON" --target "$TF5" \
      transformers==5.16.1 huggingface_hub==1.32.0 tokenizers==0.23.2 safetensors==0.8.0
    rm -rf "$TF5"/{torch,torchvision,torchaudio,triton,nvidia,numpy}
    rm -rf "$TF5"/{torch,torchvision,torchaudio,triton,nvidia,numpy}-*.dist-info
    touch "$TF5/.complete"
  fi
) 9>"$MAIN/.joblib/qwen_tf5_arm.lock"
[[ -d "$DEPS/lm_eval" ]] || { echo "missing $DEPS (run slurm/gb200_native_reasoning.sbatch once)"; exit 1; }
export PYTHONPATH="$PWD/scripts:$PWD:$TF5:$DEPS:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
"$PYTHON" scripts/native_reasoning_llama.py --model qwen27b \
  --lib "$RUNTIME/build/libmixfp4_model.so" \
  --kernel-gate "$PWD/results/task_reorder/full_model_20260920/kernel_406633" \
  --tasks gsm8k_llama --out "${1:?out}" "${@:2}"

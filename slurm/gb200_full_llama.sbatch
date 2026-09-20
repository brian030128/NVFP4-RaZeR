#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_full_llama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON=/home/u4320956/.cache/mixfp4-model-runtime/venv/bin/python
source slurm/task_reorder_env.sh
export HF_HUB_OFFLINE=1
SNAP="$PWD/results/task_reorder/full_model_20260920/source_$SLURM_JOB_ID"
mkdir -p "$SNAP/quantize"
cp scripts/benchmark_native_llama.py scripts/native_model_runtime.py "$SNAP/"
cp quantize/__init__.py quantize/quantizer.py quantize/quant_config.py quantize/utils.py quantize/task_reorder.py quantize/importance.py "$SNAP/quantize/"
"$PYTHON" "$SNAP/benchmark_native_llama.py" --lib /home/u4320956/.cache/mixfp4-model-runtime/build/libmixfp4_model.so --kernel-gate "${NATIVE_KERNEL_GATE:?Set passed kernel gate directory}" --out "$PWD/results/task_reorder/full_model_20260920/llama_$SLURM_JOB_ID" "$@"

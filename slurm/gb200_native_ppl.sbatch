#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_native_ppl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Native-kernel perplexity on the published windows. Needs SM100, so this runs on
# GB200 rather than the usual H200 partitions. gb200-dev is the only GB200
# partition this account may use; gb200-r2 is rejected as an invalid
# account/partition combination. Its two-hour cap bounds a run to roughly one
# policy over the full 422 windows, so sweep policies as separate jobs.
#
#   NATIVE_KERNEL_GATE=<passed kernel dir> \
#     sbatch slurm/gb200_native_ppl.sbatch [extra args]
#
# Pass --limit-windows N first as a smoke run: it exercises packing, the window
# build and a few native forwards before committing to the full sweep.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON=/home/u4320956/.cache/mixfp4-model-runtime/venv/bin/python
source slurm/task_reorder_env.sh
export HF_HUB_OFFLINE=1
SNAP="$PWD/results/task_reorder/native_ppl_20260921/source_$SLURM_JOB_ID"
mkdir -p "$SNAP/quantize"
cp scripts/native_ppl_llama.py scripts/benchmark_native_llama.py \
   scripts/native_model_runtime.py run_baseline_protocol_audit.py "$SNAP/"
cp quantize/__init__.py quantize/quantizer.py quantize/quant_config.py quantize/utils.py \
   quantize/task_reorder.py quantize/importance.py "$SNAP/quantize/"
PYTHONPATH="$SNAP:$PWD:${PYTHONPATH:-}" "$PYTHON" "$SNAP/native_ppl_llama.py" \
  --lib /home/u4320956/.cache/mixfp4-model-runtime/build/libmixfp4_model.so \
  --kernel-gate "${NATIVE_KERNEL_GATE:?Set a passed kernel gate directory}" \
  --out "$PWD/results/task_reorder/native_ppl_20260921/run_$SLURM_JOB_ID" "$@"

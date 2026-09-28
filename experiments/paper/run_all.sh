#!/bin/bash
# The paper experiments, steps 00-07 in order (docs/PAPER_EXPERIMENTS.md). Stops at the first failing step. Every
# step skips what it has already finished, so running this again continues where it stopped.
#
#   experiments/paper/run_all.sh [--smoke] [--models llama8b,mistral7b,phi4,qwen27b] [--units 8x64,16x64,256x64]
#
# The arguments go to every step. --policies differs per step: pass it to a single step instead.
# Paths: the PAPER_* environment variables (experiments/paper/paper_common.py).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PAPER_PYTHON:-/home/dev/.conda/envs/n16k64/bin/python}
for S in 00_check 01_calibrate 02_export 03_ppl 04_downstream 05_prefill_latency 06_gemm_latency 07_tables; do
  echo "=== $S $* ($(date -u +%FT%TZ))"
  "$PY" "$HERE/$S.py" "$@" || { echo "STOP: $S failed ($(date -u +%FT%TZ))"; exit 1; }
done
echo "=== all steps done ($(date -u +%FT%TZ))"

#!/bin/bash
# Run the remaining per-model reproduction pipelines one after another on the single GPU.
#   qwen27b  after the qwen4b pipeline has finished (54 GB model: nothing else on the GPU)
#   mistral7b, phi4  once their pinned downloads are recorded in the fetch log
set -uo pipefail
RUNS=/home/dev/n16k64_campaign/runs
FETCH=/home/dev/n16k64_campaign/provenance/fetch.log
PIPE=/home/dev/NVFP4-RaZeR-n16k64/repro_local/pipeline.sh

until grep -q "PIPELINE qwen4b DONE\|rc=[1-9]" "$RUNS/pipeline_qwen4b.log" 2>/dev/null; do sleep 60; done
echo "qwen4b pipeline finished $(date -u +%FT%TZ)"

$PIPE qwen27b --moments-device cpu > "$RUNS/pipeline_qwen27b.log" 2>&1
echo "qwen27b pipeline rc=$? $(date -u +%FT%TZ)"

for M in mistral7b phi4; do
  until grep -q "^MODEL $M ok=True" "$FETCH"; do
    if grep -q "^MODEL $M FAILED\|^MODEL $M ok=False" "$FETCH"; then echo "download of $M failed"; continue 2; fi
    sleep 60
  done
  $PIPE $M > "$RUNS/pipeline_$M.log" 2>&1
  echo "$M pipeline rc=$? $(date -u +%FT%TZ)"
done
echo "ALL PIPELINES DONE $(date -u +%FT%TZ)"

#!/bin/bash
# results/inference_memory/PROTOCOL.md: one fresh process per (model, policy); the default allocator; nothing else on the GPU.
D=/home/dev/n16k64_campaign/inference_memory
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=$D/commands.log
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1
export PYTHONPATH=$REPO/sm120:$REPO PYTHONDONTWRITEBYTECODE=1
unset PYTORCH_CUDA_ALLOC_CONF
cd $REPO
mkdir -p $D/runs $D/logs
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
for M in llama8b mistral7b phi4 qwen27b; do
  for P in bf16 nvfp4@stock_wA fo6@stock_wA fo6@stock_wB tc-8x64 tc-16x64 tc-256x64; do
    log "START ${M}_$P"
    $PY results/inference_memory/measure_memory.py --model $M --policy $P --out $D/runs/$M/$P.json < /dev/null > $D/logs/${M}_$P.log 2>&1
    rc=$?; log "END ${M}_$P rc=$rc"
    [ $rc -eq 0 ] || log "FAIL (logged, continuing): ${M}_$P"
  done
done
log "MEMORY DONE"

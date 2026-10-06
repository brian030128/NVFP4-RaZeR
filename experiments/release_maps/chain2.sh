#!/bin/bash
# Gap after batch 1: smoke runs of batch 1, the mistral-7b-base routing check, then batch 2. One GPU job at a time.
set -u
B=/home/dev/n16k64_campaign/fqrel; W=/home/dev/n16k64_campaign/fqopt/wt; PY=/home/dev/.conda/envs/n16k64/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> $B/queue.log; }
cd $B && PYTHONDONTWRITEBYTECODE=1 $PY release.py --models qwen3-1.7b,mistral-7b --ppl-only > $B/driver_1b.out 2>&1
mkdir -p $B/validation
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy before validate_mistral-7b-base_8x64"; exit 1; }
log "START validate_mistral-7b-base_8x64 (--require-reproduction, the paper's settings)"
( cd $W && env -u PYTHONPATH -u SM120_BUILD_DIR -u PYTORCH_CUDA_ALLOC_CONF PYTHONDONTWRITEBYTECODE=1 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=0 \
  $PY -m calibration.train_map --model mistral-7b-base --unit 8x64 --require-reproduction --data-root $B/tmopt_data \
  --out $B/validation/mistral-7b-base_8x64 ) < /dev/null > $B/logs/validate_mistral-7b-base_8x64.log 2>&1
rc=$?; log "END validate_mistral-7b-base_8x64 rc=$rc"
MODELS=mistral-7b-base,llama3.1-8b,qwen3-8b,phi4-14b,nemotron-nano-9b-v2,qwen3.8-27b
[ $rc -eq 0 ] || { MODELS=llama3.1-8b,qwen3-8b,phi4-14b,nemotron-nano-9b-v2,qwen3.8-27b; log "STOP mistral-7b-base: the routing check failed; batch 2 runs without it"; }
cd $B && PYTHONDONTWRITEBYTECODE=1 $PY release.py --models $MODELS > $B/driver_2.out 2>&1
log "END chain2 rc=$?"

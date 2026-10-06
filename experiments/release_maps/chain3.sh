#!/bin/bash
# After batch 2 (chain2.sh): option A's measurement re-runs, then the one-time data preparation of every model,
# measured from a fresh data root. One job at a time.
set -u
B=/home/dev/n16k64_campaign/fqrel; W=/home/dev/n16k64_campaign/fqopt/wt; PY=/home/dev/.conda/envs/n16k64/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> $B/queue.log; }
until grep -q "END chain2" $B/queue.log; do sleep 60; done
log "BEGIN chain3"
RUNS=""
for m in qwen3-1.7b mistral-7b mistral-7b-base llama3.1-8b qwen3-8b phi4-14b; do
  for u in 8x64 16x64 256x64; do RUNS="$RUNS,$m:$u"; done; done
RUNS="${RUNS#,},nemotron-nano-9b-v2:8x64"
cd $B && PYTHONDONTWRITEBYTECODE=1 $PY remeasure.py --runs $RUNS > $B/remeasure.out 2>&1
mkdir -p $B/prep_measure
for m in qwen3-1.7b mistral-7b mistral-7b-base llama3.1-8b qwen3-8b phi4-14b nemotron-nano-9b-v2 qwen3.8-27b; do
  until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
  log "START prep $m"
  ( cd $W && env -u PYTHONPATH -u SM120_BUILD_DIR -u PYTORCH_CUDA_ALLOC_CONF PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=0 \
    $PY $B/measure_vmhwm.py --out $B/prep_measure/$m.measure.json --log $B/logs/prep_$m.log --cwd $W -- \
    $PY $B/prep_run.py --model $m --root $B/prep_measure/$m ) < /dev/null
  rc=$?; log "END prep $m rc=$rc"
  r=$(cd $W && PYTHONDONTWRITEBYTECODE=1 $PY $B/compare_prep.py --model $m --fresh $B/prep_measure/$m --out $B/prep_measure/$m.compare.json 2>&1 | tail -1)
  case "$r" in IDENTICAL*) log "OK prep $r";; *) log "MISMATCH prep $r";; esac
done
log "END chain3"

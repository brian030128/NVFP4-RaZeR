#!/bin/bash
# R2 (results/tm_opt/PROTOCOL_QR.md deviation 3): the benchmark stage of queue_latency.sh, restarted after its first
# process failed (bench_latency.py wrote a trace before creating the output directory; fixed). The builds, checks
# (DECODE=1) and exports of queue_latency.sh are done and are not repeated; the rounds and policy orders are the same
# (same seed). Qwen3.8-27B runs without --decode: the SM120 decode functions do not support its hybrid cache.
S=/home/dev/n16k64_campaign/sm120_bench
O=$S/results_r2
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
G=$REPO/results/tm_opt/latency
LOG=$S/commands_r2.log
QLOG=/home/dev/n16k64_campaign/tm_opt/commands_q.log
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1
export PYTHONPATH=$S/sm120:$REPO PYTHONDONTWRITEBYTECODE=1
unset PYTORCH_CUDA_ALLOC_CONF
cd $S
mkdir -p $O/logs $O/bench
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $O/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
MODELS="llama8b mistral7b phi4 qwen27b"
DECODE=1
log "BENCH STAGE (restart): DECODE=$DECODE, Qwen without decode"

# label artifact kernel(prefill-only) kernel(with decode)
POLICIES="bf16 - - -
nvfp4@stock_wA nvfp4 stock_wA auto_stock
nvfp4@stock_wB nvfp4 stock_wB stock_wB
fo6@stock_wA fo6 stock_wA auto_stock
fo6@stock_wB fo6 stock_wB stock_wB
tmopt-8x64@n8k64_wB tmopt_8x64 n8k64_wB n8k64_wB
tmopt-16x64@n16k64_wA tmopt_16x64 n16k64_wA auto
tmopt-256x64@n16k64_wA tmopt_256x64 n16k64_wA auto
tc-8x64@n8k64_wB tc_8x64 n8k64_wB n8k64_wB
tc-16x64@n16k64_wA tc_16x64 n16k64_wA auto
tc-256x64@n16k64_wA tc_256x64 n16k64_wA auto"
SEED=20260926
for M in $MODELS; do
  ITERS=3; [ $M = qwen27b ] && ITERS=1
  EXTRA=""; [ $DECODE -eq 1 ] && [ $M != qwen27b ] && EXTRA="--decode"
  $PY -c "
import json, random, sys
labels = [l.split()[0] for l in sys.stdin.read().strip().splitlines()]
rng = random.Random($SEED + '$MODELS'.split().index('$M'))
order = []
for r in range(1, 6):
    rng.shuffle(labels)
    order.append(dict(round=r, order=list(labels)))
json.dump(dict(model='$M', seed=$SEED, order=order), open('$O/bench/order_$M.json', 'w'), indent=1)
for o in order:
    for l in o['order']:
        print(o['round'], l)
" <<< "$POLICIES" > $O/bench/order_$M.txt
  while read R L; do
    set -- $(grep "^$L " <<< "$POLICIES")
    if [ "$2" = "-" ]; then POL=""; else POL="--artifact $O/artifacts/${M}_$2 --kernel $([ $DECODE -eq 1 ] && echo $4 || echo $3)"; fi
    run bench_${M}_${L}_r$R $PY $G/bench_latency.py --model $M --label $L $POL --round $R --profile-iters $ITERS $EXTRA \
        --out $O/bench/$M/${L}_r$R.json
  done < $O/bench/order_$M.txt
  log "R2 MODEL DONE $M"
done
log "R2 DONE"

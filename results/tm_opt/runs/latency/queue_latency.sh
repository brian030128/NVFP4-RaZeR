#!/bin/bash
# R2 (results/tm_opt/PROTOCOL_QR.md): prefill latency (and batch-1 decode if the narrow-tile builds pass their
# checks) of BF16, NVFP4, FourOverSix and the TM-OPT / TM-OPT+TC maps on the SM120 deployment kernels, four models.
# CPU work (map conversion) starts once the Qwen runs are done; every GPU step waits for "PART Q DONE".
# One process per (model, policy, round); 5 rounds, the policy order shuffled per round with a recorded seed.
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
log "WAIT for the Qwen runs (map conversion) and then Part Q (GPU)"
until grep -qE "QWEN RUNS DONE|STOP" $QLOG; do sleep 60; done
grep -q "STOP" $QLOG && { log "STOP: the Part Q queue stopped"; exit 1; }
for M in $MODELS; do run convert_$M $PY $G/convert_maps.py $M $O/maps; done
until grep -qE "PART Q DONE|STOP" $QLOG; do sleep 60; done
grep -q "STOP" $QLOG && { log "STOP: the Part Q queue stopped"; exit 1; }
log "GPU idle: R2 starts"

# narrow-tile builds for decode, with the self-test gate (mixed) and tests/test_select.py (bitwise equal widths,
# batch invariance); decode is measured only if every step passes and no test is skipped
DECODE=1
for C in n16k64_wA_n64 n16k64_wA_n32 n16k64_wA_n16 stock_wA_n64 stock_wA_n32 stock_wA_n16; do
  log "START build_$C"; $PY sm120/build.py --config $C --selftest > $O/logs/build_$C.log 2>&1; rc=$?; log "END build_$C rc=$rc"
  [ $rc -eq 0 ] || DECODE=0
done
if [ $DECODE -eq 1 ]; then
  log "START pytest_select"; $PY -m pytest sm120/tests/test_select.py -q -rs -p no:cacheprovider > $O/logs/pytest_select.log 2>&1; rc=$?
  log "END pytest_select rc=$rc"
  { [ $rc -eq 0 ] && ! grep -qi "skipped" $O/logs/pytest_select.log; } || DECODE=0
fi
log "DECODE=$DECODE (1: the narrow-tile builds passed their checks; decode measured with the width-selecting kernel sets)"

for M in $MODELS; do
  run export_${M}_nvfp4 $PY sm120/eval/export_artifact.py --model $M --kind nvfp4 --out $O/artifacts/${M}_nvfp4
  run export_${M}_fo6 $PY sm120/eval/export_artifact.py --model $M --kind four_over_six --out $O/artifacts/${M}_fo6
  for X in tmopt_8x64 tmopt_16x64 tmopt_256x64 tc_8x64 tc_16x64 tc_256x64; do
    run export_${M}_$X $PY sm120/eval/export_artifact.py --model $M --map $O/maps/${M}_$X.mixfp4map --out $O/artifacts/${M}_$X
  done
done
log "EXPORTS DONE"

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
  EXTRA=""; [ $DECODE -eq 1 ] && EXTRA="--decode"
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

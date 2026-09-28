#!/bin/bash
# results/tile_table/PROTOCOL.md: the Part R prefill latencies that the table changes (Qwen3.8-27B 1x2048: its 48x5120
# linear-attention projections, 128 -> 64 in both families), measured again with the table: the same harness
# (bench_latency.py, the Part R settings for Qwen: profile iters 1, prefill 1x512,1x2048,4x2048, no decode), its six
# width-selecting policies, 5 rounds in a seeded shuffled order. The table was copied into the harness's sm120 copy
# (identical SASS to the repository's builds).
S=/home/dev/n16k64_campaign/sm120_bench
O=$S/results_r2
D=/home/dev/n16k64_campaign/tile_table
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
G=$REPO/results/tm_opt/latency
LOG=$D/commands_remeasure.log
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1
export PYTHONPATH=$S/sm120:$REPO PYTHONDONTWRITEBYTECODE=1
unset PYTORCH_CUDA_ALLOC_CONF
cd $S
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $D/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
M=qwen27b
POLICIES="nvfp4@stock_wA nvfp4 auto_stock
fo6@stock_wA fo6 auto_stock
tmopt-16x64@n16k64_wA tmopt_16x64 auto
tmopt-256x64@n16k64_wA tmopt_256x64 auto
tc-16x64@n16k64_wA tc_16x64 auto
tc-256x64@n16k64_wA tc_256x64 auto"
$PY -c "
import json, random, sys
labels = [l.split()[0] for l in sys.stdin.read().strip().splitlines()]
rng = random.Random(20260928)
order = []
for r in range(1, 6):
    rng.shuffle(labels)
    order.append(dict(round=r, order=list(labels)))
json.dump(dict(model='$M', seed=20260928, order=order), open('$D/remeasure/order_$M.json', 'w'), indent=1)
for o in order:
    for l in o['order']:
        print(o['round'], l)
" <<< "$POLICIES" > $D/remeasure/order_$M.txt
while read R L; do
  set -- $(grep "^$L " <<< "$POLICIES")
  run remeasure_${M}_${L}_r$R $PY $G/bench_latency.py --model $M --label $L --artifact $O/artifacts/${M}_$2 --kernel $3 --round $R \
      --profile-iters 1 --out $D/remeasure/$M/${L}_r$R.json
done < $D/remeasure/order_$M.txt
log "REMEASURE DONE"

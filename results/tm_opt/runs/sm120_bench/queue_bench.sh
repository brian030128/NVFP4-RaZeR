#!/bin/bash
# #3 (results/tm_opt/PROTOCOL_ITEMS.md) on the idle GPU after items #2 and #1: kernel self-tests and GEMM tests, map
# conversion, artifact export, GEMM-only and full-model prefill benchmarks; Llama-3.1-8B, then Phi-4.
S=/home/dev/n16k64_campaign/sm120_bench
O=$S/results
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
G=$REPO/results/tm_opt/gemm
LOG=$S/commands_bench.log
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1
export PYTHONPATH=$S/sm120:$REPO PYTHONDONTWRITEBYTECODE=1
unset PYTORCH_CUDA_ALLOC_CONF
cd $S
mkdir -p $O/logs
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $O/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
log "WAIT for items #2 and #1 and the kernel builds"
until grep -q "ITEM1 DONE" /home/dev/n16k64_campaign/tm_opt/commands_items.log && grep -q "END stock_wB" $S/build.log; do sleep 60; done
grep -q "STOP" /home/dev/n16k64_campaign/tm_opt/commands_items.log && { log "STOP: the items queue stopped"; exit 1; }
for C in n16k64_wA n8k64_wB stock_wA stock_wB; do grep -q "END $C rc=0" $S/build.log || { log "STOP: build of $C failed"; exit 1; }; done
log "GPU idle: #3 starts"

# kernel self-tests (PASS patched / FAIL unpatched) and the GEMM test suite, on this GPU
run selftest_n16k64_wA $PY sm120/build.py --config n16k64_wA --selftest
run selftest_n8k64_wB $PY sm120/build.py --config n8k64_wB --selftest
run pytest_gemm $PY -m pytest sm120/tests/test_gemm.py -q -p no:cacheprovider

for MOD in llama8b phi4; do
  run convert_$MOD $PY $G/convert_maps.py $MOD $O/maps
  run export_${MOD}_fo6 $PY sm120/eval/export_artifact.py --model $MOD --kind four_over_six --out $O/artifacts/${MOD}_fo6
  for X in mropt-8x64 tmopt-8x64 mropt-16x64 tmopt-16x64; do
    run export_${MOD}_$X $PY sm120/eval/export_artifact.py --model $MOD --map $O/maps/${MOD}_${X/-/_}.mixfp4map --out $O/artifacts/${MOD}_$X
  done
  run gemm_$MOD $PY $G/bench_gemm.py --model $MOD --out $O/gemm_$MOD.json
  for P in "stock-8x64 fo6 stock_wB" "fo6-8x64 fo6 n8k64_wB" "mropt-8x64 mropt-8x64 n8k64_wB" "tmopt-8x64 tmopt-8x64 n8k64_wB" \
           "stock-16x64 fo6 stock_wA" "fo6-16x64 fo6 n16k64_wA" "mropt-16x64 mropt-16x64 n16k64_wA" "tmopt-16x64 tmopt-16x64 n16k64_wA"; do
    set -- $P
    run prefill_${MOD}_$1 $PY $G/bench_prefill.py --model $MOD --artifact $O/artifacts/${MOD}_$2 --kernel $3 --label $1 --out $O/prefill_${MOD}_$1.json
  done
  log "BENCH MODEL DONE $MOD"
done
log "ITEM3 DONE"

#!/bin/bash
# #3 timing diagnostic (results/tm_opt/PROTOCOL_ITEMS.md, deviation 1), on the idle GPU after the registered #3 runs:
# (1) three GEMM timing methods in alternating shuffled order with GPU telemetry (Llama shapes, T = 2048, 8192);
# (2) the Llama prefill policies repeated twice more, in alternating orders, to measure process-to-process spread.
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
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $O/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
until grep -qE "ITEM3 DONE" $LOG; do sleep 20; done
run diag_timing_llama8b $PY $G/diagnose_timing.py --model llama8b --out $O/diagnose_timing_llama8b.json
declare -A ART=( [stock-8x64]="fo6 stock_wB" [fo6-8x64]="fo6 n8k64_wB" [mropt-8x64]="mropt-8x64 n8k64_wB" [tmopt-8x64]="tmopt-8x64 n8k64_wB"
                 [stock-16x64]="fo6 stock_wA" [fo6-16x64]="fo6 n16k64_wA" [mropt-16x64]="mropt-16x64 n16k64_wA" [tmopt-16x64]="tmopt-16x64 n16k64_wA" )
for R in 2 3; do
  if [ $R = 2 ]; then ORDER="tmopt-8x64 stock-8x64 mropt-8x64 fo6-8x64 tmopt-16x64 stock-16x64 mropt-16x64 fo6-16x64";
  else ORDER="fo6-16x64 mropt-16x64 stock-16x64 tmopt-16x64 fo6-8x64 mropt-8x64 stock-8x64 tmopt-8x64"; fi
  for P in $ORDER; do
    set -- ${ART[$P]}
    run prefill_llama8b_${P}_r$R $PY $G/bench_prefill.py --model llama8b --artifact $O/artifacts/llama8b_$1 --kernel $2 --label $P --out $O/prefill_llama8b_${P}_r$R.json
  done
done
log "DIAGNOSTIC DONE"

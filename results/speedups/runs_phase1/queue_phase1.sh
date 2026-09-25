#!/bin/bash
# Phase 1 (results/speedups/PROTOCOL_PHASE1.md): Llama end-to-end with items 1+3, then the Phi-4 / Qwen3.8-27B
# round-0 architecture checks at batch 16/8 (legacy vs new), then Qwen3.8-27B at 1/1 for documentation.
R=/home/dev/n16k64_campaign/speedups/phase1
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=/home/dev/n16k64_campaign/speedups/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p $R/logs $R/checks
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $R/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"; return $rc; }
NEW="--fused-act-quant --single-pass-epilogue"
LLAMA="--model llama8b --objective kl --data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation --memory-mode lean --dev-backend native --deterministic --skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --record-dev-values --dump-round0-scores"
for U in 256x64 8x64; do
  if [ $U = 256x64 ]; then REF=$REPO/results/native_decision/runs/det_native/report.json; MAP=/home/dev/n16k64_campaign/native_decision/det_native/map.pt; LEAN=/home/dev/n16k64_campaign/lean_memory/lean_det_native_256x64
  else REF=$REPO/results/native_decision_8x64/runs/det_native_8x64/report.json; MAP=/home/dev/n16k64_campaign/native_decision_8x64/det_native_8x64/map.pt; LEAN=/home/dev/n16k64_campaign/lean_memory/lean_det_native_8x64; fi
  run llama_det_native_$U $PY run_multiround.py $LLAMA --unit $U $NEW --out $R/llama_det_native_$U || { log "STOP run failed"; exit 1; }
  $PY results/lean_memory/compare_runs.py $REF $MAP $R/llama_det_native_$U $LEAN > $R/checks/llama_det_native_$U.json 2> $R/logs/check_llama_$U.err
  RC=$?; log "CHECK llama_det_native_$U rc=$RC"; [ $RC -eq 0 ] || { log "STOP mismatch llama_det_native_$U"; exit 1; }
done
CHECK="--objective kl --data-root /home/dev/n16k64_campaign/multimodel/data --memory-mode lean --unit 8x64 --dev-backend native --deterministic --skip-ce-backward --stop-after-scoring --dump-round0-scores --record-dev-values"
for M in phi4 qwen27b; do
  EXTRA=""; [ $M = qwen27b ] && EXTRA="--gpus 1"
  run ${M}_legacy_16x8 $PY run_multiround.py --model $M $CHECK $EXTRA --eval-batch 16 --score-batch 8 --out $R/${M}_legacy_16x8
  L=$?
  if [ $L -ne 0 ]; then
    grep -q -i "out of memory" $R/logs/${M}_legacy_16x8.log && log "SKIP $M check: out of memory at batch 16/8 in lean mode (see its log)" || { log "STOP $M legacy run failed"; exit 1; }
  else
    run ${M}_new_16x8 $PY run_multiround.py --model $M $CHECK $EXTRA --eval-batch 16 --score-batch 8 $NEW --out $R/${M}_new_16x8 || { log "STOP $M new run failed"; exit 1; }
    $PY results/multiround_models/compare_batching.py $R/${M}_legacy_16x8 $R/${M}_new_16x8 > $R/checks/${M}_legacy_vs_new.json 2> $R/logs/check_$M.err
    log "CHECK ${M}_legacy_vs_new identical=$($PY -c "import json; print(json.load(open('$R/checks/${M}_legacy_vs_new.json'))['identical'])")"
  fi
done
run qwen27b_new_1x1 $PY run_multiround.py --model qwen27b $CHECK --gpus 1 --eval-batch 1 --score-batch 1 $NEW --out $R/qwen27b_new_1x1
if [ -f $R/qwen27b_new_16x8/report.json ] && [ -f $R/qwen27b_new_1x1/report.json ]; then
  $PY results/multiround_models/compare_batching.py $R/qwen27b_new_16x8 $R/qwen27b_new_1x1 > $R/checks/qwen27b_batched_vs_single.json 2> $R/logs/check_qwen27b_batching.err
  log "DOC qwen27b batched_vs_single identical=$($PY -c "import json; print(json.load(open('$R/checks/qwen27b_batched_vs_single.json'))['identical'])")"
fi
log "PHASE1 QUEUE DONE"

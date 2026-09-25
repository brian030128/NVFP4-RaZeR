#!/bin/bash
# Phase 2 (results/speedups/PROTOCOL_PHASE2.md): B1 unit tests, the Phi-4 round-0 run with B1, Llama end-to-end
# with B1 (items 1 and 3 on), and the evaluation of the B1 maps against the committed maps.
R=/home/dev/n16k64_campaign/speedups/phase2
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=/home/dev/n16k64_campaign/speedups/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
mkdir -p $R/logs
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $R/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"; return $rc; }
run b1_unit_tests $PY repro_local/realquant/test_tile_score.py $REPO/results/speedups/phase2_unit_tests.json
ALL="--fused-act-quant --single-pass-epilogue --tile-score-kernel"
run phi4_b1_16x8 $PY run_multiround.py --model phi4 --objective kl --data-root /home/dev/n16k64_campaign/multimodel/data --memory-mode lean --unit 8x64 --dev-backend native --deterministic --skip-ce-backward --stop-after-scoring --dump-round0-scores --record-dev-values --eval-batch 16 --score-batch 8 $ALL --out $R/phi4_b1_16x8
LLAMA="--model llama8b --objective kl --data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation --memory-mode lean --dev-backend native --deterministic --skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --record-dev-values --dump-round0-scores"
for U in 256x64 8x64; do
  run llama_det_native_${U}_b1 $PY run_multiround.py $LLAMA --unit $U $ALL --out $R/llama_det_native_${U}_b1 || { log "STOP run failed"; exit 1; }
done
MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map COMMITTED-256x64=/home/dev/n16k64_campaign/native_decision/det_native/map.pt \
 --evaluate-map COMMITTED-8x64=/home/dev/n16k64_campaign/native_decision_8x64/det_native_8x64/map.pt \
 --evaluate-map B1-256x64=$R/llama_det_native_256x64_b1/map.pt --evaluate-map B1-8x64=$R/llama_det_native_8x64_b1/map.pt"
EVAL="--model llama8b --objective kl --data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation --memory-mode lean --unit 8x64 --fused-act-quant --single-pass-epilogue"
run eval_native $PY run_multiround.py $EVAL --eval-backend native $MAPS --out $R/eval_native
run eval_fake $PY run_multiround.py $EVAL --eval-backend fake $MAPS --out $R/eval_fake
log "PHASE2 QUEUE DONE"

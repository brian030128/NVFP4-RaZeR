#!/bin/bash
# Option A (results/speedups/PROTOCOL_CHUNKED.md): chunked loss + expandable segments for Qwen3.8-27B at batch 16/8.
R=/home/dev/n16k64_campaign/speedups/chunked
P1=/home/dev/n16k64_campaign/speedups/phase1
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=/home/dev/n16k64_campaign/speedups/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R/logs $R/checks
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $R/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"; return $rc; }
run chunked_unit_tests $PY repro_local/realquant/test_chunked_loss.py $REPO/results/speedups/chunked_unit_tests.json || { log "STOP chunked unit tests failed"; exit 1; }
ITEMS="--fused-act-quant --single-pass-epilogue"
LLAMA="--model llama8b --objective kl --data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation --memory-mode lean --dev-backend native --deterministic --skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --record-dev-values --dump-round0-scores"
run llama_det_native_256x64_chunked $PY run_multiround.py $LLAMA --unit 256x64 $ITEMS --chunked-loss --out $R/llama_det_native_256x64_chunked || { log "STOP run failed"; exit 1; }
$PY results/lean_memory/compare_runs.py $REPO/results/native_decision/runs/det_native/report.json /home/dev/n16k64_campaign/native_decision/det_native/map.pt \
    $R/llama_det_native_256x64_chunked /home/dev/n16k64_campaign/lean_memory/lean_det_native_256x64 > $R/checks/llama_det_native_256x64_chunked.json 2> $R/logs/check_llama.err
RC=$?; log "CHECK llama_det_native_256x64_chunked rc=$RC"; [ $RC -eq 0 ] || { log "STOP mismatch llama"; exit 1; }
CHECK="--objective kl --data-root /home/dev/n16k64_campaign/multimodel/data --memory-mode lean --unit 8x64 --dev-backend native --deterministic --skip-ce-backward --stop-after-scoring --dump-round0-scores --record-dev-values $ITEMS"
run phi4_chunked_16x8 $PY run_multiround.py --model phi4 $CHECK --eval-batch 16 --score-batch 8 --chunked-loss --out $R/phi4_chunked_16x8 || { log "STOP run failed"; exit 1; }
$PY results/multiround_models/compare_batching.py $P1/phi4_new_16x8 $R/phi4_chunked_16x8 > $R/checks/phi4_whole_vs_chunked.json 2> $R/logs/check_phi4.err
ID=$($PY -c "import json; print(json.load(open('$R/checks/phi4_whole_vs_chunked.json'))['identical'])"); log "CHECK phi4_whole_vs_chunked identical=$ID"; [ "$ID" = True ] || { log "STOP mismatch phi4"; exit 1; }
run qwen27b_whole_4x4 $PY run_multiround.py --model qwen27b $CHECK --gpus 1 --eval-batch 4 --score-batch 4 --out $R/qwen27b_whole_4x4 || { log "STOP run failed"; exit 1; }
run qwen27b_chunked_4x4 $PY run_multiround.py --model qwen27b $CHECK --gpus 1 --eval-batch 4 --score-batch 4 --chunked-loss --out $R/qwen27b_chunked_4x4 || { log "STOP run failed"; exit 1; }
$PY results/multiround_models/compare_batching.py $R/qwen27b_whole_4x4 $R/qwen27b_chunked_4x4 > $R/checks/qwen27b_whole_vs_chunked_4x4.json 2> $R/logs/check_qwen_4x4.err
ID=$($PY -c "import json; print(json.load(open('$R/checks/qwen27b_whole_vs_chunked_4x4.json'))['identical'])"); log "CHECK qwen27b_whole_vs_chunked_4x4 identical=$ID"; [ "$ID" = True ] || { log "STOP mismatch qwen27b 4x4"; exit 1; }
run qwen27b_chunked_16x8 $PY run_multiround.py --model qwen27b $CHECK --gpus 1 --eval-batch 16 --score-batch 8 --chunked-loss --out $R/qwen27b_chunked_16x8
if [ $? -ne 0 ]; then grep -q -i "out of memory" $R/logs/qwen27b_chunked_16x8.log && log "STOP qwen27b still out of memory at 16/8 (see its log)" || log "STOP qwen27b 16/8 run failed"; exit 1; fi
$PY results/multiround_models/compare_batching.py $R/qwen27b_chunked_16x8 $P1/qwen27b_new_1x1 > $R/checks/qwen27b_batched_16x8_vs_single.json 2> $R/logs/check_qwen_doc.err
log "DOC qwen27b batched_16x8_vs_single identical=$($PY -c "import json; print(json.load(open('$R/checks/qwen27b_batched_16x8_vs_single.json'))['identical'])")"
log "CHUNKED QUEUE DONE"

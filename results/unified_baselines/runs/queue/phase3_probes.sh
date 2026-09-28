#!/bin/bash
# Phase 3 (PROTOCOL.md): memory probes only, Phi-4 and Qwen3.8-27B. 3 optimizer steps of 8 sequences each
# (run_cost_distill.py --budget probe); nothing is deployed and no fallback is trained. An out-of-memory stop is a result
# (status 'oom', with the peaks at the failure); any other failure is logged and the queue continues.
source /home/dev/n16k64_campaign/unified_baselines/common.sh
P=$R/probes
mkdir -p $P
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
status () { $PY -c "import json, sys; print(json.load(open(sys.argv[1] + '/report.json')).get('status'))" $1 2>/dev/null || echo missing; }
PROBE="--budget probe --probe-steps 3 --act-rows --deterministic"
for M in phi4 qwen27b; do
  go () { local name=$1; shift; probe ${M}_$name env PYTHONPATH=$QPATH $PY run_cost_distill.py --model $M --data-root $(root $M) \
            $PROBE "$@" --out $P/${M}_$name; }
  # SCALE at the unified settings (micro-batch 8, no checkpointing, as arm D); if it does not fit, smaller micro-batches
  # with accumulation (the optimizer batch stays 8), then checkpointing
  for mb in 8 4 2 1; do
    go scale_mb$mb --arm scale --micro-batch $mb --lr 3e-4
    [ "$(status $P/${M}_scale_mb$mb)" = complete ] && break
  done
  [ "$(status $P/${M}_scale_mb1)" = oom ] && go scale_mb1_ckpt --arm scale --micro-batch 1 --checkpointing --lr 3e-4
  # QAT at the unified settings (FP32-state AdamW, micro-batch 8, checkpointing), then each fallback, measured only
  go qat_fp32_mb8 --arm qat --optimizer adamw_fp32 --micro-batch 8 --checkpointing --lr 1e-5
  [ "$(status $P/${M}_qat_fp32_mb8)" = complete ] || go qat_fp32_mb1 --arm qat --optimizer adamw_fp32 --micro-batch 1 --checkpointing --lr 1e-5
  for opt in adamw_8bit cpu_offload; do
    go qat_${opt}_mb8 --arm qat --optimizer $opt --micro-batch 8 --checkpointing --lr 1e-5
    [ "$(status $P/${M}_qat_${opt}_mb8)" = complete ] || go qat_${opt}_mb1 --arm qat --optimizer $opt --micro-batch 1 --checkpointing --lr 1e-5
  done
  go lora_mb8 --arm lora --micro-batch 8 --checkpointing --lr 1e-4
  [ "$(status $P/${M}_lora_mb8)" = complete ] || go lora_mb1 --arm lora --micro-batch 1 --checkpointing --lr 1e-4
done
log "PHASE 3 DONE"

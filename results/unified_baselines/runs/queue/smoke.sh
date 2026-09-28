#!/bin/bash
# Pre-registration smoke test of the new paths (not a result): QAT with per-token activations (1 epoch, with the dev set:
# fused-quantizer check, frozen check), the QAT export (--weights, ownership), the QAT fake (c) policy against native on
# 4 windows, --model mistral7b (SCALE, 1 epoch), and the Qwen loader (a 1-step SCALE probe).
source /home/dev/n16k64_campaign/unified_baselines/common.sh
S=$D/smoke
LOG=$S/smoke.log
mkdir -p $S
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
run smoke_llama_qat env PYTHONPATH=$QPATH $PY run_cost_distill.py --model llama8b --data-root $(root llama8b) --transformers-deviation \
    $QAT --budget c1 --epochs 1 --act-rows --deterministic --lr 1e-5 --save-state --out $S/llama_qat
run smoke_llama_art $PY export_map_artifact.py --model llama8b --data-root $(root llama8b) --kind four_over_six \
    --weights $S/llama_qat/state.pt --ownership --out $S/art_llama_qat
run smoke_llama_eval $PY run_ppl_deploy.py --model llama8b --data-root $(root llama8b) --transformers-deviation --limit-windows 4 \
    --evaluate qat-fake=fake:weights:$S/llama_qat/state.pt --evaluate qat=native:$S/art_llama_qat --out $S/eval_llama
run smoke_mistral_scale env PYTHONPATH=$QPATH $PY run_cost_distill.py --model mistral7b --data-root $(root mistral7b) \
    $SCALE --budget c1 --epochs 1 --act-rows --deterministic --lr 3e-4 --out $S/mistral_scale
probe smoke_qwen_probe env PYTHONPATH=$QPATH $PY run_cost_distill.py --model qwen27b --data-root $(root qwen27b) \
    --arm scale --micro-batch 1 --budget probe --probe-steps 1 --act-rows --deterministic --lr 3e-4 --out $S/qwen_probe
log "SMOKE DONE"

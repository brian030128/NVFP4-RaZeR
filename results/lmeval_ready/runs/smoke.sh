#!/bin/bash
# Task 3 smoke tests (results/lmeval_ready/READINESS.md): lm-eval 0.4.11 through HFLM, 4 models x {BF16, FourOverSix and
# TC 16x64 as fake (c) and NativeLinear (c)}, limit 20 per task (gsm8k 5), the TC native policy evaluated twice.
# Runs only after Task 2 has finished (no GPU job may disturb its timings). A failing model is logged and the queue
# continues (the failure is part of the readiness report).
D=/home/dev/n16k64_campaign/lmeval_ready
DE=/home/dev/n16k64_campaign/deploy_eval/artifacts
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=$D/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131
mkdir -p $D/runs $D/logs
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
try_run () { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $D/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
             [ $rc -eq 0 ] || log "FAIL (logged, continuing): $name"; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
root () { [ $1 = llama8b ] && echo /home/dev/n16k64_campaign/cost_comparison/data || echo /home/dev/n16k64_campaign/multimodel/data; }
deviation () { [ $1 = llama8b ] && echo --transformers-deviation; }
for M in llama8b mistral7b phi4 qwen27b; do
  BS=16; [ $M = qwen27b ] && BS=8
  try_run smoke_$M $PY run_lmeval_deploy.py --model $M --data-root $(root $M) $(deviation $M) --batch-size $BS \
      --evaluate BF16=bf16 --evaluate FourOverSix-fake=fake:four_over_six \
      --evaluate tc-16x64-fake=fake:map:$DE/${M}_tc_16x64.mixfp4map \
      --evaluate FourOverSix=native:$DE/${M}_fo6 --evaluate tc-16x64=native:$DE/${M}_tc_16x64 \
      --limit 20 --gsm8k-limit 5 --repeat tc-16x64 --out $D/runs/$M
done
# the version cross-check, Llama BF16 only: 0.4.11 without BOS, and 0.4.5 itself (expected equal to each other)
try_run xcheck_nobos $PY run_lmeval_deploy.py --model llama8b --data-root $(root llama8b) --transformers-deviation --batch-size 16 \
    --evaluate BF16=bf16 --no-bos --limit 20 --gsm8k-limit 5 --out $D/runs/xcheck_llama8b_0411_nobos
try_run xcheck_045 env PYTHONPATH=/home/dev/n16k64_campaign/lmeval045/pydeps:$REPO $PY run_lmeval_deploy.py --model llama8b \
    --data-root $(root llama8b) --transformers-deviation --batch-size 16 --evaluate BF16=bf16 --limit 20 --gsm8k-limit 5 \
    --out $D/runs/xcheck_llama8b_045
log "TASK 3 SMOKE DONE"

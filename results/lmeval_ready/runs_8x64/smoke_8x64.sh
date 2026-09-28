#!/bin/bash
# 8x64 smoke tests (results/lmeval_ready/READINESS.md, "8x64 and kernel widths"): lm-eval 0.4.11 through HFLM, 4 models x
# {TC 8x64 as fake (c) and NativeLinear (c) (n8k64_wB), FourOverSix and TC 16x64 as NativeLinear (c)}, limit 20 per
# task (gsm8k 5), the 8x64 native policy evaluated twice. The native policies record, per task, the GEMM calls by CTA
# tile width and the forwards by token count, so the gsm8k decode steps' kernels are measured rather than inferred.
# BF16 and the 16x64 fake policy were smoke-tested before (smoke.sh) and are not repeated.
D=/home/dev/n16k64_campaign/lmeval_ready
DE=/home/dev/n16k64_campaign/deploy_eval/artifacts
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=$D/commands_8x64.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131
mkdir -p $D/runs_8x64 $D/logs
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
try_run () { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $D/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
             [ $rc -eq 0 ] || log "FAIL (logged, continuing): $name"; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
root () { [ $1 = llama8b ] && echo /home/dev/n16k64_campaign/cost_comparison/data || echo /home/dev/n16k64_campaign/multimodel/data; }
deviation () { [ $1 = llama8b ] && echo --transformers-deviation; }
for M in llama8b mistral7b phi4 qwen27b; do
  BS=16; [ $M = qwen27b ] && BS=8
  try_run smoke8x64_$M $PY run_lmeval_deploy.py --model $M --data-root $(root $M) $(deviation $M) --batch-size $BS \
      --evaluate tc-8x64-fake=fake:map:$DE/${M}_tc_8x64.mixfp4map \
      --evaluate FourOverSix=native:$DE/${M}_fo6 --evaluate tc-16x64=native:$DE/${M}_tc_16x64 \
      --evaluate tc-8x64=native:$DE/${M}_tc_8x64 \
      --limit 20 --gsm8k-limit 5 --repeat tc-8x64 --out $D/runs_8x64/$M
done
log "8x64 SMOKE DONE"

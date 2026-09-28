#!/bin/bash
# Task 3: the Qwen smoke test again, after the coverage check was corrected (unscoped vision Linears stay BF16 and must
# never run; the first run failed only on that check, after FourOverSix native had completed every task).
D=/home/dev/n16k64_campaign/lmeval_ready
DE=/home/dev/n16k64_campaign/deploy_eval/artifacts
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=$D/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
M=qwen27b
log "START smoke_${M}_rerun"
$PY run_lmeval_deploy.py --model $M --data-root /home/dev/n16k64_campaign/multimodel/data --batch-size 8 \
    --evaluate BF16=bf16 --evaluate FourOverSix-fake=fake:four_over_six \
    --evaluate tc-16x64-fake=fake:map:$DE/${M}_tc_16x64.mixfp4map \
    --evaluate FourOverSix=native:$DE/${M}_fo6 --evaluate tc-16x64=native:$DE/${M}_tc_16x64 \
    --limit 20 --gsm8k-limit 5 --repeat tc-16x64 --out $D/runs/$M < /dev/null > $D/logs/smoke_$M.log 2>&1
log "END smoke_${M}_rerun rc=$?"

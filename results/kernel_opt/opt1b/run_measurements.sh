#!/bin/bash
# kernel-opt optimization 1b (PROTOCOL.md amendment 1), measurements M1 -> M2 -> M3; stops at the first failure
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
OUT=/home/dev/n16k64_campaign/kernel_opt/opt1b
AFTER=/home/dev/n16k64_campaign/kernel_opt/build
A=/home/dev/n16k64_campaign/paper/artifacts
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
log "BEGIN measurements (1b) at $(git rev-parse HEAD)"
for m in llama8b mistral7b phi4 qwen27b; do
  if [ -f $OUT/gemm/$m.json ] && grep -q '"status": "complete"' $OUT/gemm/$m.json; then log "SKIP M1 $m (complete)"; continue; fi
  log "START M1_gemm_$m"
  set +e
  $PY experiments/kernel_opt/bench_ab_isolated.py --model $m --after-root $AFTER --opt1-table results/kernel_opt/opt1/table_opt1.json \
      --artifact fo6=$A/${m}_fo6 --artifact tc_8x64=$A/${m}_tc_8x64 --out $OUT/gemm/$m.json > $OUT/logs/M1_gemm_$m.log 2>&1
  rc=$?
  set -e
  log "END M1_gemm_$m rc=$rc"
  [ $rc -eq 0 ] || { log "STOP: M1 $m failed"; exit 1; }
done
log "START M2_M3_e2e"
set +e
$PY experiments/kernel_opt/ab_e2e.py --what prefill,decode --out $OUT/e2e > $OUT/logs/M2_M3_e2e.log 2>&1
rc=$?
set -e
log "END M2_M3_e2e rc=$rc"
[ $rc -eq 0 ] || { log "STOP: M2/M3 failed"; exit 1; }
log "DONE measurements"

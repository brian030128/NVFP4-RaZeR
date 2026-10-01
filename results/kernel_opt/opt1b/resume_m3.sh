#!/bin/bash
# kernel-opt optimization 1b: M3 decode, resumed as registered (PROTOCOL.md, note 2 to amendment 1); stops at the first failure
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF KERNEL_OPT_BUILD KERNEL_OPT_OUT
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
OUT=/home/dev/n16k64_campaign/kernel_opt/opt1b
echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) RESUME M3 (user start relayed by nvfp4-razer-c9; note 2 to amendment 1) at $(git rev-parse HEAD)" >> $OUT/commands.log
echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) START M3_decode (ab_e2e.py --what decode)" >> $OUT/commands.log
set +e
/home/dev/.conda/envs/n16k64/bin/python experiments/kernel_opt/ab_e2e.py --what decode --out $OUT/e2e > $OUT/logs/M3_decode.log 2>&1
rc=$?
set -e
echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) END M3_decode rc=$rc" >> $OUT/commands.log
[ $rc -eq 0 ] || exit 1

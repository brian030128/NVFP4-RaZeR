#!/bin/bash
# kernel-opt 1b: pause by user request -- finish M2 prefill only (M3 decode deferred); stops at the first failure
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
OUT=/home/dev/n16k64_campaign/kernel_opt/opt1b
tail --pid=849676 -f /dev/null
echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) orphaned prefill_qwen27b_ours-8x64-opt_r4 (pid 849676) exited; record status: $(python3 -c "import json;print(json.load(open('$OUT/e2e/prefill/qwen27b/ours-8x64-opt/round4.json'))['status'])" 2>&1)" >> $OUT/commands.log
echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) START M2_prefill_finish (ab_e2e.py --what prefill)" >> $OUT/commands.log
set +e
/home/dev/.conda/envs/n16k64/bin/python experiments/kernel_opt/ab_e2e.py --what prefill --out $OUT/e2e > $OUT/logs/M2_prefill_finish.log 2>&1
rc=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) END M2_prefill_finish rc=$rc (M3 decode deferred by user request)" >> $OUT/commands.log

#!/bin/bash
# Post-adoption check of amendment 18: whole-model logits, routed as deployed ('auto' / 'auto_stock' / 'auto_stock_wB')
# from build_V on the tracked table, against each unit's previous deployment, bitwise.
set -u
cd /home/dev/NVFP4-RaZeR || exit 1
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1 \
       SM120_BUILD_DIR=/home/dev/n16k64_campaign/kernel_opt/build_V
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
O=results/kernel_opt/V/adoption
L=$KO/V/adoption_logs; mkdir -p $L
CK="$PY experiments/kernel_opt/check_model_logits.py"
echo "$(date -u +%FT%T+00:00) start at $(git rev-parse HEAD), table sha256 $(sha256sum sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.ko.json | cut -c1-16)" >> $L/commands.log
run() { local n=$1; shift; echo "$(date -u +%FT%T+00:00) START $n" >> $L/commands.log; "$@" > $L/$n.log 2>&1; echo "$(date -u +%FT%T+00:00) END $n rc=$?" >> $L/commands.log; }
run auto_256x64 $CK --unit 256x64 --before set:mixed256 --before-root $KO/build_A1 --after auto:auto --expect-family mixed256_ko --out $O/auto_256x64.json
run auto_16x64 $CK --unit 16x64 --before set:mixed_ko --before-root $KO/build_U --after auto:auto --expect-family mixed_ko --out $O/auto_16x64.json
run auto_8x64 $CK --unit 8x64 --before set:mixed_wB_ko --before-root $KO/build_U --after auto:auto --expect-family mixed_wB_ko --out $O/auto_8x64.json
run auto_stock $CK --unit fo6 --before set:stock_ko --before-root $KO/build_7 --after auto:auto_stock --expect-family stock_ko --out $O/auto_stock.json
run auto_stock_wB $CK --unit fo6 --before set:stock_wB_ko --before-root $KO/build_U --after auto:auto_stock_wB --expect-family stock_wB_ko --out $O/auto_stock_wB.json
echo "$(date -u +%FT%T+00:00) done" >> $L/commands.log

#!/bin/bash
# Diagnostics for the (i) cells that were significant (PROTOCOL.md: report with diagnostics, criterion unchanged):
# sm120/eval/layerwise.py on the window with the largest |ΔNLL| and on window 0, for the two map cells.
D=/home/dev/n16k64_campaign/deploy_eval
A=$D/artifacts
PY=/home/dev/.conda/envs/n16k64/bin/python
cd /home/dev/NVFP4-RaZeR
export HF_HUB_OFFLINE=0 PYTHONPATH=/home/dev/NVFP4-RaZeR HF_HOME=/home/dev/.cache/huggingface
for spec in "mistral7b tc_16x64 c4 124" "mistral7b tc_16x64 c4 0" "qwen27b tc_16x64 wiki 137" "qwen27b tc_16x64 wiki 0"; do
  set -- $spec
  echo "$(date -u +%FT%TZ) START layerwise $spec" >> $D/commands.log
  $PY sm120/eval/layerwise.py --model $1 --map $A/${1}_$2.mixfp4map --artifact $A/${1}_$2 --kernel n16k64_wA --domain $3 \
      --window $4 --out $D/diagnostics/layerwise_${1}_$2_$3_w$4.json > $D/diagnostics/layerwise_${1}_$2_$3_w$4.log 2>&1
  echo "$(date -u +%FT%TZ) END layerwise $spec rc=$?" >> $D/commands.log
done
echo "$(date -u +%FT%TZ) LAYERWISE DONE" >> $D/commands.log

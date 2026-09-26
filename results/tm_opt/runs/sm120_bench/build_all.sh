#!/bin/bash
# #3 build (results/tm_opt/PROTOCOL_ITEMS.md): the SM120 deployment kernel configurations, on the CPU, low priority.
cd /home/dev/n16k64_campaign/sm120_bench
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131
for C in n16k64_wA n8k64_wB stock_wA stock_wB; do
  echo "$(date -u +%FT%TZ) BUILD $C" >> build.log
  nice -n 10 /home/dev/.conda/envs/n16k64/bin/python sm120/build.py --config $C > build_$C.log 2>&1
  echo "$(date -u +%FT%TZ) END $C rc=$?" >> build.log
done

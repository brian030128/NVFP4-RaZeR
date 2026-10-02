#!/bin/bash
# kernel-opt amendment 11 (the 8x64 plan's P2): CPU-only builds from the clean main checkout at the P2 source commit.
# build_P2: every configuration of configs.py (default dispatch), 13 parallel `build.py --config` processes (round-robin
# split; `--all` is the same per-configuration build in a loop). build_P2freq: the five wB t0 builds with
# MIXFP4_DISPATCH_FREQ=1 (#2's dispatch). No GPU is visible to the builds.
cd /home/dev/NVFP4-RaZeR
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
L=$KO/P2/build_logs
mkdir -p $L
echo "$(date -u +%FT%T+00:00) builds start at $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)" >> $KO/P2/build.log
NAMES=$($PY -c "from mixfp4_sm120 import configs as C; print(' '.join(C.CONFIGS))")
NP=13
i=0; declare -a PARTS
for n in $NAMES; do PARTS[$((i % NP))]+=" --config $n"; i=$((i+1)); done
pids=()
for p in $(seq 0 $((NP-1))); do
  SM120_BUILD_DIR=$KO/build_P2 nice -n 10 $PY sm120/build.py ${PARTS[$p]} > $L/build_P2_part$p.log 2>&1 &
  pids+=($!)
done
for n in n8k64_wB_t0 n8k64_wB_m64_t0 n8k64_wB_m32_t0 n8k64_wB_m16_t0 n8k64_wB_n64_t0; do
  SM120_BUILD_DIR=$KO/build_P2freq nice -n 10 $PY sm120/build.py --define MIXFP4_DISPATCH_FREQ=1 --config $n > $L/build_P2freq_$n.log 2>&1 &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait $pid || rc=1; done
echo "$(date -u +%FT%T+00:00) builds end rc=$rc" >> $KO/P2/build.log
exit $rc

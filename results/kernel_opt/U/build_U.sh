#!/bin/bash
# kernel-opt amendment 17 (U): CPU-only builds from the clean main checkout at the U source commit.
# build_Uall: every configuration of configs.py with no extra define (G1), 13 parallel `build.py --config` processes
# (round-robin split). build_U: amendment 17's deployment directory -- part A's builds with their registered defines:
#   16x64 n16k64_wA_{n16,n32}_t0             MIXFP4_DISPATCH_FREQ=1 (today's builds, identical SASS)
#   16x64 n16k64_wA_{n64,e64}_t0             MIXFP4_DISPATCH_FREQ=1 MIXFP4_UNIFORM_DISPATCH=1
#   8x64  n8k64_wB_{m16,m32}_t0              MIXFP4_DISPATCH_FREQ=1 (today's builds, identical SASS)
#   8x64  n8k64_wB_{m64,n64}_t0, n8k64_wB_t0 MIXFP4_DISPATCH_FREQ=1 MIXFP4_PIPE_FLAGS=1
#   stock_wB_e64                             no define ('auto_stock_wB' routes to it in this directory)
# No GPU is visible to the builds.
cd /home/dev/NVFP4-RaZeR
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
L=$KO/U/build_logs
mkdir -p $L
echo "$(date -u +%FT%T+00:00) builds start at $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)" >> $KO/U/build.log
NAMES=$($PY -c "from mixfp4_sm120 import configs as C; print(' '.join(C.CONFIGS))")
NP=13
i=0; declare -a PARTS
for n in $NAMES; do PARTS[$((i % NP))]+=" --config $n"; i=$((i+1)); done
pids=()
for p in $(seq 0 $((NP-1))); do
  SM120_BUILD_DIR=$KO/build_Uall nice -n 10 $PY sm120/build.py ${PARTS[$p]} > $L/build_Uall_part$p.log 2>&1 &
  pids+=($!)
done
F="--define MIXFP4_DISPATCH_FREQ=1"
build_u() { local c=$1; shift; SM120_BUILD_DIR=$KO/build_U nice -n 10 $PY sm120/build.py "$@" --config $c > $L/build_U_$c.log 2>&1; }
for c in n16k64_wA_n16_t0 n16k64_wA_n32_t0 n8k64_wB_m16_t0 n8k64_wB_m32_t0; do build_u $c $F & pids+=($!); done
for c in n16k64_wA_n64_t0 n16k64_wA_e64_t0; do build_u $c $F --define MIXFP4_UNIFORM_DISPATCH=1 & pids+=($!); done
for c in n8k64_wB_m64_t0 n8k64_wB_n64_t0 n8k64_wB_t0; do build_u $c $F --define MIXFP4_PIPE_FLAGS=1 & pids+=($!); done
build_u stock_wB_e64 & pids+=($!)
rc=0
for pid in "${pids[@]}"; do wait $pid || rc=1; done
echo "$(date -u +%FT%T+00:00) builds end rc=$rc" >> $KO/U/build.log
exit $rc

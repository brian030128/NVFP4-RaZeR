#!/bin/bash
# kernel-opt amendment 18 (V): CPU-only builds from the clean main checkout at the V source commit.
# build_Vall: every configuration of configs.py with no extra define (G1/G2), 13 parallel `build.py --config` processes
# (round-robin split). build_V: amendment 18's deployment directory, every adopted set with its registered defines:
#   16x64 'mixed_ko'     n16k64_wA_{n16,n32}_t0 FREQ; n16k64_wA_{n64,e64}_t0 FREQ + UNIFORM   (amendment 17)
#   8x64 'mixed_wB_ko'   n8k64_wB_{m16,m32}_t0 FREQ; n8k64_wB_{m64,n64}_t0, n8k64_wB_t0 FREQ + PIPE   (amendment 17)
#   256x64 'mixed256_ko' n16k64_wA_g32_{n16,n32,n64,e64}_t0 UNIFORM   (new)
#   256x64 'mixed256'    n16k64_wA_g32{,_n64,_n32,_n16}, no define (A', for 'paper_256')
#   'stock_ko'           stock_wA_{n16,n32,n64,e64}, no define;  'stock_wB_ko' stock_wB_e64, no define
# No GPU is visible to the builds.
cd /home/dev/NVFP4-RaZeR
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
L=$KO/V/build_logs
mkdir -p $L
echo "$(date -u +%FT%T+00:00) builds start at $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)" >> $KO/V/build.log
NAMES=$($PY -c "from mixfp4_sm120 import configs as C; print(' '.join(C.CONFIGS))")
NP=13
i=0; declare -a PARTS
for n in $NAMES; do PARTS[$((i % NP))]+=" --config $n"; i=$((i+1)); done
pids=()
for p in $(seq 0 $((NP-1))); do
  SM120_BUILD_DIR=$KO/build_Vall nice -n 10 $PY sm120/build.py ${PARTS[$p]} > $L/build_Vall_part$p.log 2>&1 &
  pids+=($!)
done
F="--define MIXFP4_DISPATCH_FREQ=1"; U="--define MIXFP4_UNIFORM_DISPATCH=1"; P="--define MIXFP4_PIPE_FLAGS=1"
build_v() { local c=$1; shift; SM120_BUILD_DIR=$KO/build_V nice -n 10 $PY sm120/build.py "$@" --config $c > $L/build_V_$c.log 2>&1; }
for c in n16k64_wA_n16_t0 n16k64_wA_n32_t0 n8k64_wB_m16_t0 n8k64_wB_m32_t0; do build_v $c $F & pids+=($!); done
for c in n16k64_wA_n64_t0 n16k64_wA_e64_t0; do build_v $c $F $U & pids+=($!); done
for c in n8k64_wB_m64_t0 n8k64_wB_n64_t0 n8k64_wB_t0; do build_v $c $F $P & pids+=($!); done
for c in n16k64_wA_g32_n16_t0 n16k64_wA_g32_n32_t0 n16k64_wA_g32_n64_t0 n16k64_wA_g32_e64_t0; do build_v $c $U & pids+=($!); done
for c in n16k64_wA_g32 n16k64_wA_g32_n64 n16k64_wA_g32_n32 n16k64_wA_g32_n16 stock_wA_n16 stock_wA_n32 stock_wA_n64 stock_wA_e64 stock_wB_e64; do
  build_v $c & pids+=($!); done
rc=0
for pid in "${pids[@]}"; do wait $pid || rc=1; done
echo "$(date -u +%FT%T+00:00) builds end rc=$rc" >> $KO/V/build.log
exit $rc

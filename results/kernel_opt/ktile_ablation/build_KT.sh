#!/bin/bash
# kernel-opt K-tile ablation: CPU-only builds from the kernel-opt checkout (no GPU visible).
# build_KT:      the per-MMA-branch variants, with their deployed counterparts' --define sets (inert under the per-MMA
#                branch; the _nodef builds check that) -- n16k64_wA_e64_t0_permma FREQ + UNIFORM; n8k64_wB_t0_permma FREQ + PIPE
# build_KT_nodef: the same two without any --define (their SASS must equal build_KT's)
# build_KT_ref:  the deployed counterparts rebuilt from the modified sources with build_V's defines (their SASS must equal
#                build_V's: the new hooks are inactive unless asked for)
cd /home/dev/NVFP4-RaZeR
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
L=$KO/KT/build_logs
F="--define MIXFP4_DISPATCH_FREQ=1"; U="--define MIXFP4_UNIFORM_DISPATCH=1"; P="--define MIXFP4_PIPE_FLAGS=1"
echo "$(date -u +%FT%T+00:00) builds start at $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)" >> $KO/KT/build.log
b() { local dir=$1 c=$2; shift 2; SM120_BUILD_DIR=$KO/$dir nice -n 10 $PY sm120/build.py "$@" --config $c > $L/${dir}_$c.log 2>&1; }
pids=()
b build_KT n16k64_wA_e64_t0_permma $F $U & pids+=($!)
b build_KT n8k64_wB_t0_permma $F $P & pids+=($!)
b build_KT_nodef n16k64_wA_e64_t0_permma & pids+=($!)
b build_KT_nodef n8k64_wB_t0_permma & pids+=($!)
b build_KT_ref n16k64_wA_e64_t0 $F $U & pids+=($!)
b build_KT_ref n8k64_wB_t0 $F $P & pids+=($!)
b build_KT_ref stock_wA_e64 & pids+=($!)
rc=0
for pid in "${pids[@]}"; do wait $pid || rc=1; done
echo "$(date -u +%FT%T+00:00) builds end rc=$rc" >> $KO/KT/build.log
exit $rc

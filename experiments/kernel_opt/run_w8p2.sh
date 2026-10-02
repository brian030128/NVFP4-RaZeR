#!/bin/bash
# kernel-opt amendment 11 (the 8x64 plan's P2: t0 for the weights-on-B family, with P1's dispatch choice). Gates: G0
# (provenance), G3 (self-tests, in separate directories, tied to the registered device code), the patcher check, G1/G2
# (SASS), G3 (pytest), G4 (bitwise) and G5 (model logits). Then M1 (GEMM, 4 models), C2w''' (4096^3) and the report
# (results/kernel_opt/PROTOCOL.md). Stops at the first failure. build_P2 holds every configuration at the P2 sources and
# build_P2freq the five wB t0 builds with MIXFP4_DISPATCH_FREQ=1. Both were built CPU-only before registration
# (P2/build_P2.sh).
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
BP=$KO/build_P2
BPF=$KO/build_P2freq
ST=$KO/build_P2st
STF=$KO/build_P2stfreq
KOPT=$KO/build
FREQ=$KO/build_freq
B7=$KO/build_7
W=$KO/build_W
OUT=$KO/w8p2
R=results/kernel_opt/w8/p2
A=/home/dev/n16k64_campaign/paper/artifacts
WBT0=n8k64_wB_t0,n8k64_wB_m64_t0,n8k64_wB_m32_t0,n8k64_wB_m16_t0,n8k64_wB_n64_t0
mkdir -p $R $OUT/logs $OUT/gemm
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
# One `build.py --selftest` process per wB t0 configuration, into build directory $1 ($2: extra build.py arguments).
selftests() {
  local root=$1 extra=$2 pids=() rc=0 c
  for c in ${WBT0//,/ }; do
    SM120_BUILD_DIR=$root $PY sm120/build.py --selftest $extra --config $c > $OUT/logs/selftest_$(basename $root)_$c.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || rc=1; done
  cat $OUT/logs/selftest_$(basename $root)_*.log
  return $rc
}
log "BEGIN P2 (amendment 11) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_11.json \
     --out $R/g0_provenance.json
step G3_selftest selftests $ST ""
step G3_selftest_freq selftests $STF "--define MIXFP4_DISPATCH_FREQ=1"
step G3_selftest_sass $PY experiments/kernel_opt/check_same_sass.py --root $ST --like $BP --configs $WBT0 \
     --out $R/g3_selftest_sass.json
step G3_selftest_sass_freq $PY experiments/kernel_opt/check_same_sass.py --root $STF --like $BPF --configs $WBT0 \
     --out $R/g3_selftest_sass_freq.json
step PATCHER_sites $PY experiments/kernel_opt/check_patcher_sites.py \
     --tagged-roots /home/dev/NVFP4-RaZeR/sm120/build,$KOPT,$FREQ,$KO/build_4,$KO/build_A1,$KO/build_e0m3,$KO/build_T,$B7,$BP \
     --t0-roots $BP,$BPF,$ST,$STF,$W --out $R/patcher_sites.json
step G1_G2_sass $PY experiments/kernel_opt/check_sass.py --after-root $BP --tmopt-root $KO/build_tmopt --new $WBT0 \
     --before-roots $KOPT,$KO/build_e0m3,$KO/build_A1,$KO/build_4,$KO/build_T,$B7,$W,$KO/build_C3k --out $R/g1_g2_sass.json
step G2_sass_freq $PY experiments/kernel_opt/check_sass.py --after-root $BPF --tmopt-root $KO/build_tmopt --new $WBT0 \
     --only-new --out $R/g2_sass_freq.json
step G3_pytest env SM120_BUILD_DIR=$BP SM120_REF_BUILD_DIR=/home/dev/NVFP4-RaZeR/sm120/build $PY -m pytest \
     sm120/tests/test_gemm.py sm120/tests/test_select.py -rs -p no:cacheprovider -v
step G3_pytest_freq env SM120_BUILD_DIR=$BPF $PY -m pytest sm120/tests/test_gemm.py sm120/tests/test_select.py -k "wB" -rs \
     -p no:cacheprovider -v
step G4_bitwise $PY experiments/kernel_opt/check_bitwise.py --family wB --build-root $BP --candidates $WBT0 \
     --set mixed_wB_t0 --out $R/g4_bitwise.json
step G4_bitwise_freq $PY experiments/kernel_opt/check_bitwise.py --family wB --build-root $BPF --candidates $WBT0 \
     --set mixed_wB_t0 --out $R/g4_bitwise_freq.json
step G5_logits $PY experiments/kernel_opt/check_model_logits.py --unit 8x64 --before set:mixed_wB --before-root $KOPT \
     --after set:mixed_wB_t0 --after-root $BP --out $R/g5_logits.json
step G5_logits_freq $PY experiments/kernel_opt/check_model_logits.py --unit 8x64 --before set:mixed_wB --before-root $FREQ \
     --after set:mixed_wB_t0 --after-root $BPF --out $R/g5_logits_freq.json
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_w8p2_isolated.py --model $m --kopt-root $KOPT --freq-root $FREQ \
       --t0-root $BP --t0freq-root $BPF --b7 $B7 --artifact fo6=$A/${m}_fo6 --artifact tc_8x64=$A/${m}_tc_8x64 \
       --out $OUT/gemm/$m.json
done
step C2w3 $PY experiments/kernel_opt/c2_w8p2.py --kopt-root $KOPT --freq-root $FREQ --t0-root $BP --t0freq-root $BPF \
     --w-root $W --b7 $B7 --out $OUT/c2_w8p2.json
log "MEASUREMENTS DONE"
step REPORT $PY experiments/kernel_opt/w8p2_report.py --src $OUT --out-dir $R
log "DONE P2"

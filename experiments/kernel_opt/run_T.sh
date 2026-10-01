#!/bin/bash
# kernel-opt amendment 6 (t0: the blob's site-0 prmt tags dropped): gates G3/G1/G2/pytest/G4/G5, then M1 and C2‴
# (results/kernel_opt/PROTOCOL.md); stops at the first failure. build_T holds `build.py --all` and build_Tfreq the four
# 16x64 t0 builds with MIXFP4_DISPATCH_FREQ=1, both from the t0 sources (CPU, before registration).
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
BT=$KO/build_T
BTF=$KO/build_Tfreq
FREQ=$KO/build_freq
A1=$KO/build_A1
OUT=$KO/T
R=results/kernel_opt/t0
A=/home/dev/n16k64_campaign/paper/artifacts
T16=n16k64_wA_t0,n16k64_wA_n64_t0,n16k64_wA_n32_t0,n16k64_wA_n16_t0
T256=n16k64_wA_g32_t0,n16k64_wA_g32_n64_t0,n16k64_wA_g32_n32_t0,n16k64_wA_g32_n16_t0
mkdir -p $R $OUT/logs $OUT/gemm
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN t0 at $(git rev-parse HEAD)"
step G3_selftest env SM120_BUILD_DIR=$BT $PY sm120/build.py --selftest \
     --config n16k64_wA_t0 --config n16k64_wA_n64_t0 --config n16k64_wA_n32_t0 --config n16k64_wA_n16_t0 \
     --config n16k64_wA_g32_t0 --config n16k64_wA_g32_n64_t0 --config n16k64_wA_g32_n32_t0 --config n16k64_wA_g32_n16_t0
step G1_G2_sass $PY experiments/kernel_opt/check_sass.py --after-root $BT --tmopt-root $KO/build_tmopt \
     --new $T16,$T256,n16k64_wA_nodisp_t0 --before-roots $KO/build,$KO/build_e0m3,$KO/build_A1,$KO/build_4 \
     --out $R/g1_g2_sass.json
step G2_sass_freq $PY experiments/kernel_opt/check_sass.py --after-root $BTF --tmopt-root $KO/build_tmopt --new $T16 \
     --only-new --out $R/g2_sass_freq.json
step G3_pytest env SM120_BUILD_DIR=$BT SM120_REF_BUILD_DIR=/home/dev/NVFP4-RaZeR/sm120/build $PY -m pytest \
     sm120/tests/test_gemm.py sm120/tests/test_select.py sm120/tests/test_g32.py -rs -p no:cacheprovider -v
step G3_pytest_freq env SM120_BUILD_DIR=$BTF $PY -m pytest sm120/tests/test_select.py -k mixed_t0 -rs -p no:cacheprovider -v
step G4_bitwise_16x64 $PY experiments/kernel_opt/check_bitwise.py --family wA --build-root $BT --candidates $T16 \
     --set mixed_t0 --out $R/g4_bitwise_16x64.json
step G4_bitwise_256x64 $PY experiments/kernel_opt/check_bitwise.py --family g32 --build-root $BT --candidates $T256 \
     --set mixed256_t0 --out $R/g4_bitwise_256x64.json
step G4_bitwise_freq $PY experiments/kernel_opt/check_bitwise.py --family wA --build-root $BTF --candidates $T16 \
     --set mixed_t0 --out $R/g4_bitwise_freq.json
step G5_logits_16x64 $PY experiments/kernel_opt/check_model_logits.py --unit 16x64 --before set:mixed --after set:mixed_t0 \
     --after-root $BT --out $R/g5_logits_16x64.json
step G5_logits_16x64_freq $PY experiments/kernel_opt/check_model_logits.py --unit 16x64 --before set:mixed --before-root $FREQ \
     --after set:mixed_t0 --after-root $BTF --out $R/g5_logits_16x64_freq.json
step G5_logits_256x64 $PY experiments/kernel_opt/check_model_logits.py --unit 256x64 --before set:mixed256 --before-root $A1 \
     --after set:mixed256_t0 --after-root $BT --out $R/g5_logits_256x64.json
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_abT_isolated.py --model $m --t0-root $BT --t0freq-root $BTF --freq-root $FREQ \
       --a1-root $A1 --artifact fo6=$A/${m}_fo6 --artifact tc_16x64=$A/${m}_tc_16x64 --artifact tc_256x64=$A/${m}_tc_256x64 \
       --out $OUT/gemm/$m.json
done
step C2_t0 $PY experiments/kernel_opt/c2_t0.py --t0-root $BT --freq-root $FREQ --t0freq-root $BTF --out $OUT/c2_t0.json
log "DONE t0"

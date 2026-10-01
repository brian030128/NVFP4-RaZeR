#!/bin/bash
# kernel-opt amendment 7 (the user's adoption of t0 for 16x64, #4 for both families and the 4b widths for both): builds
# n16k64_wA_e64_t0 (+ #2's dispatch) and n16k64_wA_g32_e64, tunes the adopted sets' scheduler rows at the 4b widths,
# then gates G3/G1/G2/pytest/G4/G5 (results/kernel_opt/PROTOCOL.md); stops at the first failure. No M1: the combined
# path is measured in the cumulative registered run. build_7 holds all 39 configurations and build_7freq the four
# mixed_ko builds with MIXFP4_DISPATCH_FREQ=1, both from the amendment 7 sources (CPU, before registration).
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
B7=$KO/build_7
B7F=$KO/build_7freq
FREQ=$KO/build_freq
OUT=$KO/7
R=results/kernel_opt/7
A=/home/dev/n16k64_campaign/paper/artifacts
SLUG=nvidia_rtx_pro_6000_blackwell_workstation_edition
TB=results/kernel_opt/retune/b/tables/$SLUG.json
NEW=$R/table/$SLUG.json
KO16=n16k64_wA_n16_t0,n16k64_wA_n32_t0,n16k64_wA_n64_t0,n16k64_wA_e64_t0
mkdir -p $R/table $OUT/logs
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN adoption (amendment 7) at $(git rev-parse HEAD)"
step G3_selftest env SM120_BUILD_DIR=$B7 $PY sm120/build.py --selftest --config n16k64_wA_e64_t0 --config n16k64_wA_g32_e64
step G1_G2_sass $PY experiments/kernel_opt/check_sass.py --after-root $B7 --tmopt-root $KO/build_tmopt \
     --new n16k64_wA_e64_t0,n16k64_wA_g32_e64 --before-roots $KO/build,$KO/build_e0m3,$KO/build_A1,$KO/build_4,$KO/build_T \
     --out $R/g1_g2_sass.json
step G2_sass_freq $PY experiments/kernel_opt/check_sass.py --after-root $B7F --tmopt-root $KO/build_tmopt --new $KO16 \
     --only-new --out $R/g2_sass_freq.json
step SCHED_TUNE env SM120_BUILD_DIR=$B7 $PY sm120/bench/tune_tiles.py --schedule --families mixed_ko,stock_ko --cold --act-warm \
     --rounds 3 --iters 30 --width-table $TB --maps llama8b=$A/llama8b_tc_16x64.mixfp4map \
     --maps mistral7b=$A/mistral7b_tc_16x64.mixfp4map --maps phi4=$A/phi4_tc_16x64.mixfp4map \
     --maps qwen27b=$A/qwen27b_tc_16x64.mixfp4map --out-dir $R/table
step G3_pytest env SM120_BUILD_DIR=$B7 SM120_REF_BUILD_DIR=/home/dev/NVFP4-RaZeR/sm120/build $PY -m pytest \
     sm120/tests/test_gemm.py sm120/tests/test_select.py sm120/tests/test_g32.py -rs -p no:cacheprovider -v
step G3_pytest_freq env SM120_BUILD_DIR=$B7F $PY -m pytest sm120/tests/test_select.py -k "mixed_ko" -rs -p no:cacheprovider -v
step G4_bitwise_16x64 $PY experiments/kernel_opt/check_bitwise.py --family wA --build-root $B7 --candidates n16k64_wA_e64_t0 \
     --set mixed_ko --table $NEW --out $R/g4_bitwise_16x64.json
step G4_bitwise_freq $PY experiments/kernel_opt/check_bitwise.py --family wA --build-root $B7F --candidates n16k64_wA_e64_t0 \
     --set mixed_ko --table $NEW --out $R/g4_bitwise_freq.json
step G4_bitwise_g32 $PY experiments/kernel_opt/check_bitwise.py --family g32 --build-root $B7 --candidates n16k64_wA_g32_e64 \
     --out $R/g4_bitwise_g32.json
step G5_logits_16x64 $PY experiments/kernel_opt/check_model_logits.py --unit 16x64 --before set:mixed --after set:mixed_ko \
     --after-root $B7 --after-table $NEW --out $R/g5_logits_16x64.json
step G5_logits_16x64_freq $PY experiments/kernel_opt/check_model_logits.py --unit 16x64 --before set:mixed --before-root $FREQ \
     --after set:mixed_ko --after-root $B7F --after-table $NEW --out $R/g5_logits_16x64_freq.json
step G5_logits_fo6 $PY experiments/kernel_opt/check_model_logits.py --unit fo6 --before set:stock --after set:stock_ko \
     --after-root $B7 --after-table $NEW --out $R/g5_logits_fo6.json
log "GATES PASSED"
log "DONE adoption gates"

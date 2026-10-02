#!/bin/bash
# kernel-opt amendment 12 (the 8x64 plan's P3, with P4). Gates G0 (provenance) and G1/G2 (SASS), and the same-SASS checks:
# build_P3freq's 8x64 builds carry build_P2freq's device code, and its stock_wB_e64 build_P3's. Then the act-warm
# tuning: the widths of the adopted 8x64 set (mixed_wB_ko, the fastest median as 4b), the adopted table composed from
# them, and the scheduler rows of mixed_wB_ko and stock_wB_ko (amendment 7's rule). Then the decisive-margin sensitivity,
# the gates G3 (pytest), G4 (bitwise) and G5 (model logits), M1 (GEMM, 4 models) and the report
# (results/kernel_opt/PROTOCOL.md). Stops at the first failure. build_P3 holds every configuration at the P3 sources.
# build_P3freq holds the five 8x64 t0 builds with MIXFP4_DISPATCH_FREQ=1, and stock_wB_e64 without it (no dispatch).
# Both were built CPU-only before registration (P3/build_P3.sh).
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
BP3=$KO/build_P3
BP3F=$KO/build_P3freq
BP2=$KO/build_P2
BP2F=$KO/build_P2freq
B7=$KO/build_7
OUT=$KO/w8p3
R=results/kernel_opt/w8/p3
A=/home/dev/n16k64_campaign/paper/artifacts
SLUG=nvidia_rtx_pro_6000_blackwell_workstation_edition
KOT=sm120/configs/$SLUG.ko.json
TABLE=$R/table/$SLUG.json
WBT0=n8k64_wB_t0,n8k64_wB_m64_t0,n8k64_wB_m32_t0,n8k64_wB_m16_t0,n8k64_wB_n64_t0
MAPS8="--maps8 llama8b=$A/llama8b_tc_8x64.mixfp4map --maps8 mistral7b=$A/mistral7b_tc_8x64.mixfp4map \
       --maps8 phi4=$A/phi4_tc_8x64.mixfp4map --maps8 qwen27b=$A/qwen27b_tc_8x64.mixfp4map"
mkdir -p $R $OUT/logs $OUT/gemm
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN P3 (amendment 12) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_12.json \
     --out $R/g0_provenance.json
step G1_G2_sass $PY experiments/kernel_opt/check_sass.py --after-root $BP3 --tmopt-root $KO/build_tmopt --new stock_wB_e64 \
     --before-roots $KO/build,$KO/build_e0m3,$KO/build_A1,$KO/build_4,$KO/build_T,$B7,$KO/build_W,$KO/build_C3k,$BP2 \
     --out $R/g1_g2_sass.json
step G2_sass_freq $PY experiments/kernel_opt/check_sass.py --after-root $BP3F --tmopt-root $KO/build_tmopt \
     --new $WBT0,stock_wB_e64 --only-new --out $R/g2_sass_freq.json
step SAME_SASS_8x64 $PY experiments/kernel_opt/check_same_sass.py --root $BP3F --like $BP2F --configs $WBT0 --no-selftest \
     --out $R/same_sass_8x64.json
step SAME_SASS_stock $PY experiments/kernel_opt/check_same_sass.py --root $BP3F --like $BP3 --configs stock_wB_e64 \
     --no-selftest --out $R/same_sass_stock.json
step TUNE_widths env SM120_BUILD_DIR=$BP3F $PY sm120/bench/tune_tiles.py --families mixed_wB_ko $MAPS8 --cold --act-warm \
     --rounds 3 --iters 30 --out-dir $R/widths
step COMPOSE $PY experiments/kernel_opt/p3_tables.py compose --ko $KOT --widths $R/widths/$SLUG.json \
     --out $R/table_widths/$SLUG.json
step TUNE_schedule env SM120_BUILD_DIR=$BP3F $PY sm120/bench/tune_tiles.py --schedule --families mixed_wB_ko,stock_wB_ko \
     --width-table $R/table_widths/$SLUG.json $MAPS8 --cold --act-warm --rounds 3 --iters 30 --out-dir $R/table
step SENSITIVITY $PY experiments/kernel_opt/p3_tables.py sensitivity --raw $R/widths/$SLUG.raw.json \
     --widths $R/widths/$SLUG.json --out $R/sensitivity.json
step G3_pytest env SM120_BUILD_DIR=$BP3 SM120_REF_BUILD_DIR=/home/dev/NVFP4-RaZeR/sm120/build $PY -m pytest \
     sm120/tests/test_gemm.py sm120/tests/test_select.py -rs -p no:cacheprovider -v
step G3_pytest_freq env SM120_BUILD_DIR=$BP3F $PY -m pytest sm120/tests/test_gemm.py sm120/tests/test_select.py -k "wB" -rs \
     -p no:cacheprovider -v
step G4_bitwise $PY experiments/kernel_opt/check_bitwise.py --family wB --build-root $BP3F --candidates $WBT0 \
     --set mixed_wB_ko --table $TABLE --out $R/g4_bitwise.json
step G5_logits $PY experiments/kernel_opt/check_model_logits.py --unit 8x64 --before set:mixed_wB_t0 --before-root $BP2F \
     --after set:mixed_wB_ko --after-root $BP3F --after-table $TABLE --out $R/g5_logits.json
step G5_logits_fo6 $PY experiments/kernel_opt/check_model_logits.py --unit fo6 --before stock_wB --after set:stock_wB_ko \
     --after-root $BP3F --after-table $TABLE --out $R/g5_logits_fo6.json
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_w8p3_isolated.py --model $m --p3freq-root $BP3F --b7 $B7 --table $TABLE \
       --sensitivity $R/sensitivity.json --artifact fo6=$A/${m}_fo6 --artifact tc_8x64=$A/${m}_tc_8x64 \
       --out $OUT/gemm/$m.json
done
log "MEASUREMENTS DONE"
step REPORT $PY experiments/kernel_opt/w8p3_report.py --src $OUT --out-dir $R --sensitivity $R/sensitivity.json
log "DONE P3"

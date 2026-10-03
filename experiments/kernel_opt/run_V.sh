#!/bin/bash
# kernel-opt amendment 18 (V): the 256x64 path brought to the 16x64 state -- the set 'mixed256_ko' (A''s g32 builds with
# t0, #4's 64 x 64 epilogue tile at width 128 and the uniform-branch dispatch) and its act-warm re-tuned widths and
# scheduler rows. Gates: G0 (provenance), G3 (self-tests, in a separate directory, tied to the registered device code),
# the patcher check, G1/G2/G2u (SASS), G3 (pytest), G4 (bitwise) and G5 (model logits, set and routed). Then the table
# (today's widths, the width and scheduler tuning, the decisive-margin sensitivity), G5b (logits on the candidate table),
# M1 (GEMM, 4 models), C2V (4096^3) and the report (results/kernel_opt/PROTOCOL.md). Stops at the first failure. build_V
# and build_Vall were built CPU-only before registration (results/kernel_opt/V/build_V.sh). Run with `bash`.
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
BV=$KO/build_V
BVA=$KO/build_Vall
ST=$KO/build_Vst
BU=$KO/build_U
A1=$KO/build_A1
B7=$KO/build_7
C3K=$KO/build_C3k
OUT=$KO/V/run
R=results/kernel_opt/V
TB=$R/table_v
A=/home/dev/n16k64_campaign/paper/artifacts
NEW="n16k64_wA_g32_n16_t0 n16k64_wA_g32_n32_t0 n16k64_wA_g32_n64_t0 n16k64_wA_g32_e64_t0"
ALLV=n16k64_wA_n16_t0,n16k64_wA_n32_t0,n16k64_wA_n64_t0,n16k64_wA_e64_t0,n8k64_wB_m16_t0,n8k64_wB_m32_t0,n8k64_wB_m64_t0
ALLV=$ALLV,n8k64_wB_n64_t0,n8k64_wB_t0,stock_wB_e64,${NEW// /,},n16k64_wA_g32,n16k64_wA_g32_n64,n16k64_wA_g32_n32
ALLV=$ALLV,n16k64_wA_g32_n16,stock_wA_n16,stock_wA_n32,stock_wA_n64,stock_wA_e64
MAPS256="--maps256 llama8b=$A/llama8b_tc_256x64.mixfp4map --maps256 mistral7b=$A/mistral7b_tc_256x64.mixfp4map
         --maps256 phi4=$A/phi4_tc_256x64.mixfp4map --maps256 qwen27b=$A/qwen27b_tc_256x64.mixfp4map"
mkdir -p $R $OUT/logs $OUT/gemm $TB
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
# One `build.py --selftest` process per new build (the 256x64 path's four), into $ST, with its registered define.
selftests() {
  local pids=() rc=0 c
  for c in $NEW; do
    SM120_BUILD_DIR=$ST $PY sm120/build.py --selftest --define MIXFP4_UNIFORM_DISPATCH=1 --config $c \
      > $OUT/logs/selftest_$c.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || rc=1; done
  cat $OUT/logs/selftest_*.log
  return $rc
}
log "BEGIN V (amendment 18) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_18.json \
     --out $R/g0_provenance.json
step G3_selftest selftests
step G3_selftest_sass $PY experiments/kernel_opt/check_same_sass.py --root $ST --like $BV --configs ${NEW// /,} \
     --out $R/g3_selftest_sass.json
step PATCHER_sites $PY experiments/kernel_opt/check_patcher_sites.py --tagged-roots $BVA --t0-roots $BVA,$BV,$ST \
     --out $R/patcher_sites.json
step G1_sass $PY experiments/kernel_opt/check_sass.py --after-root $BVA --tmopt-root $KO/build_tmopt --new n16k64_wA_g32_e64_t0 \
     --before-roots $KO/build,$KO/build_e0m3,$A1,$KO/build_4,$KO/build_T,$B7,$KO/build_W,$C3K,$KO/build_P2,$KO/build_P3,$KO/build_P5 \
     --out $R/g1_sass.json
step G2_sass $PY experiments/kernel_opt/check_sass.py --after-root $BV --tmopt-root $KO/build_tmopt --new $ALLV --only-new \
     --out $R/g2_sass.json
step G2u_sass $PY experiments/kernel_opt/check_uniform_sass.py --set V --root $BV --like-u $BU --like-a1 $A1 \
     --like-stock $B7 --out $R/g2u_sass.json
step G3_pytest env SM120_BUILD_DIR=$BV SM120_REF_BUILD_DIR=/home/dev/NVFP4-RaZeR/sm120/build $PY -m pytest \
     sm120/tests/test_gemm.py sm120/tests/test_select.py sm120/tests/test_g32.py -rs -p no:cacheprovider -v
step G4_bitwise_256x64 $PY experiments/kernel_opt/check_bitwise.py --family g32 --build-root $BV --candidates ${NEW// /,} \
     --set mixed256_ko --out $R/g4_bitwise_256x64.json
step G5_logits_256x64 $PY experiments/kernel_opt/check_model_logits.py --unit 256x64 --before set:mixed256 --before-root $A1 \
     --after set:mixed256_ko --after-root $BV --out $R/g5_logits_256x64.json
step G5_auto_256x64 env SM120_BUILD_DIR=$BV $PY experiments/kernel_opt/check_model_logits.py --unit 256x64 \
     --before set:mixed256 --before-root $A1 --after auto:auto --expect-family mixed256_ko --out $R/g5_auto_256x64.json
log "GATES PASSED"
# The table: today's widths (M1's build effect), the act-warm width tuning (4b's method), the scheduler rows (amendment
# 7's decisive rule), the decisive-margin sensitivity on the widths (reported only), and G5b on the candidate table
step TODAY_V $PY experiments/kernel_opt/V_tables.py today --out-dir $TB
step TUNE_V_widths env SM120_BUILD_DIR=$BV $PY sm120/bench/tune_tiles.py --families mixed256_ko --cold --act-warm --rounds 3 \
     --iters 30 $MAPS256 --out-dir $TB/widths
step COMPOSE_V $PY experiments/kernel_opt/V_tables.py compose --widths $TB/widths --out-dir $TB
step TUNE_V_schedule env SM120_BUILD_DIR=$BV $PY sm120/bench/tune_tiles.py --schedule --families mixed256_ko --cold \
     --act-warm --rounds 3 --iters 30 --width-table $TB/table_v0w.json $MAPS256 --out-dir $TB/schedule
step FINALIZE_V $PY experiments/kernel_opt/V_tables.py finalize --schedule $TB/schedule --out-dir $TB
step SENS_V $PY experiments/kernel_opt/V_tables.py sensitivity --widths $TB/widths --out-dir $TB
step G5b_logits_256x64 $PY experiments/kernel_opt/check_model_logits.py --unit 256x64 --before set:mixed256 --before-root $A1 \
     --after set:mixed256_ko --after-root $BV --after-table $TB/table_v.json --out $R/g5b_logits_256x64.json
log "TABLE DONE"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_V_isolated.py --model $m --v-root $BV --a1 $A1 --b7 $B7 --c3k $C3K \
       --table-v0 $TB/table_v0.json --table-v $TB/table_v.json --artifact fo6=$A/${m}_fo6 \
       --artifact tc_256x64=$A/${m}_tc_256x64 --out $OUT/gemm/$m.json
done
step C2V $PY experiments/kernel_opt/c2_V.py --v-root $BV --a1 $A1 --b7 $B7 --c3k $C3K --out $OUT/c2_V.json
log "MEASUREMENTS DONE"
step REPORT $PY experiments/kernel_opt/V_report.py --src $OUT --out-dir $R
log "DONE V"

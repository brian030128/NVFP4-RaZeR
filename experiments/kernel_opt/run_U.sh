#!/bin/bash
# kernel-opt amendment 17 (U): part A, the uniform-branch dispatch for 16x64 (widths 64/128) and the pipelined flag read
# for 8x64 (widths 64/128x64/128); part B, the re-tuned 16x64 width cells. Gates: G0 (provenance), G3 (self-tests, in a
# separate directory, tied to the registered device code), the patcher check, G1/G2/G2u (SASS), G3 (pytest), G4 (bitwise)
# and G5 (model logits). Then part B's tuning and table, G5b (logits on part B's table), M1 (GEMM, 4 models), C2U
# (4096^3) and the report (results/kernel_opt/PROTOCOL.md). Stops at the first failure. build_U and build_Uall were
# built CPU-only before registration (results/kernel_opt/U/build_U.sh).
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
BU=$KO/build_U
BUA=$KO/build_Uall
ST=$KO/build_Ust
B7=$KO/build_7
B7F=$KO/build_7freq
P3F=$KO/build_P3freq
C3K=$KO/build_C3k
P5=$KO/build_P5
OUT=$KO/U/run
R=results/kernel_opt/U
A=/home/dev/n16k64_campaign/paper/artifacts
K16=n16k64_wA_n16_t0,n16k64_wA_n32_t0,n16k64_wA_n64_t0,n16k64_wA_e64_t0
K8=n8k64_wB_m16_t0,n8k64_wB_m32_t0,n8k64_wB_m64_t0,n8k64_wB_n64_t0,n8k64_wB_t0
NEW16="n16k64_wA_n64_t0 n16k64_wA_e64_t0"
NEW8="n8k64_wB_m64_t0 n8k64_wB_n64_t0 n8k64_wB_t0"
CELLS=4096x4096@128,4096x14336@128,17408x5120@128,1024x4096@512,1024x5120@512,12288x5120@512
MAPS="--maps llama8b=$A/llama8b_tc_16x64.mixfp4map --maps mistral7b=$A/mistral7b_tc_16x64.mixfp4map
      --maps phi4=$A/phi4_tc_16x64.mixfp4map --maps qwen27b=$A/qwen27b_tc_16x64.mixfp4map"
TB=$R/table_b
mkdir -p $R $OUT/logs $OUT/gemm $TB
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
# One `build.py --selftest` process per build with new defines (part A's 16x64 and 8x64 builds), into $ST.
selftests() {
  local pids=() rc=0 c
  for c in $NEW16; do
    SM120_BUILD_DIR=$ST $PY sm120/build.py --selftest --define MIXFP4_DISPATCH_FREQ=1 --define MIXFP4_UNIFORM_DISPATCH=1 \
      --config $c > $OUT/logs/selftest_$c.log 2>&1 &
    pids+=($!)
  done
  for c in $NEW8; do
    SM120_BUILD_DIR=$ST $PY sm120/build.py --selftest --define MIXFP4_DISPATCH_FREQ=1 --define MIXFP4_PIPE_FLAGS=1 \
      --config $c > $OUT/logs/selftest_$c.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || rc=1; done
  cat $OUT/logs/selftest_*.log
  return $rc
}
log "BEGIN U (amendment 17) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_17.json \
     --out $R/g0_provenance.json
step G3_selftest selftests
step G3_selftest_sass $PY experiments/kernel_opt/check_same_sass.py --root $ST --like $BU \
     --configs ${NEW16// /,},${NEW8// /,} --out $R/g3_selftest_sass.json
step PATCHER_sites $PY experiments/kernel_opt/check_patcher_sites.py --tagged-roots $BUA --t0-roots $BUA,$BU,$ST \
     --out $R/patcher_sites.json
step G1_sass $PY experiments/kernel_opt/check_sass.py --after-root $BUA --tmopt-root $KO/build_tmopt --new none \
     --before-roots $KO/build,$KO/build_e0m3,$KO/build_A1,$KO/build_4,$KO/build_T,$B7,$KO/build_W,$C3K,$KO/build_P2,$KO/build_P3,$P5 \
     --out $R/g1_sass.json
step G2_sass $PY experiments/kernel_opt/check_sass.py --after-root $BU --tmopt-root $KO/build_tmopt --new $K16,$K8,stock_wB_e64 \
     --only-new --out $R/g2_sass.json
step G2u_sass $PY experiments/kernel_opt/check_uniform_sass.py --root $BU --like16 $B7F --like8 $P3F --out $R/g2u_sass.json
step G3_pytest env SM120_BUILD_DIR=$BU SM120_REF_BUILD_DIR=/home/dev/NVFP4-RaZeR/sm120/build $PY -m pytest \
     sm120/tests/test_gemm.py sm120/tests/test_select.py -rs -p no:cacheprovider -v
step G4_bitwise_16x64 $PY experiments/kernel_opt/check_bitwise.py --family wA --build-root $BU --candidates $K16 \
     --set mixed_ko --out $R/g4_bitwise_16x64.json
step G4_bitwise_8x64 $PY experiments/kernel_opt/check_bitwise.py --family wB --build-root $BU --candidates $K8 \
     --set mixed_wB_ko --out $R/g4_bitwise_8x64.json
step G5_logits_16x64 $PY experiments/kernel_opt/check_model_logits.py --unit 16x64 --before set:mixed_ko --before-root $B7F \
     --after set:mixed_ko --after-root $BU --out $R/g5_logits_16x64.json
step G5_logits_8x64 $PY experiments/kernel_opt/check_model_logits.py --unit 8x64 --before set:mixed_wB_ko --before-root $P3F \
     --after set:mixed_wB_ko --after-root $BU --out $R/g5_logits_8x64.json
log "GATES PASSED"
# Part B: the cells' widths (the 4b method on part A's builds), the table, the changed cells' scheduler rows (amendment
# 7's decisive rule), and G5b
step TUNE_B_widths env SM120_BUILD_DIR=$BU $PY sm120/bench/tune_tiles.py --families mixed_ko --cold --act-warm --rounds 3 \
     --iters 30 --cells $CELLS $MAPS --out-dir $TB/widths
step COMPOSE_B $PY experiments/kernel_opt/U_tables.py widths --tuned $TB/widths --out-dir $TB
CHANGED=$(tail -n 1 $OUT/logs/COMPOSE_B.log)
log "part B: changed cells: ${CHANGED:-none}"
if [ -n "$CHANGED" ]; then
  step TUNE_B_schedule env SM120_BUILD_DIR=$BU $PY sm120/bench/tune_tiles.py --schedule --families mixed_ko --cold \
       --act-warm --rounds 3 --iters 30 --width-table $TB/table_b0.json --cells $CHANGED $MAPS --out-dir $TB/schedule
  step FINALIZE_B $PY experiments/kernel_opt/U_tables.py finalize --out-dir $TB --scheduled $TB/schedule
else
  step FINALIZE_B $PY experiments/kernel_opt/U_tables.py finalize --out-dir $TB
fi
step G5b_logits_16x64 $PY experiments/kernel_opt/check_model_logits.py --unit 16x64 --before set:mixed_ko --before-root $B7F \
     --after set:mixed_ko --after-root $BU --after-table $TB/table_b.json --out $R/g5b_logits_16x64.json
log "PART B TABLE DONE"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_U_isolated.py --model $m --u-root $BU --b7 $B7 --b7freq $B7F --p3freq $P3F \
       --c3k $C3K --p5 $P5 --table-b $TB/table_b.json --artifact fo6=$A/${m}_fo6 --artifact tc_16x64=$A/${m}_tc_16x64 \
       --artifact tc_8x64=$A/${m}_tc_8x64 --out $OUT/gemm/$m.json
done
step C2U $PY experiments/kernel_opt/c2_U.py --u-root $BU --b7 $B7 --b7freq $B7F --p3freq $P3F --c3k $C3K --p5 $P5 \
     --out $OUT/c2_U.json
log "MEASUREMENTS DONE"
step REPORT $PY experiments/kernel_opt/U_report.py --src $OUT --out-dir $R
log "DONE U"

#!/bin/bash
# kernel-opt amendment 13 (the 8x64 plan's P5, decision a: the same-placement reference). Gates: G0 (provenance), G1/G2
# (SASS; the four new no-dispatch ceilings are E2M1-only and unpredicated), and pytest (the ceiling's widths are bitwise
# interchangeable). Then M1 (GEMM, 4 models) and the report (results/kernel_opt/PROTOCOL.md). Stops at the first failure.
# build_P5 holds every configuration at the P5 sources, built CPU-only before registration (P5/build_P5.sh). The 8x64 path
# and stock_wB tuned alike come from build_P3freq, and stock_ko from build_7, all on the tracked adopted table.
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
BP5=$KO/build_P5
BP3F=$KO/build_P3freq
B7=$KO/build_7
OUT=$KO/w8p5
R=results/kernel_opt/w8/p5
A=/home/dev/n16k64_campaign/paper/artifacts
CEIL=n8k64_wB_m16_nodisp_t0,n8k64_wB_m32_nodisp_t0,n8k64_wB_m64_nodisp_t0,n8k64_wB_n64_nodisp_t0
mkdir -p $R $OUT/logs $OUT/gemm
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN P5 (amendment 13) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_13.json \
     --out $R/g0_provenance.json
step G1_G2_sass $PY experiments/kernel_opt/check_sass.py --after-root $BP5 --tmopt-root $KO/build_tmopt --new $CEIL \
     --before-roots $KO/build,$KO/build_e0m3,$KO/build_A1,$KO/build_4,$KO/build_T,$B7,$KO/build_W,$KO/build_C3k,$KO/build_P2,$KO/build_P3 \
     --out $R/g1_g2_sass.json
step G3_pytest env SM120_BUILD_DIR=$BP5 $PY -m pytest sm120/tests/test_select.py -k "nodisp_wB_ko" -rs -p no:cacheprovider -v
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_w8p5_isolated.py --model $m --p3freq-root $BP3F --ceil-root $BP5 --b7 $B7 \
       --artifact fo6=$A/${m}_fo6 --artifact tc_8x64=$A/${m}_tc_8x64 --out $OUT/gemm/$m.json
done
log "MEASUREMENTS DONE"
step REPORT $PY experiments/kernel_opt/w8p5_report.py --src $OUT --out-dir $R
log "DONE P5"

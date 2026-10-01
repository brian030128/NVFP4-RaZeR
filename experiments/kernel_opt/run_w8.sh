#!/bin/bash
# kernel-opt amendment 10 (the 8x64 baseline): gates G0 (provenance) and G2 (census of the two diagnostic ceilings), then
# M1 (GEMM, 4 models) and C2w (4096^3 breakdown) (results/kernel_opt/PROTOCOL.md); stops at the first failure. The 8x64
# path is optimization 1b's mixed_wB set (kernel-opt build) and #2's build of it (build_freq), on the paper table; the
# references are stock_ko (build_7, adopted table), the paper stock and stock_wB (sm120/build).
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
KOPT=$KO/build
FREQ=$KO/build_freq
B7=$KO/build_7
W=$KO/build_W
OUT=$KO/w8
R=results/kernel_opt/w8
A=/home/dev/n16k64_campaign/paper/artifacts
mkdir -p $R $OUT/logs $OUT/gemm
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN 8x64 baseline (amendment 10) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_10.json \
     --out $R/g0_provenance.json
step G2_sass_ceilings $PY experiments/kernel_opt/check_sass.py --after-root $W --tmopt-root $KO/build_tmopt \
     --new n8k64_wB_nodisp,n8k64_wB_nodisp_t0 --only-new --out $R/g2_sass_ceilings.json
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_w8_isolated.py --model $m --kopt-root $KOPT --freq-root $FREQ --b7 $B7 \
       --artifact fo6=$A/${m}_fo6 --artifact tc_8x64=$A/${m}_tc_8x64 --out $OUT/gemm/$m.json
done
step C2w $PY experiments/kernel_opt/c2_w8.py --kopt-root $KOPT --freq-root $FREQ --w-root $W --b7 $B7 --out $OUT/c2_w8.json
log "MEASUREMENTS DONE"
step REPORT $PY experiments/kernel_opt/w8_report.py --src $OUT --out-dir $R
log "DONE 8x64 baseline"

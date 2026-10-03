#!/bin/bash
# kernel-opt amendment 19 (C3v, the E0M3-fraction sweep on the adopted paths of the three units): G0 (provenance), the
# 256x64, 16x64 and 8x64 sweeps, their tables and figures (results/kernel_opt/c3v/PROTOCOL.md); stops at the first
# failure. No new builds. Run with `bash`.
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
OUT=$KO/c3v
R=results/kernel_opt/c3v
SW="$PY experiments/kernel_opt/c3v_fraction.py --bv $KO/build_V --b7 $KO/build_7 --paper-root /home/dev/NVFP4-RaZeR/sm120/build"
mkdir -p $R $OUT/logs
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN C3v (amendment 19) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration $R/registration.json --out $R/g0_provenance.json
step SWEEP_256x64 $SW --unit 256x64 --bprev $KO/build_A1 --bceil $KO/build_C3k --out $OUT/c3v_256x64.json
step SWEEP_16x64 $SW --unit 16x64 --bprev $KO/build_7freq --bceil $KO/build_C3k --out $OUT/c3v_16x64.json
step SWEEP_8x64 $SW --unit 8x64 --bprev $KO/build_P3freq --bceil $KO/build_P5 --out $OUT/c3v_8x64.json
step ANALYZE $PY experiments/kernel_opt/c3v_analyze.py --src $OUT --out-dir $R \
     --c3k results/kernel_opt/c3k/C3k_raw.json --c3w results/kernel_opt/c3w/C3w_raw.json
for u in 256x64 16x64 8x64; do cp $OUT/c3v_$u.json $R/C3v_${u}_raw.json; done
log "DONE C3v"

#!/bin/bash
# kernel-opt amendment 8 (C3k, the E0M3-fraction sweep on the adopted path): G2 for the ceiling builds, the sweep, its
# tables and figures (results/kernel_opt/c3k/PROTOCOL.md); stops at the first failure. build_C3k holds the four
# nodisp_ko builds from the amendment 8 sources (CPU, before registration).
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
OUT=$KO/c3k
R=results/kernel_opt/c3k
mkdir -p $R $OUT/logs
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN C3k at $(git rev-parse HEAD)"
step G2_ceiling $PY experiments/kernel_opt/check_sass.py --after-root $KO/build_C3k --tmopt-root $KO/build_tmopt \
     --new n16k64_wA_nodisp_n16_t0,n16k64_wA_nodisp_n32_t0,n16k64_wA_nodisp_n64_t0,n16k64_wA_nodisp_e64_t0 --only-new \
     --out $R/g2_ceiling.json
step SWEEP $PY experiments/kernel_opt/c3k_fraction.py --b7 $KO/build_7 --b7freq $KO/build_7freq --bceil $KO/build_C3k \
     --paper-root /home/dev/NVFP4-RaZeR/sm120/build --out $OUT/c3k.json
step ANALYZE $PY experiments/kernel_opt/c3k_analyze.py --src $OUT/c3k.json --out-dir $R
cp $OUT/c3k.json $R/C3k_raw.json
log "DONE C3k"

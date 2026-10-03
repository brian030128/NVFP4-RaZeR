#!/bin/bash
# kernel-opt amendment 16 (C3w, the E0M3-fraction sweep on the adopted 8x64 path): G0 (provenance), the sweep, its tables
# and figures (results/kernel_opt/c3w/PROTOCOL.md); stops at the first failure. No new builds.
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
OUT=$KO/c3w
R=results/kernel_opt/c3w
mkdir -p $R $OUT/logs
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN C3w (amendment 16) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration $R/registration.json --out $R/g0_provenance.json
step SWEEP $PY experiments/kernel_opt/c3w_fraction.py --bp3freq $KO/build_P3freq --bp3 $KO/build_P3 --bceil $KO/build_P5 \
     --b7 $KO/build_7 --paper-root /home/dev/NVFP4-RaZeR/sm120/build --out $OUT/c3w.json
step ANALYZE $PY experiments/kernel_opt/c3w_analyze.py --src $OUT/c3w.json --out-dir $R --c3k results/kernel_opt/c3k/C3k_raw.json
cp $OUT/c3w.json $R/C3w_raw.json
log "DONE C3w"

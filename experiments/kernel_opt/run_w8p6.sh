#!/bin/bash
# kernel-opt amendment 14 (the 8x64 plan's P6, decision b): 'auto' routes 8x64 maps to the adopted 8x64 set. Gates: G0
# (provenance), the routing tests in three build directories, and G5 with the routed kernel. There is no build and no
# timing; the routing is in sm120/mixfp4_sm120/model.py (09f67e4). The directories:
# - build_P2freq: the adopted 8x64 path, the t0 builds with #2's dispatch;
# - sm120/build: the paper builds, where the route falls back to n8k64_wB;
# - build_P3: every configuration with the default dispatch, where the note must say so.
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
PAPER=/home/dev/NVFP4-RaZeR/sm120/build
OUT=$KO/w8p6
R=results/kernel_opt/w8/p6
mkdir -p $R $OUT/logs
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN P6 (amendment 14) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_14.json \
     --out $R/g0_provenance.json
step ROUTE_adopted env SM120_BUILD_DIR=$KO/build_P2freq $PY -m pytest sm120/tests/test_select.py -k auto_routes_8x64 -rs \
     -p no:cacheprovider -v
step ROUTE_paper env SM120_BUILD_DIR=$PAPER $PY -m pytest sm120/tests/test_select.py -k auto_routes_8x64 -rs \
     -p no:cacheprovider -v
step ROUTE_default env SM120_BUILD_DIR=$KO/build_P3 $PY -m pytest sm120/tests/test_select.py sm120/tests/test_g32.py \
     -k "auto_routes" -rs -p no:cacheprovider -v
step G5_auto env SM120_BUILD_DIR=$KO/build_P2freq $PY experiments/kernel_opt/check_model_logits.py --unit 8x64 \
     --before n8k64_wB --before-root $PAPER --after auto:auto --expect-family mixed_wB_ko --out $R/g5_auto.json
log "DONE P6"

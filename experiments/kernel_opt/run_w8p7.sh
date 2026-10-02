#!/bin/bash
# kernel-opt amendment 15 (the 8x64 plan's P7: the cumulative registered 8x64 run). Gates: G0 (provenance), and the
# routing test on build_P3freq ('auto' -> mixed_wB_ko, 'auto_stock_wB' -> stock_wB_ko). Then:
# - the in-graph split with the SM clock (D4's method): Llama-3.1-8B 1x2048 / 1x4096, and Phi-4 1x512 with three passes;
# - the end-to-end CUDA-graph prefill (4 models) and decode (3 models);
# - the reports (results/kernel_opt/PROTOCOL.md).
# Stops at the first failure. No build: the libraries are sm120/build's, build_7's and build_P3freq's.
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
OUT=$KO/w8p7
R=results/kernel_opt/w8/p7
mkdir -p $R $OUT/logs
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN P7 (amendment 15) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_15.json \
     --out $R/g0_provenance.json
step ROUTE env SM120_BUILD_DIR=$KO/build_P3freq $PY -m pytest sm120/tests/test_select.py -k auto_routes_8x64 -rs \
     -p no:cacheprovider -v
log "GATES PASSED"
step SPLIT_llama8b $PY experiments/kernel_opt/p7_split.py --model llama8b --shapes 1x2048,1x4096 --passes 1 \
     --out $OUT/split_llama8b.json
step SPLIT_phi4 $PY experiments/kernel_opt/p7_split.py --model phi4 --shapes 1x512 --passes 3 --out $OUT/split_phi4.json
step E2E_prefill $PY experiments/kernel_opt/cum8_e2e.py --what prefill --out $OUT/e2e
step E2E_decode $PY experiments/kernel_opt/cum8_e2e.py --what decode --out $OUT/e2e
log "MEASUREMENTS DONE"
step REPORT_e2e $PY experiments/kernel_opt/cum8_e2e_report.py --src $OUT/e2e --out-dir $R
step REPORT_split $PY experiments/kernel_opt/p7_split_report.py --src $OUT/split_llama8b.json --src $OUT/split_phi4.json \
     --out-dir $R
log "DONE P7"

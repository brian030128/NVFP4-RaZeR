#!/bin/bash
# kernel-opt amendment 3 (A'): gates G3'', G1''/G2'', G-span, G4'', G5'', then M1'' and C2'' (results/kernel_opt/PROTOCOL.md);
# stops at the first failure. build_A1 holds `build.py --all` from the A' sources (CPU, before registration).
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
A1=$KO/build_A1
OUT=$KO/A1
R=results/kernel_opt/A1
A=/home/dev/n16k64_campaign/paper/artifacts
G32=n16k64_wA_g32,n16k64_wA_g32_n64,n16k64_wA_g32_n32,n16k64_wA_g32_n16
mkdir -p $R $OUT/logs $OUT/gemm
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN A' gates and measurements at $(git rev-parse HEAD)"
step G3_selftest env SM120_BUILD_DIR=$A1 $PY sm120/build.py --selftest \
     --config n16k64_wA_g32 --config n16k64_wA_g32_n64 --config n16k64_wA_g32_n32 --config n16k64_wA_g32_n16
step G1_G2_sass $PY experiments/kernel_opt/check_sass.py --after-root $A1 --tmopt-root $KO/build_tmopt --new $G32 \
     --before-roots $KO/build,$KO/build_e0m3 --out $R/g1_g2_sass.json
step G3_pytest env SM120_BUILD_DIR=$A1 SM120_REF_BUILD_DIR=/home/dev/NVFP4-RaZeR/sm120/build $PY -m pytest \
     sm120/tests/test_gemm.py sm120/tests/test_select.py sm120/tests/test_g32.py -rs -p no:cacheprovider -v
step G_span $PY experiments/kernel_opt/check_granule_map.py --a1-root $A1 --out $R/g_span_granule_map.json
step G4_bitwise_g32 $PY experiments/kernel_opt/check_bitwise.py --family g32 --build-root $A1 --candidates $G32 \
     --set mixed256 --out $R/g4_bitwise_g32.json
step G5_logits_256x64 $PY experiments/kernel_opt/check_model_logits.py --unit 256x64 --before set:mixed --after set:mixed256 \
     --after-root $A1 --out $R/g5_logits_256x64.json
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_abA1_isolated.py --model $m --a1-root $A1 \
       --artifact fo6=$A/${m}_fo6 --artifact tc_256x64=$A/${m}_tc_256x64 --out $OUT/gemm/$m.json
done
step C2_g32 $PY experiments/kernel_opt/c2_g32.py --a1-root $A1 --out $OUT/c2_g32.json
log "DONE A' gates and measurements"

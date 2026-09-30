#!/bin/bash
# kernel-opt amendment 2 (#2): gates G1'-G5', then M1' and C2' (results/kernel_opt/PROTOCOL.md); stops at the first failure
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
FREQ=$KO/build_freq
OUT=$KO/opt2
R=results/kernel_opt/opt2
A=/home/dev/n16k64_campaign/paper/artifacts
mkdir -p $R
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN #2 gates and measurements at $(git rev-parse HEAD)"
# G3' self-tests (rebuilds these four in build_freq with the self-test driver; same define)
step G3_selftest env SM120_BUILD_DIR=$FREQ $PY sm120/build.py --selftest --define MIXFP4_DISPATCH_FREQ=1 \
     --config n16k64_wA --config n16k64_wA_n16 --config n8k64_wB --config n8k64_wB_m16
# G1' / G2': the hook unset is inert (build_hookcheck), non-dispatching builds in build_freq keep their SASS,
# dispatching ones keep their census
step G1_hookcheck $PY - <<'PYEOF'
import json, sys
bad = []
for c in ('n16k64_wA', 'n8k64_wB'):
    a = json.load(open(f'/home/dev/n16k64_campaign/kernel_opt/build_hookcheck/{c}/manifest.json'))
    b = json.load(open(f'/home/dev/NVFP4-RaZeR/sm120/build/{c}/manifest.json'))
    for key in ('sass_sha256', 'unpatched_sass_sha256'):
        print(c, key, a[key] == b[key])
        bad += [] if a[key] == b[key] else [(c, key)]
sys.exit(1 if bad else 0)
PYEOF
step G1_G2_sass $PY experiments/kernel_opt/check_sass.py --after-root $FREQ --tmopt-root $KO/build_tmopt \
     --new n16k64_wA,n16k64_wA_8x1,n16k64_wA_sk,n16k64_wA_n64,n16k64_wA_n32,n16k64_wA_n16,n8k64_wB,n8k64_wB_m64,n8k64_wB_m32,n8k64_wB_m16,n8k64_wB_n64,n16k64_wA_xor,n8k64_wB_xor \
     --out $R/g1_g2_sass.json
step G3_pytest env SM120_BUILD_DIR=$FREQ $PY -m pytest sm120/tests/test_gemm.py sm120/tests/test_select.py -rs -p no:cacheprovider -v
step G4_bitwise_wA $PY experiments/kernel_opt/check_bitwise.py --family wA --build-root $FREQ \
     --candidates n16k64_wA,n16k64_wA_n64,n16k64_wA_n32,n16k64_wA_n16,n16k64_wA_8x1 --set mixed --out $R/g4_bitwise_wA.json
step G4_bitwise_wB $PY experiments/kernel_opt/check_bitwise.py --family wB --build-root $FREQ \
     --candidates n8k64_wB,n8k64_wB_m64,n8k64_wB_m32,n8k64_wB_m16,n8k64_wB_n64 --set mixed_wB --out $R/g4_bitwise_wB.json
for u in 16x64 256x64; do
  step G5_logits_$u $PY experiments/kernel_opt/check_model_logits.py --unit $u --before set:mixed --after set:mixed \
       --after-root $FREQ --out $R/g5_logits_$u.json
done
step G5_logits_8x64 $PY experiments/kernel_opt/check_model_logits.py --unit 8x64 --before n8k64_wB --after set:mixed_wB \
     --after-root $FREQ --out $R/g5_logits_8x64.json
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_ab2_isolated.py --model $m --freq-root $FREQ --kopt-root $KO/build \
       --artifact fo6=$A/${m}_fo6 --artifact tc_16x64=$A/${m}_tc_16x64 --artifact tc_256x64=$A/${m}_tc_256x64 \
       --artifact tc_8x64=$A/${m}_tc_8x64 --out $OUT/gemm/$m.json
done
step C2_freq $PY experiments/kernel_opt/c2_freq.py --freq-root $FREQ --out $OUT/c2_freq.json
log "DONE #2 gates and measurements"

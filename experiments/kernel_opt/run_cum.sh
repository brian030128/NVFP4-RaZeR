#!/bin/bash
# kernel-opt amendment 9 (the cumulative registered run): gates G0 (provenance), G2 (census), G3 (pytest), G4 (bitwise),
# G5 (model logits), then M1 (GEMM, 4 models), the end-to-end prefill (4 models) and decode (3 models)
# (results/kernel_opt/PROTOCOL.md); stops at the first failure. The combined 16x64 build is amendment 7's build_7freq
# ('mixed_ko' with MIXFP4_DISPATCH_FREQ=1), the adopted stock set build_7's 'stock_ko', both on the adopted table; the
# references are today's paper builds in sm120/build on the paper table.
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
B7=$KO/build_7
B7F=$KO/build_7freq
OUT=$KO/cum
R=results/kernel_opt/cum
A=/home/dev/n16k64_campaign/paper/artifacts
KOT=sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.ko.json
KO16=n16k64_wA_n16_t0,n16k64_wA_n32_t0,n16k64_wA_n64_t0,n16k64_wA_e64_t0
STOCKKO=stock_wA_n16,stock_wA_n32,stock_wA_n64,stock_wA_e64
mkdir -p $R $OUT/logs $OUT/gemm
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN cumulative run (amendment 9) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_9.json \
     --out $R/g0_provenance.json
step G2_sass_freq $PY experiments/kernel_opt/check_sass.py --after-root $B7F --tmopt-root $KO/build_tmopt --new $KO16 \
     --only-new --out $R/g2_sass_freq.json
step G2_sass_ko $PY experiments/kernel_opt/check_sass.py --after-root $B7 --tmopt-root $KO/build_tmopt --new $KO16,$STOCKKO \
     --only-new --out $R/g2_sass_ko.json
step G3_pytest_freq env SM120_BUILD_DIR=$B7F $PY -m pytest sm120/tests/test_gemm.py sm120/tests/test_select.py -rs \
     -p no:cacheprovider -v
step G3_pytest_ko env SM120_BUILD_DIR=$B7 $PY -m pytest sm120/tests/test_select.py -k "mixed_ko or stock_ko" -rs \
     -p no:cacheprovider -v
step G4_bitwise_freq $PY experiments/kernel_opt/check_bitwise.py --family wA --build-root $B7F --candidates $KO16 \
     --set mixed_ko --table $KOT --out $R/g4_bitwise_freq.json
step G5_logits_16x64 $PY experiments/kernel_opt/check_model_logits.py --unit 16x64 --before set:mixed --after set:mixed_ko \
     --after-root $B7F --after-table $KOT --out $R/g5_logits_16x64.json
step G5_logits_fo6 $PY experiments/kernel_opt/check_model_logits.py --unit fo6 --before set:stock --after set:stock_ko \
     --after-root $B7 --after-table $KOT --out $R/g5_logits_fo6.json
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_cum_isolated.py --model $m --b7 $B7 --b7freq $B7F \
       --artifact fo6=$A/${m}_fo6 --artifact tc_16x64=$A/${m}_tc_16x64 --out $OUT/gemm/$m.json
done
step E2E_prefill $PY experiments/kernel_opt/cum_e2e.py --what prefill --out $OUT/e2e
step E2E_decode $PY experiments/kernel_opt/cum_e2e.py --what decode --out $OUT/e2e
log "MEASUREMENTS DONE"
step REPORT_gemm $PY experiments/kernel_opt/cum_report.py --src $OUT --out-dir $R
step REPORT_e2e $PY experiments/kernel_opt/cum_e2e_report.py --src $OUT/e2e --out-dir $R
log "DONE cumulative run"

#!/bin/bash
# kernel-opt amendment 12b (the 8x64 plan's P3b, option C). Gates: G0 (provenance), G4 (bitwise; the reduced table's
# set) and G5 (model logits). Then a fresh M1 (GEMM, 4 models) and the report (results/kernel_opt/PROTOCOL.md). Stops at
# the first failure. No build and no source change: the libraries are build_P3freq's (amendment 12). The reduced table
# and the fallback were composed CPU-only before registration (p3b_tables.py).
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
BP3F=$KO/build_P3freq
BP2F=$KO/build_P2freq
B7=$KO/build_7
OUT=$KO/w8p3b
R=results/kernel_opt/w8/p3b
A=/home/dev/n16k64_campaign/paper/artifacts
SLUG=nvidia_rtx_pro_6000_blackwell_workstation_edition
TABLE=$R/table_p3b/$SLUG.json
WBT0=n8k64_wB_t0,n8k64_wB_m64_t0,n8k64_wB_m32_t0,n8k64_wB_m16_t0,n8k64_wB_n64_t0
mkdir -p $R $OUT/logs $OUT/gemm
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN P3b (amendment 12b) at $(git rev-parse HEAD)"
step G0_provenance $PY experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_12b.json \
     --out $R/g0_provenance.json
step G4_bitwise $PY experiments/kernel_opt/check_bitwise.py --family wB --build-root $BP3F --candidates $WBT0 \
     --set mixed_wB_ko --table $TABLE --out $R/g4_bitwise.json
step G5_logits $PY experiments/kernel_opt/check_model_logits.py --unit 8x64 --before set:mixed_wB_t0 --before-root $BP2F \
     --after set:mixed_wB_ko --after-root $BP3F --after-table $TABLE --out $R/g5_logits.json
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_w8p3b_isolated.py --model $m --p3freq-root $BP3F --b7 $B7 --table $TABLE \
       --artifact fo6=$A/${m}_fo6 --artifact tc_8x64=$A/${m}_tc_8x64 --out $OUT/gemm/$m.json
done
log "MEASUREMENTS DONE"
step REPORT $PY experiments/kernel_opt/w8p3b_report.py --src $OUT --out-dir $R
log "DONE P3b"

#!/bin/bash
# kernel-opt amendment 4 (tile-table re-tune): TUNE, gates GW1/GW2, then M1''' (results/kernel_opt/PROTOCOL.md); stops at
# the first failure. The new table is written to results/kernel_opt/retune/tables; the tracked table is not touched.
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
FREQ=$KO/build_freq
A1=$KO/build_A1
PAPER=/home/dev/NVFP4-RaZeR/sm120/build
OUT=$KO/retune
R=results/kernel_opt/retune
A=/home/dev/n16k64_campaign/paper/artifacts
NEW=$R/tables/nvidia_rtx_pro_6000_blackwell_workstation_edition.json
mkdir -p $R/tables $OUT/logs $OUT/gemm
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN re-tune at $(git rev-parse HEAD)"
step TUNE env SM120_BUILD_DIR=$FREQ $PY sm120/bench/tune_tiles.py --families mixed,stock --cold --rounds 3 --iters 30 \
     --maps llama8b=$A/llama8b_tc_16x64.mixfp4map --maps mistral7b=$A/mistral7b_tc_16x64.mixfp4map \
     --maps phi4=$A/phi4_tc_16x64.mixfp4map --maps qwen27b=$A/qwen27b_tc_16x64.mixfp4map --out-dir $R/tables
step GW1_widths_freq env SM120_BUILD_DIR=$FREQ $PY -m pytest sm120/tests/test_select.py -rs -p no:cacheprovider -v
step GW1_widths_A1 env SM120_BUILD_DIR=$A1 $PY -m pytest sm120/tests/test_select.py -rs -p no:cacheprovider -v
step GW2_logits_16x64 $PY experiments/kernel_opt/check_model_logits.py --unit 16x64 --before set:mixed --before-root $FREQ \
     --after set:mixed --after-root $FREQ --after-table $NEW --out $R/gw2_logits_16x64.json
step GW2_logits_256x64 $PY experiments/kernel_opt/check_model_logits.py --unit 256x64 --before set:mixed256 --before-root $A1 \
     --after set:mixed256 --after-root $A1 --after-table $NEW --out $R/gw2_logits_256x64.json
step GW2_logits_fo6 $PY experiments/kernel_opt/check_model_logits.py --unit fo6 --before set:stock \
     --after set:stock --after-root $PAPER --after-table $NEW --out $R/gw2_logits_fo6.json
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_ab_retune.py --model $m --new-table $NEW --freq-root $FREQ --a1-root $A1 \
       --artifact fo6=$A/${m}_fo6 --artifact tc_16x64=$A/${m}_tc_16x64 --artifact tc_256x64=$A/${m}_tc_256x64 \
       --out $OUT/gemm/$m.json
done
log "DONE re-tune"

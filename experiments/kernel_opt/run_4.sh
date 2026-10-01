#!/bin/bash
# kernel-opt amendment 5 (#4: epilogue tile and tile-scheduler order): the schedule tuning, gates G3/G1/G2/pytest/G4/G5, then
# M1 (results/kernel_opt/PROTOCOL.md); stops at the first failure. build_4 holds `build.py --all` from the #4 sources (CPU,
# before registration).
set -e
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONPATH=/home/dev/NVFP4-RaZeR/sm120:/home/dev/NVFP4-RaZeR PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
KO=/home/dev/n16k64_campaign/kernel_opt
B4=$KO/build_4
FREQ=$KO/build_freq
OUT=$KO/4
R=results/kernel_opt/4
A=/home/dev/n16k64_campaign/paper/artifacts
CUR=sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.json
NEW=$R/table/nvidia_rtx_pro_6000_blackwell_workstation_edition.json
NEWB=n16k64_wA_e64,n16k64_wA_n64_e64,stock_wA_e64,stock_wA_n64_e64
mkdir -p $R/table $OUT/logs $OUT/gemm
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN #4 at $(git rev-parse HEAD)"
step G3_selftest env SM120_BUILD_DIR=$B4 $PY sm120/build.py --selftest --config n16k64_wA_e64 --config n16k64_wA_n64_e64
step G1_G2_sass $PY experiments/kernel_opt/check_sass.py --after-root $B4 --tmopt-root $KO/build_tmopt --new $NEWB \
     --before-roots $KO/build,$KO/build_e0m3,$KO/build_A1 --out $R/g1_g2_sass.json
step SCHED_TUNE env SM120_BUILD_DIR=$B4 $PY sm120/bench/tune_tiles.py --schedule --families mixed_e,stock_e --cold --act-warm \
     --rounds 3 --iters 30 --width-table $CUR --maps llama8b=$A/llama8b_tc_16x64.mixfp4map \
     --maps mistral7b=$A/mistral7b_tc_16x64.mixfp4map --maps phi4=$A/phi4_tc_16x64.mixfp4map \
     --maps qwen27b=$A/qwen27b_tc_16x64.mixfp4map --out-dir $R/table
step G3_pytest env SM120_BUILD_DIR=$B4 $PY -m pytest sm120/tests/test_gemm.py sm120/tests/test_select.py -rs -p no:cacheprovider -v
step G4_bitwise $PY experiments/kernel_opt/check_bitwise.py --family wA --build-root $B4 \
     --candidates n16k64_wA_e64,n16k64_wA_n64_e64 --set mixed_e --table $NEW --out $R/g4_bitwise.json
step G5_logits_16x64 $PY experiments/kernel_opt/check_model_logits.py --unit 16x64 --before set:mixed --after set:mixed_e \
     --after-root $B4 --after-table $NEW --out $R/g5_logits_16x64.json
step G5_logits_fo6 $PY experiments/kernel_opt/check_model_logits.py --unit fo6 --before set:stock --after set:stock_e \
     --after-root $B4 --after-table $NEW --out $R/g5_logits_fo6.json
log "GATES PASSED"
for m in llama8b mistral7b phi4 qwen27b; do
  step M1_$m $PY experiments/kernel_opt/bench_ab4_isolated.py --model $m --root4 $B4 --freq-root $FREQ --a1-root $KO/build_A1 \
       --table4 $NEW --artifact fo6=$A/${m}_fo6 --artifact tc_16x64=$A/${m}_tc_16x64 --artifact tc_256x64=$A/${m}_tc_256x64 \
       --out $OUT/gemm/$m.json
done
log "DONE #4"

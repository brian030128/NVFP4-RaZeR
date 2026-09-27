#!/bin/bash
# Task 1 (results/nodev_cost/PROTOCOL.md): calibration cost without the development set.
# (a) TM-OPT+TC --no-dev --no-eval, deterministic: Llama/Mistral/Phi-4 x 3 units, Qwen 8x64; (b) after each run its
# map.pt must equal the committed map (sha256), else STOP; (c) Llama QAT C1 and scale-only D1 --no-dev, deterministic
# off and on (a deterministic run that cannot run is logged FAIL and skipped), and TM-OPT+TC deterministic off.
D=/home/dev/n16k64_campaign/nodev_cost
R=$D/runs
TM=/home/dev/n16k64_campaign/tm_opt/runs
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=$D/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R $D/logs
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $D/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
try_run () { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $D/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
             [ $rc -eq 0 ] || log "FAIL (logged, continuing): $name"; }
root () { [ $1 = llama8b ] && echo /home/dev/n16k64_campaign/cost_comparison/data || echo /home/dev/n16k64_campaign/multimodel/data; }
deviation () { [ $1 = llama8b ] && echo --transformers-deviation; }
mapdir () { [ $1 = qwen27b ] && echo $TM/q_tc_qwen27b_$2 || echo $TM/tc_$1_$2; }
check () {
  local new old
  new=$(sha256sum $1/map.pt | cut -d' ' -f1)
  old=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['map_sha256'])" $2/report.json)
  if [ "$new" = "$old" ]; then log "BITWISE OK $(basename $1) = $(basename $2) ($new)"
  else log "STOP bitwise mismatch $(basename $1): $new vs $(basename $2) $old"; exit 1; fi; }
TC="--tm-opt --tile-grad-tc --param ste --lr 0.02 --init-logit -1 --epochs 20 --eval-every 2 --no-dev --no-eval"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
for M in llama8b mistral7b phi4; do
  for U in 8x64 16x64 256x64; do
    run nd_tc_${M}_$U $PY run_train_map.py --model $M --data-root $(root $M) $(deviation $M) --unit $U $TC --out $R/nd_tc_${M}_$U
    check $R/nd_tc_${M}_$U $(mapdir $M $U)
  done
done
run nd_tc_qwen27b_8x64 $PY run_train_map.py --model qwen27b --data-root $(root qwen27b) --gpus 1 --unit 8x64 $TC --batch 2 --accum 4 \
    --out $R/nd_tc_qwen27b_8x64
check $R/nd_tc_qwen27b_8x64 $(mapdir qwen27b 8x64)
log "PART A DONE (all maps bitwise equal)"
QPATH=$REPO:/home/dev/n16k64_campaign/cost_comparison/pydeps
for DET in off on; do
  if [ $DET = on ]; then DF=--deterministic; else DF=--no-deterministic; fi
  try_run nd_qat_c1_det$DET env PYTHONPATH=$QPATH $PY run_cost_distill.py --arm qat --optimizer adamw_fp32 --micro-batch 8 \
      --checkpointing --lr 1e-6 --budget c1 --no-dev $DF --data-root $(root llama8b) --transformers-deviation --out $R/nd_qat_c1_det$DET
  try_run nd_scale_d1_det$DET env PYTHONPATH=$QPATH $PY run_cost_distill.py --arm scale --micro-batch 8 --lr 1e-3 --budget c1 \
      --no-dev $DF --data-root $(root llama8b) --transformers-deviation --out $R/nd_scale_d1_det$DET
done
for U in 8x64 16x64 256x64; do
  run nd_tc_llama8b_${U}_detoff $PY run_train_map.py --model llama8b --data-root $(root llama8b) --transformers-deviation --unit $U \
      $TC --no-deterministic --out $R/nd_tc_llama8b_${U}_detoff
done
log "TASK 1 RUNS DONE"

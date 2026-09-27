#!/bin/bash
# Task 2 (results/scale_additivity/PROTOCOL.md): scale training and TM-OPT+TC on Llama-3.1-8B, 16x64 and 8x64.
# 1. SCALE's learning rate on the development set (grid, one edge extension); 2. SCALE deployed (--no-dev);
# 3. SCALE->OURS; 4. OURS->SCALE; 5. artifacts (learned scales, ownership); 6. NativeLinear (c) evaluation, one process.
# Any failed step stops the queue.
D=/home/dev/n16k64_campaign/scale_additivity
R=$D/runs
A=$D/artifacts
TM=/home/dev/n16k64_campaign/tm_opt/runs
DE=/home/dev/n16k64_campaign/deploy_eval/artifacts
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
ROOT=/home/dev/n16k64_campaign/cost_comparison/data
LOG=$D/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131
QPATH=$REPO:/home/dev/n16k64_campaign/cost_comparison/pydeps
mkdir -p $R $A $D/logs
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $D/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
SC="--arm scale --micro-batch 8 --budget c1 --epochs 20 --act-rows --deterministic --data-root $ROOT --transformers-deviation"
TC="--tm-opt --tile-grad-tc --param ste --lr 0.02 --init-logit -1 --epochs 20 --eval-every 2 --no-dev --no-eval"
for LR in 1e-4 3e-4 1e-3; do
  run scale_dev_lr$LR env PYTHONPATH=$QPATH $PY run_cost_distill.py $SC --lr $LR --out $R/scale_dev_lr$LR
done
EXTRA=$($PY results/scale_additivity/choose_lr.py $R edge) || { log "STOP choose_lr edge failed"; exit 1; }
if [ -n "$EXTRA" ]; then
  log "EDGE extension: $EXTRA"
  run scale_dev_lr$EXTRA env PYTHONPATH=$QPATH $PY run_cost_distill.py $SC --lr $EXTRA --out $R/scale_dev_lr$EXTRA
fi
LR=$($PY results/scale_additivity/choose_lr.py $R choose) || { log "STOP choose_lr failed"; exit 1; }
log "CHOSEN learning rate $LR ($(cat $R/lr_choice.json | tr -d '\n '))"
run scale_nodev env PYTHONPATH=$QPATH $PY run_cost_distill.py $SC --lr $LR --no-dev --out $R/scale_nodev
for U in 16x64 8x64; do
  run scale_ours_$U $PY run_train_map.py --model llama8b --data-root $ROOT --transformers-deviation --unit $U $TC \
      --base-scales $R/scale_nodev/state.pt --out $R/scale_ours_$U
done
for U in 16x64 8x64; do
  run ours_scale_$U env PYTHONPATH=$QPATH $PY run_cost_distill.py $SC --lr $LR --no-dev --map $TM/tc_llama8b_$U/map.pt --unit $U \
      --out $R/ours_scale_$U
done
run art_scale $PY export_map_artifact.py --model llama8b --data-root $ROOT --kind four_over_six --scales $R/scale_nodev/state.pt \
    --ownership --out $A/scale
for U in 16x64 8x64; do
  run art_scale_ours_$U $PY export_map_artifact.py --model llama8b --data-root $ROOT --map $R/scale_ours_$U/map.pt --unit $U \
      --policy-name 'SCALE->TM-OPT+TC' --scales $R/scale_nodev/state.pt --scales-apply e2m1 --ownership --out $A/scale_ours-$U
  run art_ours_scale_$U $PY export_map_artifact.py --model llama8b --data-root $ROOT --map $TM/tc_llama8b_$U/map.pt --unit $U \
      --policy-name 'TM-OPT+TC->SCALE' --scales $R/ours_scale_$U/state.pt --scales-apply all --ownership --out $A/ours_scale-$U
done
EV="--evaluate BF16=bf16 --evaluate FourOverSix=native:$DE/llama8b_fo6 --evaluate NVFP4=native:$DE/llama8b_nvfp4"
EV="$EV --evaluate scale=native:$A/scale"
for U in 16x64 8x64; do
  EV="$EV --evaluate ours-$U=native:$DE/llama8b_tc_$U --evaluate scale_ours-$U=native:$A/scale_ours-$U"
  EV="$EV --evaluate ours_scale-$U=native:$A/ours_scale-$U"
done
run eval_llama8b $PY run_ppl_deploy.py --model llama8b --data-root $ROOT --transformers-deviation $EV --out $R/llama8b
log "TASK 2 RUNS DONE"

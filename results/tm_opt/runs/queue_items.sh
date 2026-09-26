#!/bin/bash
# Items #2 and #1 (results/tm_opt/PROTOCOL_ITEMS.md): TM-OPT seeds 1 and 2 on Llama 8x64 and their evaluation, then
# TM-OPT on Llama 16x64/256x64 and Mistral/Phi-4 8x64/16x64/256x64, each model followed by its evaluations.
R=/home/dev/n16k64_campaign/tm_opt/runs
L=/home/dev/n16k64_campaign/tm_opt/logs
M=/home/dev/n16k64_campaign/mr_variants/runs
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=/home/dev/n16k64_campaign/tm_opt/commands_items.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R $L
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $L/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
data () { if [ $1 = llama8b ]; then echo "--data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation";
          else echo "--data-root /home/dev/n16k64_campaign/multimodel/data"; fi; }
evalargs () { echo "--model $1 $(data $1) --objective kl --memory-mode lean --unit 8x64 --fused-act-quant --single-pass-epilogue"; }
TM="--tm-opt --param ste --lr 0.02 --init-logit -1 --epochs 20 --eval-every 2"

# #2: seeds 1 and 2 (seed 0 = results/tm_opt/runs/g2_tmopt), one native and one fake evaluation
for S in 1 2; do
  run tm_llama8b_8x64_seed$S $PY run_train_map.py --model llama8b $(data llama8b) --unit 8x64 $TM --seed $S --out $R/tm_llama8b_8x64_seed$S
done
SEEDS="--evaluate-map FourOverSix=fourover6 --evaluate-map tm-seed0=$R/g2_tmopt/map.pt"
SEEDS="$SEEDS --evaluate-map tm-seed1=$R/tm_llama8b_8x64_seed1/map.pt --evaluate-map tm-seed2=$R/tm_llama8b_8x64_seed2/map.pt"
run seeds_eval_native $PY run_multiround.py $(evalargs llama8b) --eval-backend native $SEEDS --evaluate-map mropt-8x64=$M/llama8b/mropt_8x64/map.pt --out $R/seeds_eval_native
run seeds_eval_fake $PY run_multiround.py $(evalargs llama8b) --eval-backend fake $SEEDS --out $R/seeds_eval_fake
log "ITEM2 DONE"

# #1: TM-OPT vs MR-OPT, model by model
for MOD in llama8b mistral7b phi4; do
  if [ $MOD = llama8b ]; then NEW="16x64 256x64"; else NEW="8x64 16x64 256x64"; fi
  for U in $NEW; do
    run tm_${MOD}_$U $PY run_train_map.py --model $MOD $(data $MOD) --unit $U $TM --out $R/tm_${MOD}_$U
  done
  NM="--evaluate-map FourOverSix=fourover6"; FM="--evaluate-map FourOverSix=fourover6"
  for U in 8x64 16x64 256x64; do
    if [ $MOD = llama8b ] && [ $U = 8x64 ]; then P=$R/g2_tmopt/map.pt; else P=$R/tm_${MOD}_$U/map.pt; fi
    NM="$NM --evaluate-map tmopt-$U=$P"; FM="$FM --evaluate-map tmopt-$U=$P"
  done
  for U in 8x64 16x64 256x64; do NM="$NM --evaluate-map mropt-$U=$M/$MOD/mropt_$U/map.pt"; done
  run items_${MOD}_eval_native $PY run_multiround.py $(evalargs $MOD) --eval-backend native $NM --out $R/items_${MOD}_eval_native
  run items_${MOD}_eval_fake $PY run_multiround.py $(evalargs $MOD) --eval-backend fake $FM --out $R/items_${MOD}_eval_fake
  log "ITEM1 MODEL DONE $MOD"
done
log "ITEM1 DONE (the GPU is idle for #3)"

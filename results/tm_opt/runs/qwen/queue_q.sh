#!/bin/bash
# Part Q (results/tm_opt/PROTOCOL_QR.md): the Qwen3.8-27B batch probe (micro-batch 8 -> 4 -> 2 -> 1, one optimizer step
# of 8 sequences each), then TM-OPT and TM-OPT+TC at 8x64, 16x64, 256x64 with the largest micro-batch that fits
# (optimizer batch 8 through --accum), then the evaluations. If micro-batch 1 does not fit, stop and report.
R=/home/dev/n16k64_campaign/tm_opt/runs
L=/home/dev/n16k64_campaign/tm_opt/logs
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=/home/dev/n16k64_campaign/tm_opt/commands_q.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R $L
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $L/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
QWEN="--model qwen27b --data-root /home/dev/n16k64_campaign/multimodel/data --gpus 1"
TM="--tm-opt --param ste --lr 0.02 --init-logit -1 --epochs 20 --eval-every 2"

# batch probe (TM-OPT, 8x64): FIT or OOM per micro-batch, with the per-phase peak memory in each report
CHOSEN=""
for B in 8 4 2 1; do
  A=$((8 / B))
  log "START qwen_probe_b$B micro-batch $B accum $A"
  $PY run_train_map.py $QWEN --unit 8x64 $TM --batch $B --accum $A --probe-steps 1 --out $R/qwen_probe_b$B > $L/qwen_probe_b$B.log 2>&1
  rc=$?
  log "END qwen_probe_b$B rc=$rc"
  if [ $rc -eq 0 ] && grep -q '"status": "probe_complete"' $R/qwen_probe_b$B/report.json; then CHOSEN=$B; log "PROBE FIT micro-batch $B (accum $A)"; break; fi
  if grep -q '"status": "out_of_memory"' $R/qwen_probe_b$B/report.json 2>/dev/null; then log "PROBE OOM micro-batch $B"; continue; fi
  log "STOP probe failed for another reason: micro-batch $B"; exit 1
done
[ -n "$CHOSEN" ] || { log "STOP: even micro-batch 1 does not fit (no offloading improvised)"; exit 1; }
echo $CHOSEN > $R/qwen_micro_batch.txt
A=$((8 / CHOSEN))
log "QWEN PROBE DONE: micro-batch $CHOSEN, accum $A"

# six runs: TM-OPT then TM-OPT+TC, each at 8x64, 16x64, 256x64
for V in tmopt tc; do
  EXTRA=""
  [ $V = tc ] && EXTRA="--tile-grad-tc"
  for U in 8x64 16x64 256x64; do
    run q_${V}_$U $PY run_train_map.py $QWEN --unit $U $TM $EXTRA --batch $CHOSEN --accum $A --out $R/q_${V}_qwen27b_$U
  done
done
log "QWEN RUNS DONE"

# evaluations (convention (a)): native and fake with FourOverSix, NVFP4 and the six maps; BF16 in its own fake process
EVAL="$QWEN --objective kl --memory-mode lean --unit 8x64 --fused-act-quant --single-pass-epilogue"
MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map NVFP4=nvfp4"
for V in tmopt tc; do for U in 8x64 16x64 256x64; do MAPS="$MAPS --evaluate-map $V-$U=$R/q_${V}_qwen27b_$U/map.pt"; done; done
run q_eval_native $PY run_multiround.py $EVAL --eval-backend native $MAPS --out $R/q_eval_native
run q_eval_fake $PY run_multiround.py $EVAL --eval-backend fake $MAPS --out $R/q_eval_fake
run q_eval_bf16 $PY run_multiround.py $EVAL --eval-backend fake --evaluate-map BF16=bf16 --out $R/q_eval_bf16
log "PART Q DONE"

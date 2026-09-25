#!/bin/bash
# Qwen3-4B native-decision safety addendum (results/multiround_models/qwen4b/det_fake/PROTOCOL.md).
# Starts after Mistral-7B's Part C queue logs MODEL DONE; 256x64 (calibration + evaluation) before 8x64.
TAG=qwen4b-detfake
R=/home/dev/n16k64_campaign/multimodel/runs/qwen4b_det_fake
N=/home/dev/n16k64_campaign/multimodel/runs/qwen4b
DATA=/home/dev/n16k64_campaign/multimodel/data
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=/home/dev/n16k64_campaign/multimodel/commands.log
until grep -q "mistral7b MODEL DONE" $LOG; do sleep 30; done
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p $R/logs
log () { echo "$(date -u +%FT%TZ) $TAG $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $R/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
COMMON="--model qwen4b --objective kl --data-root $DATA --memory-mode lean --transformers-deviation"
CAL="--dev-backend fake --skip-ce-backward --deterministic --eval-batch 1 --score-batch 1 --budget-hours 12 --record-dev-values"
for U in 256x64 8x64; do
  run det_fake_$U $PY run_multiround.py $COMMON --unit $U $CAL --out $R/det_fake_$U
  MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map DET-NATIVE-$U=$N/calib_$U/map.pt --evaluate-map DET-FAKE-$U=$R/det_fake_$U/map.pt"
  run eval_${U}_native $PY run_multiround.py $COMMON --unit $U --eval-backend native $MAPS --out $R/eval_${U}_native
  run eval_${U}_fake $PY run_multiround.py $COMMON --unit $U --eval-backend fake $MAPS --out $R/eval_${U}_fake
  log "UNIT DONE $U"
done
log "MODEL DONE rc=0"

#!/bin/bash
# After the Qwen3-4B DET-FAKE addendum: GPU candidate check, then evaluate the N16K64 one-shot k=3 maps
# (results/multiround_models/qwen4b/oneshot_k3_eval/PROTOCOL.md, deviation 1), then resume Part C with Phi-4.
LOG=/home/dev/n16k64_campaign/multimodel/commands.log
R=/home/dev/n16k64_campaign/multimodel/runs/qwen4b_oneshot
RUNS=/home/dev/n16k64_campaign/multimodel/runs
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
TAG=qwen4b-oneshot
log () { echo "$(date -u +%FT%TZ) $TAG $*" >> $LOG; }
until grep -q "qwen4b-detfake MODEL DONE" $LOG; do
  grep -q "qwen4b-detfake STOP" $LOG && { log "NOT STARTED: the DET-FAKE queue stopped"; exit 1; }
  sleep 30
done
cd $REPO
export HF_HUB_OFFLINE=0 HF_HOME=/home/dev/.cache/huggingface PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p $R/logs
run () { local name=$1; shift; log "START $name $*"; "$@" > $R/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
mv $R/maps $R/maps_cpu_checks 2>/dev/null
run convert_check_gpu $PY results/multiround_models/qwen4b/oneshot_k3_eval/convert_check.py /home/dev/n16k64_campaign/multimodel/data $R/maps
COMMON="--model qwen4b --objective kl --data-root /home/dev/n16k64_campaign/multimodel/data --memory-mode lean --transformers-deviation --unit 8x64"
MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map N16K64-n8-k3=$R/maps/N16K64-n8-k3.pt --evaluate-map N16K64-n16-k3=$R/maps/N16K64-n16-k3.pt \
 --evaluate-map DET-NATIVE-8x64=$RUNS/qwen4b/calib_8x64/map.pt --evaluate-map DET-NATIVE-256x64=$RUNS/qwen4b/calib_256x64/map.pt \
 --evaluate-map DET-FAKE-8x64=$RUNS/qwen4b_det_fake/det_fake_8x64/map.pt --evaluate-map DET-FAKE-256x64=$RUNS/qwen4b_det_fake/det_fake_256x64/map.pt"
run eval_native $PY run_multiround.py $COMMON --eval-backend native $MAPS --out $R/eval_native
run eval_fake $PY run_multiround.py $COMMON --eval-backend fake $MAPS --out $R/eval_fake
log "MODEL DONE rc=0"
exec /home/dev/n16k64_campaign/multimodel/queue_model.sh phi4 12 "" 16 8

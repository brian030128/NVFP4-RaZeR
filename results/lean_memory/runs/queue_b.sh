#!/bin/bash
# Part B (results/lean_memory/PROTOCOL.md): per configuration a legacy rerun, then a lean run, each
# checked with compare_runs.py; the queue stops at the first failed run or mismatch.
R=/home/dev/n16k64_campaign/lean_memory
DATA=/home/dev/n16k64_campaign/cost_comparison/data
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8
log () { echo "$(date -u +%FT%TZ) $*" >> $R/commands.log; }
run () { NAME=$1; shift; log "START $NAME $*"; "$@" > $R/logs/$NAME.log 2>&1; RC=$?; log "END $NAME rc=$RC"
         [ $RC -eq 0 ] || { log "STOP run failed: $NAME"; exit 1; }; }
check () { NAME=$1; shift; $PY results/lean_memory/compare_runs.py "$@" > $R/checks/$NAME.json 2> $R/logs/check_$NAME.err
           RC=$?; log "CHECK $NAME rc=$RC"; [ $RC -eq 0 ] || { log "STOP mismatch: $NAME"; exit 1; }; }
FLAGS="--objective kl --data-root $DATA --transformers-deviation --skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --deterministic --record-dev-values --dump-round0-scores"
config () {
  NAME=$1; UNIT=$2; BACKEND=$3; REF=$4; MAP=$5
  run legacy_$NAME $PY run_multiround.py --unit $UNIT $FLAGS --dev-backend $BACKEND --memory-mode legacy --out $R/legacy_$NAME
  check legacy_$NAME $REF $MAP $R/legacy_$NAME
  run lean_$NAME $PY run_multiround.py --unit $UNIT $FLAGS --dev-backend $BACKEND --memory-mode lean --out $R/lean_$NAME
  check lean_$NAME $REF $MAP $R/lean_$NAME $R/legacy_$NAME
}
config det_fake_256x64 256x64 fake $REPO/results/native_decision/runs/det_fake/report.json /home/dev/n16k64_campaign/native_decision/det_fake/map.pt
config det_native_256x64 256x64 native $REPO/results/native_decision/runs/det_native/report.json /home/dev/n16k64_campaign/native_decision/det_native/map.pt
config det_fake_8x64 8x64 fake $REPO/results/native_decision_8x64/runs/det_fake_8x64/report.json /home/dev/n16k64_campaign/native_decision_8x64/det_fake_8x64/map.pt
config det_native_8x64 8x64 native $REPO/results/native_decision_8x64/runs/det_native_8x64/report.json /home/dev/n16k64_campaign/native_decision_8x64/det_native_8x64/map.pt
log "QUEUE DONE"

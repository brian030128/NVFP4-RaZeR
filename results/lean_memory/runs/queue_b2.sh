#!/bin/bash
# Part B (results/lean_memory/PROTOCOL.md), resumed after queue_b.sh stopped on its own variable bug
# (deviation 1): legacy_det_fake_256x64 is complete and checked. Function variables are local now.
R=/home/dev/n16k64_campaign/lean_memory
DATA=/home/dev/n16k64_campaign/cost_comparison/data
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8
log () { echo "$(date -u +%FT%TZ) $*" >> $R/commands.log; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $R/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
check () { local name=$1; shift; $PY results/lean_memory/compare_runs.py "$@" > $R/checks/$name.json 2> $R/logs/check_$name.err
           local rc=$?; log "CHECK $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP mismatch: $name"; exit 1; }; }
FLAGS="--objective kl --data-root $DATA --transformers-deviation --skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --deterministic --record-dev-values --dump-round0-scores"
legacy () { local cfg=$1 unit=$2 backend=$3 ref=$4 map=$5
  run legacy_$cfg $PY run_multiround.py --unit $unit $FLAGS --dev-backend $backend --memory-mode legacy --out $R/legacy_$cfg
  check legacy_$cfg $ref $map $R/legacy_$cfg; }
lean () { local cfg=$1 unit=$2 backend=$3 ref=$4 map=$5
  run lean_$cfg $PY run_multiround.py --unit $unit $FLAGS --dev-backend $backend --memory-mode lean --out $R/lean_$cfg
  check lean_$cfg $ref $map $R/lean_$cfg $R/legacy_$cfg; }
C1="$REPO/results/native_decision/runs/det_fake/report.json /home/dev/n16k64_campaign/native_decision/det_fake/map.pt"
C2="$REPO/results/native_decision/runs/det_native/report.json /home/dev/n16k64_campaign/native_decision/det_native/map.pt"
C3="$REPO/results/native_decision_8x64/runs/det_fake_8x64/report.json /home/dev/n16k64_campaign/native_decision_8x64/det_fake_8x64/map.pt"
C4="$REPO/results/native_decision_8x64/runs/det_native_8x64/report.json /home/dev/n16k64_campaign/native_decision_8x64/det_native_8x64/map.pt"
lean det_fake_256x64 256x64 fake $C1
legacy det_native_256x64 256x64 native $C2
lean det_native_256x64 256x64 native $C2
legacy det_fake_8x64 8x64 fake $C3
lean det_fake_8x64 8x64 fake $C3
legacy det_native_8x64 8x64 native $C4
lean det_native_8x64 8x64 native $C4
log "QUEUE DONE"

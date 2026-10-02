#!/bin/bash
# Amendment 2 (results/main_ppl/PROTOCOL.md): after the nemotron zou process and the phi4/mistral7b rechecks, run
# phi4 and mistral7b's fo6-fake/if4/zou in two concurrent drivers, then qwen27b alone (fo6 recheck + 5 rows).
set -u
WT=/home/dev/n16k64_campaign/mainppl/wt
O=/home/dev/n16k64_campaign/main_ppl
PY=/home/dev/.conda/envs/n16k64/bin/python
log() { echo "$(date -u +%FT%T+00:00) $*" >> $O/commands.log; }
while kill -0 1172653 2>/dev/null || kill -0 1173412 2>/dev/null; do sleep 15; done
S=$($PY -c "import json; print(json.load(open('$O/ppl/nemotron9b/zou/report.json'))['status'])" 2>/dev/null)
log "NOTE nemotron9b zou (child of the stopped driver 1) finished; report status=$S"
cd $WT
log "NOTE amendment 2: phi4 and mistral7b rows (two drivers)"
$PY experiments/main_ppl/run.py --models phi4 --rows fo6-fake,if4,zou > $O/run_amend2_phi4.out 2>&1 &
P=$!
$PY experiments/main_ppl/run.py --models mistral7b --rows fo6-fake,if4,zou > $O/run_amend2_mistral7b.out 2>&1 &
M=$!
wait $P; RP=$?
wait $M; RM=$?
log "NOTE amendment 2: phi4 driver rc=$RP, mistral7b driver rc=$RM; qwen27b alone next"
$PY experiments/main_ppl/run.py --models qwen27b --rows fo6,fo6-fake,if4,zou,if4w,zouw > $O/run_amend2_qwen27b.out 2>&1
log "NOTE amendment 2: qwen27b driver rc=$?; ALL RUNS DONE"

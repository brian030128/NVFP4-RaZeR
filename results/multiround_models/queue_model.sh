#!/bin/bash
# Part C (results/multiround_models/PROTOCOL.md) for one model: stages 3-6.
#   queue_model.sh MODEL BUDGET_HOURS "EXTRA FLAGS" BATCHED_EVAL BATCHED_SCORE
# Stops at a failed run, a failed pre-check or an out-of-memory error; every stage is logged to commands.log.
MODEL=$1; BUDGET=$2; EXTRA=$3; BE=${4:-16}; BS=${5:-8}
R=/home/dev/n16k64_campaign/multimodel/runs/$MODEL
DATA=/home/dev/n16k64_campaign/multimodel/data
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p $R/logs
log () { echo "$(date -u +%FT%TZ) $MODEL $*" >> /home/dev/n16k64_campaign/multimodel/commands.log; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $R/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         if [ $rc -ne 0 ]; then grep -q -i "out of memory" $R/logs/$name.log && log "STOP out of memory in $name"; log "STOP run failed: $name"; exit 1; fi; }
COMMON="--model $MODEL --objective kl --data-root $DATA --memory-mode lean $EXTRA"
CHECK="--unit 8x64 --dev-backend native --deterministic --skip-ce-backward --stop-after-scoring --dump-round0-scores --record-dev-values"
run batch_batched $PY run_multiround.py $COMMON $CHECK --eval-batch $BE --score-batch $BS --out $R/batch_batched
run batch_single $PY run_multiround.py $COMMON $CHECK --eval-batch 1 --score-batch 1 --out $R/batch_single
$PY results/multiround_models/compare_batching.py $R/batch_batched $R/batch_single > $R/batching.json 2> $R/logs/batching.err
E=$($PY -c "import json; print(json.load(open('$R/batching.json'))['decision']['eval_batch'])")
S=$($PY -c "import json; print(json.load(open('$R/batching.json'))['decision']['score_batch'])")
log "CHECK batching identical=$($PY -c "import json; print(json.load(open('$R/batching.json'))['identical'])") eval_batch=$E score_batch=$S"
run pre_native $PY run_multiround.py $COMMON --unit 8x64 --eval-backend native --evaluate-map FourOverSix=fourover6 --out $R/pre_native
run pre_fake $PY run_multiround.py $COMMON --unit 8x64 --eval-backend fake --evaluate-map FourOverSix=fourover6 --out $R/pre_fake
$PY results/multiround_models/precheck.py $R/pre_native $R/pre_fake > $R/precheck.json 2> $R/logs/precheck.err
RC=$?; log "CHECK precheck rc=$RC"; [ $RC -eq 0 ] || { log "STOP pre-check failed"; exit 1; }
CAL="--dev-backend native --skip-ce-backward --deterministic --eval-batch $E --score-batch $S --budget-hours $BUDGET --record-dev-values"
run calib_8x64 $PY run_multiround.py $COMMON --unit 8x64 $CAL --out $R/calib_8x64
run calib_256x64 $PY run_multiround.py $COMMON --unit 256x64 $CAL --out $R/calib_256x64
MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map NVFP4=nvfp4 --evaluate-map MixFP4-8x64=$R/calib_8x64/map.pt --evaluate-map MixFP4-256x64=$R/calib_256x64/map.pt"
run eval_native $PY run_multiround.py $COMMON --unit 8x64 --eval-backend native $MAPS --out $R/eval_native
run eval_fake $PY run_multiround.py $COMMON --unit 8x64 --eval-backend fake $MAPS --out $R/eval_fake
run eval_bf16 $PY run_multiround.py $COMMON --unit 8x64 --eval-backend fake --evaluate-map BF16=bf16 --out $R/eval_bf16
$PY results/multiround_models/analyze_model.py $MODEL $R > $R/logs/analyze.log 2>&1
log "MODEL DONE rc=$?"

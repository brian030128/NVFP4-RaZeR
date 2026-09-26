#!/bin/bash
# TM-OPT+TC (results/tm_opt/PROTOCOL_TC.md): refactor check -> profiles -> unit test (gate) -> 9 runs -> evaluations
# -> non-deterministic timing probe. Any failure stops the queue.
R=/home/dev/n16k64_campaign/tm_opt/runs
L=/home/dev/n16k64_campaign/tm_opt/logs
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=/home/dev/n16k64_campaign/tm_opt/commands_tc.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R $L
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $L/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
check () { local name=$1; shift; "$@" >> $L/checks_tc.log 2>&1 && log "PASS $name" || { log "STOP check failed: $name"; exit 1; }; }
data () { if [ $1 = llama8b ]; then echo "--data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation";
          else echo "--data-root /home/dev/n16k64_campaign/multimodel/data"; fi; }
evalargs () { echo "--model $1 $(data $1) --objective kl --memory-mode lean --unit 8x64 --fused-act-quant --single-pass-epilogue"; }
LLAMA="--model llama8b $(data llama8b)"
TM="--tm-opt --param ste --lr 0.02 --init-logit -1 --epochs 20 --eval-every 2"

# 0. refactor check: the exact group-1 TM-OPT command (queue2.sh), TC off, must reproduce g1_ste_tmopt_nob1 bitwise
STE="--unit 8x64 --param ste --lr 0.02 --init-logit -0.2 --epochs 3 --eval-every 1 --record-theta-hashes"
NOB1="--tm-opt --no-tile-grad-kernel --dev-backend fake --eval-backend fake"
run tc_regression_g1_ste $PY run_train_map.py $LLAMA $STE $NOB1 --out $R/tc_regression_g1_ste
check tc_regression $PY results/tm_opt/compare_tm.py bitwise tc-regression $R/g1_ste_tmopt_nob1 $R/tc_regression_g1_ste

# 1. profiles: one TM-OPT epoch on Llama 8x64, then the same with TC
run profile_tmopt $PY run_train_map.py $LLAMA --unit 8x64 $TM --profile $R/profile_tmopt_llama8b_8x64.json --out $R/profile_tmopt
run profile_tmopt_tc $PY run_train_map.py $LLAMA --unit 8x64 $TM --tile-grad-tc --profile $R/profile_tmopt_tc_llama8b_8x64.json --out $R/profile_tmopt_tc

# 2. unit test: the registered tolerance gates the runs
run unit_tc $PY repro_local/realquant/test_tile_grad_tc.py $R/unit_tests_tc.json
log "PROFILE AND UNIT TEST DONE"

# 3. nine TM-OPT+TC runs, as the committed TM-OPT runs (seed 0, deterministic, STE, main's hyperparameters)
for M in llama8b mistral7b phi4; do
  for U in 8x64 16x64 256x64; do
    run tc_${M}_$U $PY run_train_map.py --model $M $(data $M) --unit $U $TM --tile-grad-tc --out $R/tc_${M}_$U
  done
done
log "TC RUNS DONE"

# 4. evaluations: native = FourOverSix + committed TM-OPT maps + TC maps; fake = FourOverSix + TC maps
for M in llama8b mistral7b phi4; do
  NM="--evaluate-map FourOverSix=fourover6"; FM="--evaluate-map FourOverSix=fourover6"
  for U in 8x64 16x64 256x64; do
    if [ $M = llama8b ] && [ $U = 8x64 ]; then P=$R/g2_tmopt/map.pt; else P=$R/tm_${M}_$U/map.pt; fi
    NM="$NM --evaluate-map tmopt-$U=$P --evaluate-map tc-$U=$R/tc_${M}_$U/map.pt"
    FM="$FM --evaluate-map tc-$U=$R/tc_${M}_$U/map.pt"
  done
  run tc_${M}_eval_native $PY run_multiround.py $(evalargs $M) --eval-backend native $NM --out $R/tc_${M}_eval_native
  run tc_${M}_eval_fake $PY run_multiround.py $(evalargs $M) --eval-backend fake $FM --out $R/tc_${M}_eval_fake
done
log "TC EVALUATIONS DONE"

# 5. non-deterministic timing probe (vs QAT C1's non-deterministic 26.9 s per epoch)
run tc_probe_nondet $PY run_train_map.py $LLAMA --unit 8x64 --tm-opt --tile-grad-tc --no-deterministic --param ste --lr 0.02 \
    --init-logit -1 --epochs 3 --eval-every 2 --out $R/tc_probe_nondet
log "TC DONE"

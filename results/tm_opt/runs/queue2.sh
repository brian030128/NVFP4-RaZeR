#!/bin/bash
# TM-OPT verification after deviation 2 (results/tm_opt/PROTOCOL.md): group 1 as registered (bitwise, B1 off), then
# group 2 = one full STE 8x64 run with the TM-OPT preset (no B1) and its joint native and fake evaluations.
R=/home/dev/n16k64_campaign/tm_opt/runs
L=/home/dev/n16k64_campaign/tm_opt/logs
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=/home/dev/n16k64_campaign/tm_opt/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
unset PYTORCH_CUDA_ALLOC_CONF
mkdir -p $R $L
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $L/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
check () { local name=$1; shift; "$@" >> $L/checks.log 2>&1 && log "PASS $name" || { log "STOP check failed: $name"; exit 1; }; }
EXP="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
DATA="--model llama8b --data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation"
log "QUEUE 2 (deviation 2: B1 out of the TM-OPT preset; group 2 = one full TM-OPT run)"

# Group 1: bitwise, B1 off; legacy vs TM-OPT without B1, both deterministic, fake evaluation (as registered)
STE="--unit 8x64 --param ste --lr 0.02 --init-logit -0.2 --epochs 3 --eval-every 1 --record-theta-hashes"
SIG="--unit 8x64 --param sigmoid --lr 0.05 --init-logit -3 --epochs 2 --eval-every 1 --record-theta-hashes"
NOB1="--tm-opt --no-tile-grad-kernel --dev-backend fake --eval-backend fake"
run g1_ste_legacy $PY run_train_map.py $DATA $STE --deterministic --out $R/g1_ste_legacy
run g1_ste_tmopt_nob1 $EXP $PY run_train_map.py $DATA $STE $NOB1 --out $R/g1_ste_tmopt_nob1
check g1_ste $PY results/tm_opt/compare_tm.py bitwise ste $R/g1_ste_legacy $R/g1_ste_tmopt_nob1
run g1_sig_legacy $PY run_train_map.py $DATA $SIG --deterministic --out $R/g1_sig_legacy
run g1_sig_tmopt_nob1 $EXP $PY run_train_map.py $DATA $SIG $NOB1 --out $R/g1_sig_tmopt_nob1
check g1_sig $PY results/tm_opt/compare_tm.py bitwise sigmoid $R/g1_sig_legacy $R/g1_sig_tmopt_nob1
log "GROUP 1 PASSED"

# Group 2 (deviation 2): one full STE 8x64 run with the TM-OPT preset (main's settings), then its map with
# FourOverSix in one native and one fake evaluator process (convention (a))
FULL="--unit 8x64 --param ste --lr 0.02 --init-logit -1 --epochs 20 --eval-every 2"
run g2_tmopt $EXP $PY run_train_map.py $DATA $FULL --tm-opt --out $R/g2_tmopt
EVAL="$DATA --objective kl --memory-mode lean --unit 8x64 --fused-act-quant --single-pass-epilogue"
MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map tmopt=$R/g2_tmopt/map.pt"
run g2_eval_native $EXP $PY run_multiround.py $EVAL --eval-backend native $MAPS --out $R/g2_eval_native
run g2_eval_fake $EXP $PY run_multiround.py $EVAL --eval-backend fake $MAPS --out $R/g2_eval_fake
check g2_e2e $PY results/tm_opt/compare_tm.py e2e $R
log "DONE: TM-OPT verification complete (stop and report)"

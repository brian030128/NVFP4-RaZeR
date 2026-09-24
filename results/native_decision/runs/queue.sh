#!/bin/bash
R=/home/dev/n16k64_campaign/native_decision
DATA=/home/dev/n16k64_campaign/cost_comparison/data
CR=/home/dev/n16k64_campaign/cost_comparison/runs
PY=/home/dev/.conda/envs/n16k64/bin/python
cd /home/dev/NVFP4-RaZeR
export HF_HUB_OFFLINE=0 PYTHONPATH=/home/dev/NVFP4-RaZeR
run () { NAME=$1; shift; echo "$(date -u +%FT%TZ) START $NAME $*" >> $R/commands.log; "$@" > $R/logs/$NAME.log 2>&1; RC=$?; echo "$(date -u +%FT%TZ) END $NAME rc=$RC" >> $R/commands.log; }
COMMON="--unit 256x64 --objective kl --data-root $DATA --transformers-deviation"
DET="--skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --deterministic"
CUBLAS_WORKSPACE_CONFIG=:4096:8 run det_fake $PY run_multiround.py $COMMON $DET --dev-backend fake --out $R/det_fake
CUBLAS_WORKSPACE_CONFIG=:4096:8 run det_native $PY run_multiround.py $COMMON $DET --dev-backend native --out $R/det_native
MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map DET-FAKE=$R/det_fake/map.pt --evaluate-map DET-NATIVE=$R/det_native/map.pt --evaluate-map B-256-opt=$CR/B_256x64_opt/map.pt --evaluate-map Bprime-256-opt=$CR/Bprime_256x64_opt/map.pt --evaluate-map SHADOW=/home/dev/n16k64_campaign/native_dev_shadow/shadow_256x64/map.pt --evaluate-map B-256-ref=$CR/B_256x64_ref/map.pt"
run eval_native $PY run_multiround.py $COMMON --eval-backend native $MAPS --out $R/eval_native
run eval_fake $PY run_multiround.py $COMMON --eval-backend fake $MAPS --evaluate-map BF16=bf16 --out $R/eval_fake
echo "$(date -u +%FT%TZ) QUEUE DONE" >> $R/commands.log

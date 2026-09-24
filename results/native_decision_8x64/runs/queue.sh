#!/bin/bash
R=/home/dev/n16k64_campaign/native_decision_8x64
DATA=/home/dev/n16k64_campaign/cost_comparison/data
PY=/home/dev/.conda/envs/n16k64/bin/python
cd /home/dev/NVFP4-RaZeR
export HF_HUB_OFFLINE=0 PYTHONPATH=/home/dev/NVFP4-RaZeR
run () { NAME=$1; shift; echo "$(date -u +%FT%TZ) START $NAME $*" >> $R/commands.log; "$@" > $R/logs/$NAME.log 2>&1; RC=$?; echo "$(date -u +%FT%TZ) END $NAME rc=$RC" >> $R/commands.log; }
COMMON="--unit 8x64 --objective kl --data-root $DATA --transformers-deviation"
DET="--skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --deterministic"
CUBLAS_WORKSPACE_CONFIG=:4096:8 run det_fake_8x64 $PY run_multiround.py $COMMON $DET --dev-backend fake --out $R/det_fake_8x64
CUBLAS_WORKSPACE_CONFIG=:4096:8 run det_native_8x64 $PY run_multiround.py $COMMON $DET --dev-backend native --out $R/det_native_8x64
MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map DET-FAKE-8x64=$R/det_fake_8x64/map.pt --evaluate-map DET-NATIVE-8x64=$R/det_native_8x64/map.pt"
run eval_native $PY run_multiround.py $COMMON --eval-backend native $MAPS --out $R/eval_native
run eval_fake $PY run_multiround.py $COMMON --eval-backend fake $MAPS --out $R/eval_fake
echo "$(date -u +%FT%TZ) QUEUE DONE" >> $R/commands.log

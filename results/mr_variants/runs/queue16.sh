#!/bin/bash
# MR-OPT variants, 16x64 addendum (results/mr_variants/ADDENDUM_16x64.md): pre-run checks, MR-OPT 16x64 on
# Llama -> Mistral -> Phi-4 and their evaluations, then the variants per model; STOP at the Qwen3.8-27B gate.
R=/home/dev/n16k64_campaign/mr_variants/runs
C=/home/dev/n16k64_campaign/mr_variants/checks16
L=/home/dev/n16k64_campaign/mr_variants/logs
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=/home/dev/n16k64_campaign/mr_variants/commands_16x64.log
STOPFILE=/home/dev/n16k64_campaign/mr_variants/STOP_VARIANTS
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > $L/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
check () { local name=$1; shift; "$@" >> $L/checks16.log 2>&1 && log "PASS $name" || { log "STOP check failed: $name"; exit 1; }; }
data () { if [ $1 = llama8b ]; then echo "--data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation";
          else echo "--data-root /home/dev/n16k64_campaign/multimodel/data"; fi; }
BASE="--objective kl --memory-mode lean --dev-backend native --deterministic --skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --record-dev-values --fused-act-quant --single-pass-epilogue --chunked-loss"
OPT="$BASE --tile-score-kernel"
declare -A VARIANT=( [sig]="--significant-steps" [ws]="--warm-start" [sigws]="--significant-steps --warm-start" )
mkdir -p $C

# Pre-run checks 1-4
run b1_unit_16x64 $PY repro_local/realquant/test_tile_score.py $C/b1_unit_tests_16x64.json --rows 16 --models llama8b mistral7b phi4
check b1_unit_16x64 $PY results/mr_variants/check_16x64.py unit $C/b1_unit_tests_16x64.json
for M in llama8b mistral7b phi4; do
  run ${M}_round0_legacy_16x64 $PY run_multiround.py --model $M $(data $M) --unit 16x64 $BASE --stop-after-scoring --dump-round0-scores --out $C/${M}_legacy_16x64
  run ${M}_round0_b1_16x64 $PY run_multiround.py --model $M $(data $M) --unit 16x64 $OPT --stop-after-scoring --dump-round0-scores --out $C/${M}_b1_16x64
  check round0_$M $PY results/mr_variants/check_16x64.py round0 $M $C/${M}_legacy_16x64 $C/${M}_b1_16x64
done
run llama8b_regression_8x64 $PY run_multiround.py --model llama8b $(data llama8b) --unit 8x64 $OPT --stop-after-scoring --dump-round0-scores --out $C/llama8b_regression_8x64
check regression_8x64 $PY results/mr_variants/check_16x64.py regression $C/llama8b_regression_8x64
log "CHECKS PASSED"

# MR-OPT 16x64 and its evaluations
for M in llama8b mistral7b phi4; do
  run ${M}_mropt_16x64 $PY run_multiround.py --model $M $(data $M) --unit 16x64 $OPT --out $R/$M/mropt_16x64
done
for M in llama8b mistral7b phi4; do
  EVAL="--model $M $(data $M) --objective kl --memory-mode lean --unit 8x64 --fused-act-quant --single-pass-epilogue"
  run ${M}_eval16_native $PY run_multiround.py $EVAL --eval-backend native --evaluate-map FourOverSix=fourover6 \
    --evaluate-map mropt-256x64=$R/$M/mropt_256x64/map.pt --evaluate-map mropt-8x64=$R/$M/mropt_8x64/map.pt \
    --evaluate-map mropt-16x64=$R/$M/mropt_16x64/map.pt --out $R/$M/eval16_native
  run ${M}_eval16_fake $PY run_multiround.py $EVAL --eval-backend fake --evaluate-map FourOverSix=fourover6 \
    --evaluate-map mropt-16x64=$R/$M/mropt_16x64/map.pt --out $R/$M/eval16_fake
done
log "MROPT16 DONE"

# The variants at 16x64, model by model, each followed by its evaluation
for M in llama8b mistral7b phi4; do
  for V in sig ws sigws; do
    if [ -e $STOPFILE ]; then log "STOP variants cut (stop file) before ${M}_${V}_16x64"; log "GATE: stopped before Qwen3.8-27B (waiting for the user's confirmation)"; exit 0; fi
    run ${M}_${V}_16x64 $PY run_multiround.py --model $M $(data $M) --unit 16x64 $OPT ${VARIANT[$V]} --out $R/$M/${V}_16x64
  done
  EVAL="--model $M $(data $M) --objective kl --memory-mode lean --unit 8x64 --fused-act-quant --single-pass-epilogue"
  MAPS="--evaluate-map FourOverSix=fourover6 --evaluate-map mropt-16x64=$R/$M/mropt_16x64/map.pt"
  for V in sig ws sigws; do MAPS="$MAPS --evaluate-map ${V}-16x64=$R/$M/${V}_16x64/map.pt"; done
  run ${M}_eval16v_native $PY run_multiround.py $EVAL --eval-backend native $MAPS --out $R/$M/eval16v_native
  log "VARIANTS16 DONE $M"
done
log "GATE: stopped before Qwen3.8-27B (waiting for the user's confirmation)"

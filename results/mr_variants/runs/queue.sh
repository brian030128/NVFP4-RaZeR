#!/bin/bash
# MR-OPT variants (results/mr_variants/PROTOCOL.md): Llama -> Mistral -> Phi-4, then STOP at the Qwen3.8-27B gate.
R=/home/dev/n16k64_campaign/mr_variants/runs
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=/home/dev/n16k64_campaign/mr_variants/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" > /home/dev/n16k64_campaign/mr_variants/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
OPT="--objective kl --memory-mode lean --dev-backend native --deterministic --skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --record-dev-values --fused-act-quant --single-pass-epilogue --tile-score-kernel --chunked-loss"
declare -A VARIANT=( [mropt]="" [sig]="--significant-steps" [ws]="--warm-start" [sigws]="--significant-steps --warm-start" )
for M in llama8b mistral7b phi4; do
  if [ $M = llama8b ]; then DATA="--data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation"; else DATA="--data-root /home/dev/n16k64_campaign/multimodel/data"; fi
  mkdir -p $R/$M
  for V in mropt sig ws sigws; do
    for U in 256x64 8x64; do
      run ${M}_${V}_${U} $PY run_multiround.py --model $M $DATA --unit $U $OPT ${VARIANT[$V]} --out $R/$M/${V}_${U}
    done
    if [ $M = llama8b ] && [ $V = mropt ]; then
      A=$(sha256sum $R/$M/mropt_256x64/map.pt | cut -d' ' -f1); B=$(sha256sum $R/$M/mropt_8x64/map.pt | cut -d' ' -f1)
      if [ $A = 6e9704f5fd1a41c5bdf8b3ea59800c3a63d1a666bf5deb75e95e769ea50b41e8 ] && [ $B = 471aa56a00b3d2c2272b49956c3e346a7d46bb3015f70717e2b5cffdc1f98121 ]; then
        log "CHECK llama sanity gate: MR-OPT maps equal the committed DET-NATIVE maps"
      else log "STOP llama sanity gate failed: $A $B"; exit 1; fi
    fi
  done
  MAPS=""
  for V in mropt sig ws sigws; do for U in 256x64 8x64; do MAPS="$MAPS --evaluate-map ${V}-${U}=$R/$M/${V}_${U}/map.pt"; done; done
  EVAL="--model $M $DATA --objective kl --memory-mode lean --unit 8x64 --fused-act-quant --single-pass-epilogue"
  run ${M}_eval_native $PY run_multiround.py $EVAL --eval-backend native --evaluate-map FourOverSix=fourover6 --evaluate-map NVFP4=nvfp4 $MAPS --out $R/$M/eval_native
  run ${M}_eval_fake $PY run_multiround.py $EVAL --eval-backend fake --evaluate-map FourOverSix=fourover6 --evaluate-map NVFP4=nvfp4 --evaluate-map mropt-256x64=$R/$M/mropt_256x64/map.pt --evaluate-map mropt-8x64=$R/$M/mropt_8x64/map.pt --out $R/$M/eval_fake
  run ${M}_eval_bf16 $PY run_multiround.py $EVAL --eval-backend fake --evaluate-map BF16=bf16 --out $R/$M/eval_bf16
  log "MODEL DONE $M"
done
log "GATE: stopped before Qwen3.8-27B (waiting for the user's confirmation)"

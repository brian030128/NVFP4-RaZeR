#!/bin/bash
# Parts 2-3 (results/deploy_eval/PROTOCOL.md, registered 2026-09-27T13:07:11Z): the 20 artifacts (FourOverSix, NVFP4
# and the three TM-OPT+TC maps per model) with the ownership check (B3), then one convention (c) evaluation per model
# (BF16; fake (c) and NativeLinear (c) for every artifact). Stops on the first failure.
D=/home/dev/n16k64_campaign/deploy_eval
A=$D/artifacts
R=$D/runs
TM=/home/dev/n16k64_campaign/tm_opt/runs
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=$D/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $A $R $D/logs
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $D/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
root () { [ $1 = llama8b ] && echo /home/dev/n16k64_campaign/cost_comparison/data || echo /home/dev/n16k64_campaign/multimodel/data; }
deviation () { [ $1 = llama8b ] && echo --transformers-deviation; }
mapdir () { [ $1 = qwen27b ] && echo $TM/q_tc_qwen27b_$2 || echo $TM/tc_$1_$2; }
MODELS="llama8b mistral7b phi4 qwen27b"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
for M in $MODELS; do
  run export_${M}_fo6 $PY export_map_artifact.py --model $M --data-root $(root $M) --kind four_over_six --ownership --out $A/${M}_fo6
  run export_${M}_nvfp4 $PY export_map_artifact.py --model $M --data-root $(root $M) --kind nvfp4 --ownership --out $A/${M}_nvfp4
  for U in 8x64 16x64 256x64; do
    run export_${M}_tc_$U $PY export_map_artifact.py --model $M --data-root $(root $M) --map $(mapdir $M $U)/map.pt --unit $U \
        --ownership --out $A/${M}_tc_$U
  done
done
log "EXPORTS DONE"
for M in $MODELS; do
  E="--evaluate BF16=bf16 --evaluate FourOverSix-fake=fake:four_over_six --evaluate NVFP4-fake=fake:nvfp4"
  for U in 8x64 16x64 256x64; do E="$E --evaluate tc-$U-fake=fake:map:$A/${M}_tc_$U.mixfp4map"; done
  E="$E --evaluate FourOverSix=native:$A/${M}_fo6 --evaluate NVFP4=native:$A/${M}_nvfp4"
  for U in 8x64 16x64 256x64; do E="$E --evaluate tc-$U=native:$A/${M}_tc_$U"; done
  run eval_$M $PY run_ppl_deploy.py --model $M --data-root $(root $M) $(deviation $M) $E --out $R/$M
done
log "PARTS 2-3 RUNS DONE"

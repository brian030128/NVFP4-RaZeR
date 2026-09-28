#!/bin/bash
# Phase 2 (PROTOCOL.md): Mistral-7B-v0.3 SCALE and QAT (learning rate, deployed run, artifact each), then the evaluation.
source /home/dev/n16k64_campaign/unified_baselines/common.sh
M=mistral7b
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
select_lr $M scale "$SCALE"
run ${M}_scale_nodev env PYTHONPATH=$QPATH $PY run_cost_distill.py --model $M --data-root $(root $M) $SCALE $UNIFIED \
    --lr $CHOSEN --no-dev --out $R/$M/scale_nodev
check_repeat $M scale
run ${M}_art_scale $PY export_map_artifact.py --model $M --data-root $(root $M) --kind four_over_six --scales $R/$M/scale_nodev/state.pt \
    --ownership --out $A/${M}_scale
select_lr $M qat "$QAT"
run ${M}_qat_nodev env PYTHONPATH=$QPATH $PY run_cost_distill.py --model $M --data-root $(root $M) $QAT $UNIFIED \
    --lr $CHOSEN --no-dev --out $R/$M/qat_nodev
check_repeat $M qat
run ${M}_art_qat $PY export_map_artifact.py --model $M --data-root $(root $M) --kind four_over_six --weights $R/$M/qat_nodev/state.pt \
    --ownership --out $A/${M}_qat
run ${M}_eval $PY run_ppl_deploy.py --model $M --data-root $(root $M) --evaluate BF16=bf16 \
    --evaluate qat-fake=fake:weights:$R/$M/qat_nodev/state.pt \
    --evaluate NVFP4=native:$DE/${M}_nvfp4 --evaluate FourOverSix=native:$DE/${M}_fo6 \
    --evaluate ours-8x64=native:$DE/${M}_tc_8x64 --evaluate ours-16x64=native:$DE/${M}_tc_16x64 \
    --evaluate scale=native:$A/${M}_scale --evaluate qat=native:$A/${M}_qat --out $R/$M/eval
run ${M}_check_eval $PY results/unified_baselines/check_eval.py $M $R/$M/eval/report.json
log "PHASE 2 DONE"

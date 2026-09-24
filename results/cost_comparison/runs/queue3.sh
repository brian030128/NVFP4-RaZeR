#!/bin/bash
# PROTOCOL.md sections 3-4 and 7: reference/live-teacher probes, grids, selected-run evaluations, remaining B runs.
L=/home/dev/n16k64_campaign/cost_comparison/runs/launch.sh
DATA=/home/dev/n16k64_campaign/cost_comparison/data
RUNS=/home/dev/n16k64_campaign/cost_comparison/runs
REPO=/home/dev/NVFP4-RaZeR
PY=/home/dev/.conda/envs/n16k64/bin/python
TB=1945.5836238861084                        # B-256-opt setup_seconds + optimization_seconds
COMMON="--data-root $DATA --transformers-deviation"
C="--arm qat --optimizer adamw_fp32 --micro-batch 8 --checkpointing"
D="--arm scale --micro-batch 8"
# Determinism diagnostic with deterministic kernels (deviation 4).
echo "$(date -u +%FT%TZ) START determinism_check_deterministic" >> $RUNS/commands.log
(cd $REPO && HF_HUB_OFFLINE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$REPO $PY results/cost_comparison/determinism_check.py $DATA $RUNS/determinism_check_deterministic.json --deterministic > $RUNS/logs/determinism_check_deterministic.log 2>&1)
echo "$(date -u +%FT%TZ) END determinism_check_deterministic rc=$?" >> $RUNS/commands.log
# Reference probes of every fallback (not trained): plain setting and the chosen C setting.
$L ref_qat_8bit_mb8 distill --arm qat --optimizer adamw_8bit --lr 1e-5 --budget probe --micro-batch 8 $COMMON --out $RUNS/probes/ref_qat_8bit_mb8
$L ref_qat_8bit_mb8_ckpt distill --arm qat --optimizer adamw_8bit --lr 1e-5 --budget probe --micro-batch 8 --checkpointing $COMMON --out $RUNS/probes/ref_qat_8bit_mb8_ckpt
$L ref_qat_offload_mb8 distill --arm qat --optimizer cpu_offload --lr 1e-5 --budget probe --micro-batch 8 $COMMON --out $RUNS/probes/ref_qat_offload_mb8
$L ref_qat_offload_mb8_ckpt distill --arm qat --optimizer cpu_offload --lr 1e-5 --budget probe --micro-batch 8 --checkpointing $COMMON --out $RUNS/probes/ref_qat_offload_mb8_ckpt
$L ref_lora_mb8 distill --arm lora --optimizer torch_adamw --lr 1e-4 --budget probe --micro-batch 8 $COMMON --out $RUNS/probes/ref_lora_mb8
$L ref_lora_mb8_ckpt distill --arm lora --optimizer torch_adamw --lr 1e-4 --budget probe --micro-batch 8 --checkpointing $COMMON --out $RUNS/probes/ref_lora_mb8_ckpt
# Live-teacher probes at the chosen settings.
$L live_qat_fp32_mb8_ckpt distill $C --lr 1e-5 --budget probe --live-teacher $COMMON --out $RUNS/probes/live_qat_fp32_mb8_ckpt
$L live_scale_mb8 distill $D --lr 1e-3 --budget probe --live-teacher $COMMON --out $RUNS/probes/live_scale_mb8

evaluate () {   # $1 = grid prefix, $2 = arm flags
  BEST=$(cd $REPO && $PY results/cost_comparison/select_and_evaluate.py $RUNS $1 2>> $RUNS/logs/selection.log)
  echo "$(date -u +%FT%TZ) SELECTED $1 -> $BEST" >> $RUNS/commands.log
  [ -n "$BEST" ] && $L ${BEST}_eval distill $2 --budget zero --evaluate $RUNS/$BEST $COMMON --out $RUNS/${BEST}_eval
}
for LR in 1e-6 1e-5 1e-4; do $L grid_C1_lr$LR distill $C --lr $LR --budget c1 --save-state $COMMON --out $RUNS/grid_C1_lr$LR; done
evaluate grid_C1 "--arm qat"
for LR in 1e-4 1e-3 1e-2; do $L grid_D1_lr$LR distill $D --lr $LR --budget c1 --save-state $COMMON --out $RUNS/grid_D1_lr$LR; done
evaluate grid_D1 "--arm scale"
for LR in 1e-6 1e-5 1e-4; do $L grid_C2_lr$LR distill $C --lr $LR --budget c2 --time-budget $TB --save-state $COMMON --out $RUNS/grid_C2_lr$LR; done
evaluate grid_C2 "--arm qat"
for LR in 1e-4 1e-3 1e-2; do $L grid_D2_lr$LR distill $D --lr $LR --budget c2 --time-budget $TB --save-state $COMMON --out $RUNS/grid_D2_lr$LR; done
evaluate grid_D2 "--arm scale"
for LR in 1e-6 1e-5 1e-4; do $L grid_C3x10_lr$LR distill $C --lr $LR --budget pool --pool $DATA/pool/pool.pt --pool-per-source 640 --save-state $COMMON --out $RUNS/grid_C3x10_lr$LR; done
evaluate grid_C3x10 "--arm qat"
for LR in 1e-6 1e-5 1e-4; do $L grid_C3x100_lr$LR distill $C --lr $LR --budget pool --pool $DATA/pool/pool.pt --pool-per-source 6400 --save-state $COMMON --out $RUNS/grid_C3x100_lr$LR; done
evaluate grid_C3x100 "--arm qat"
# Remaining B runs.
$L B_256x64_ref multiround --unit 256x64 --objective kl --budget-hours 12 $COMMON --out $RUNS/B_256x64_ref
$L B_8x64_opt multiround --unit 8x64 --objective kl --eval-batch 16 --score-batch 8 --budget-hours 12 $COMMON --out $RUNS/B_8x64_opt
echo QUEUE3 DONE >> $RUNS/commands.log

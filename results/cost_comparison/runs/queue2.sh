#!/bin/bash
# Memory-configuration probes (PROTOCOL.md section 3): C with FP32 AdamW states, then D.
L=/home/dev/n16k64_campaign/cost_comparison/runs/launch.sh
DATA=/home/dev/n16k64_campaign/cost_comparison/data
RUNS=/home/dev/n16k64_campaign/cost_comparison/runs
until grep -q "QUEUE1 DONE" $RUNS/commands.log; do sleep 20; done
echo "$(date -u +%FT%TZ) START determinism_check" >> $RUNS/commands.log
(cd /home/dev/NVFP4-RaZeR && HF_HUB_OFFLINE=0 PYTHONPATH=/home/dev/NVFP4-RaZeR /home/dev/.conda/envs/n16k64/bin/python results/cost_comparison/determinism_check.py $DATA $RUNS/determinism_check.json > $RUNS/logs/determinism_check.log 2>&1)
echo "$(date -u +%FT%TZ) END determinism_check rc=$?" >> $RUNS/commands.log
for CK in "" "--checkpointing"; do
  for MB in 8 4 2 1; do
    TAG=mb${MB}${CK:+_ckpt}
    $L probe_qat_fp32_$TAG distill --arm qat --optimizer adamw_fp32 --lr 1e-5 --budget probe --probe-steps 3 --micro-batch $MB $CK --data-root $DATA --transformers-deviation --out $RUNS/probes/qat_fp32_$TAG
  done
done
for CK in "" "--checkpointing"; do
  for MB in 8 4 2 1; do
    TAG=mb${MB}${CK:+_ckpt}
    $L probe_scale_$TAG distill --arm scale --lr 1e-3 --budget probe --probe-steps 3 --micro-batch $MB $CK --data-root $DATA --transformers-deviation --out $RUNS/probes/scale_$TAG
  done
done
echo QUEUE2 DONE >> $RUNS/commands.log

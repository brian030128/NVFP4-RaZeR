#!/bin/bash
L=/home/dev/n16k64_campaign/cost_comparison/runs/launch.sh
DATA=/home/dev/n16k64_campaign/cost_comparison/data
RUNS=/home/dev/n16k64_campaign/cost_comparison/runs
$L A_fourover6 multiround --unit 256x64 --objective kl --max-rounds 0 --eval-batch 16 --score-batch 8 --budget-hours 12 --data-root $DATA --transformers-deviation --out $RUNS/A_fourover6
$L eval_equivalence distill --arm qat --budget zero --data-root $DATA --transformers-deviation --out $RUNS/eval_equivalence
$L B_256x64_opt multiround --unit 256x64 --objective kl --eval-batch 16 --score-batch 8 --budget-hours 12 --data-root $DATA --transformers-deviation --out $RUNS/B_256x64_opt
$L Bprime_256x64_opt multiround --unit 256x64 --objective kl --eval-batch 16 --score-batch 8 --budget-hours 12 --skip-ce-backward --data-root $DATA --transformers-deviation --out $RUNS/Bprime_256x64_opt
echo QUEUE1 DONE >> $RUNS/commands.log

#!/bin/bash
# Re-run the native evaluations that used the weights-on-A kernel (nvfp4, four_over_six, n16_k3).
#
# The first native runs returned the weights-on-A output as a strided view of D^T. Downstream
# attention then received non-contiguous q/k/v, and SDPA fell back to its FP32 math backend instead
# of the bf16 flash kernel the fake-quant runs used. rq.RealLinear now returns a contiguous
# [tokens, out] tensor (same values, verified bitwise). The weights-on-B runs (n8_k3,
# four_over_six_b8x64) were already contiguous and are kept.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source repro_local/env.sh
RUNS=$CAMPAIGN_ROOT/runs
for M in llama8b qwen4b mistral7b phi4 qwen27b; do
  cat > repro_local/plans/realfix_$M.json <<EOF
[{"name": "nvfp4", "kind": "nvfp4", "kernel": "wt_as_A"},
 {"name": "four_over_six", "kind": "four_over_six", "kernel": "wt_as_A"},
 {"name": "n16_k3", "kind": "map", "from_job": "calib_$M", "map_policy": "n16_k3"}]
EOF
  if [ "$M" = llama8b ]; then REF=$RUNS/fake_maps_llama8b_attempt1; else REF=$(ls -d $RUNS/fake_${M}_attempt* | tail -1); fi
  echo "START realfix_$M $(date -u +%FT%TZ)"
  $PY repro_local/run_local.py realfix_$M -m repro_local.realquant.evaluate_ppl_real --model $M \
    --plan repro_local/plans/realfix_$M.json --domains wiki,c4 --reference-run "$REF" > $RUNS/realfix_$M.log 2>&1
  echo "END realfix_$M rc=$? $(date -u +%FT%TZ)"
done
echo "RERUN DONE"

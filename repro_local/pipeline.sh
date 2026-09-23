#!/bin/bash
# Local reproduction pipeline for one model (no Slurm; single RTX PRO 6000):
#   1. calib_<model>  campaign.calibrate seed0 -> n8_k3 / n16_k3 maps (fake quant, CE+KL, k=3)
#   2. fake_<model>   campaign.evaluate_ppl        bf16, nvfp4, four_over_six, n8_k3, n16_k3
#   3. real_<model>   realquant evaluate_ppl_real  nvfp4, four_over_six (both kernels), n16_k3, n8_k3
# Each stage is skipped when a complete attempt already exists.
#
#   repro_local/pipeline.sh qwen4b [extra calibrate args...]
set -uo pipefail
M="$1"; shift
CAL_EXTRA=("$@")
cd /home/dev/NVFP4-RaZeR-n16k64
source repro_local/env.sh
PLANS=repro_local/plans
RUNS=$CAMPAIGN_ROOT/runs

done_run() {  # done_run <name>: a complete attempt exists
  for lr in "$RUNS"/"$1"_attempt*/launch_record.json; do
    [ -f "$lr" ] && grep -q '"status": "complete"' "$lr" && return 0
  done
  return 1
}

stage() {  # stage <name> <module> args...
  local name="$1"; shift
  if done_run "$name"; then echo "SKIP $name (complete)"; return 0; fi
  echo "START $name $(date -u +%FT%TZ)"
  $PY repro_local/run_local.py "$name" -m "$@" > "$RUNS/$name.log" 2>&1
  local rc=$?
  echo "END $name rc=$rc $(date -u +%FT%TZ)"
  return $rc
}

cat > $PLANS/fake_$M.json <<EOF
[{"name": "bf16", "kind": "bf16"},
 {"name": "nvfp4", "kind": "nvfp4"},
 {"name": "four_over_six", "kind": "four_over_six"},
 {"name": "n8_k3", "kind": "map", "from_job": "calib_$M", "map_policy": "n8_k3"},
 {"name": "n16_k3", "kind": "map", "from_job": "calib_$M", "map_policy": "n16_k3"}]
EOF
cat > $PLANS/real_$M.json <<EOF
[{"name": "nvfp4", "kind": "nvfp4", "kernel": "wt_as_A"},
 {"name": "four_over_six", "kind": "four_over_six", "kernel": "wt_as_A"},
 {"name": "four_over_six_b8x64", "kind": "four_over_six", "kernel": "b8x64"},
 {"name": "n16_k3", "kind": "map", "from_job": "calib_$M", "map_policy": "n16_k3"},
 {"name": "n8_k3", "kind": "map", "from_job": "calib_$M", "map_policy": "n8_k3"}]
EOF

stage "calib_$M" campaign.calibrate --model "$M" --draw seed0 --freeze $FREEZE --freeze-sha256 $FREEZE_SHA "${CAL_EXTRA[@]}" || exit 1
stage "fake_$M" campaign.evaluate_ppl --model "$M" --plan $PLANS/fake_$M.json --domains wiki,c4 --teacher none \
  --freeze $FREEZE --freeze-sha256 $FREEZE_SHA || exit 1
FAKE_RUN=$(ls -d $RUNS/fake_${M}_attempt* | tail -1)
stage "real_$M" repro_local.realquant.evaluate_ppl_real --model "$M" --plan $PLANS/real_$M.json --domains wiki,c4 \
  --reference-run "$FAKE_RUN" || exit 1
echo "PIPELINE $M DONE"

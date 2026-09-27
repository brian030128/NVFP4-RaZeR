#!/bin/bash
# Final RTX 5090 runs after the quantizer/selection optimizations (sequential, idle GPU).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}      # the recorded run used the sm120 venv (requirements.lock.txt)
export HF_DATASETS_CACHE=${SM120_DATASETS_CACHE:-$HOME/.cache/sm120_hf_datasets}
S=sm120; R=sm120/results
$PY $S/bench/linear.py --out $R/bench/linear_rtx5090.json
for spec in "qwen4b bf16 bf16" "qwen4b native:$S/artifacts/qwen4b_nvfp4:auto_stock nvfp4_stock" "qwen4b native:$S/artifacts/qwen4b_four_over_six four_over_six" \
            "qwen4b native:$S/artifacts/qwen4b_n16_k3 n16_k3" "qwen4b native:$S/artifacts/qwen4b_n16_k3:n16k64_wA n16_k3_w128" \
            "llama8b bf16 bf16" "llama8b native:$S/artifacts/llama8b_nvfp4:auto_stock nvfp4_stock" "llama8b native:$S/artifacts/llama8b_n16_k3 n16_k3"; do
  set -- $spec
  $PY $S/bench/model.py --model $1 --policy "$2" --out $R/bench/model_${1}_$3.json
done
echo "[final] bench done"
$PY $S/eval/lmeval.py --model llama8b --suite representative \
  --policy fake_four_over_six=fake:four_over_six --policy fake_n16_k3=fake:map:$S/maps/llama8b_seed0_n16_k3.mixfp4map \
  --policy native_four_over_six=native:$S/artifacts/llama8b_four_over_six --policy native_n16_k3=native:$S/artifacts/llama8b_n16_k3 \
  --out $R/lmeval/llama8b.json
echo "[final] all done"

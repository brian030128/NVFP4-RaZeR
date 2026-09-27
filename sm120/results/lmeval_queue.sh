#!/bin/bash
# Downstream accuracy queue (RTX 5090). The shared HF datasets cache on this machine is not writable,
# so datasets are prepared in a private cache; the hub files themselves come from the shared cache.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}      # the recorded run used the sm120 venv (requirements.lock.txt)
export HF_DATASETS_CACHE=${SM120_DATASETS_CACHE:-$HOME/.cache/sm120_hf_datasets}   # override the shared, read-only cache
S=sm120; R=sm120/results
for m in qwen4b llama8b; do
  $PY $S/eval/lmeval.py --model $m --suite representative --policy bf16=bf16 \
    --policy fake_four_over_six=fake:four_over_six --policy fake_n16_k3=fake:map:$S/maps/${m}_seed0_n16_k3.mixfp4map \
    --policy native_four_over_six=native:$S/artifacts/${m}_four_over_six --policy native_n16_k3=native:$S/artifacts/${m}_n16_k3 \
    --out $R/lmeval/$m.json
done
echo "[lmeval] all done"

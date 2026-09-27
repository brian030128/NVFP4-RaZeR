#!/bin/bash
# Post-quality-chain queue as run on the RTX 5090 (sequential; the GPU is otherwise idle).
# Its lm-eval step failed on this machine's read-only shared datasets cache; see lmeval_queue.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}      # the recorded run used the sm120 venv (requirements.lock.txt)
S=sm120; R=sm120/results
until grep -q "done mistral7b" $R/ppl/run_quality.log; do sleep 30; done
echo "[queue] start $(date -u +%FT%TZ)"
for m in llama8b; do
  $PY $S/eval/layerwise.py --model $m --map $S/maps/${m}_seed0_n16_k3.mixfp4map --artifact $S/artifacts/${m}_n16_k3 --domain c4 --out $R/layerwise/${m}_n16_k3_c4w0.json
  $PY $S/eval/decode_check.py --model $m --artifact $S/artifacts/${m}_n16_k3 --map $S/maps/${m}_seed0_n16_k3.mixfp4map --out $R/decode/${m}_n16_k3.json
done
for a in qwen4b_n8_k3:qwen4b_seed0_n8_k3 llama8b_n16_k3:llama8b_seed0_n16_k3 llama8b_n8_k3:llama8b_seed0_n8_k3 mistral7b_n16_k3:mistral7b_seed0_n16_k3; do
  $PY $S/eval/ownership_check.py --artifact $S/artifacts/${a%%:*} --map $S/maps/${a##*:}.mixfp4map --out $R/ownership/${a%%:*}.json
done
echo "[queue] checks done $(date -u +%FT%TZ)"
$PY $S/bench/profile_ncu.py --out $R/bench/ncu_rtx5090.json
$PY $S/bench/linear.py --out $R/bench/linear_rtx5090.json
for spec in "qwen4b bf16 bf16" "qwen4b native:$S/artifacts/qwen4b_nvfp4:auto_stock nvfp4_stock" "qwen4b native:$S/artifacts/qwen4b_four_over_six four_over_six" \
            "qwen4b native:$S/artifacts/qwen4b_n16_k3 n16_k3" "qwen4b native:$S/artifacts/qwen4b_n16_k3:n16k64_wA n16_k3_w128" \
            "llama8b bf16 bf16" "llama8b native:$S/artifacts/llama8b_nvfp4:auto_stock nvfp4_stock" "llama8b native:$S/artifacts/llama8b_n16_k3 n16_k3"; do
  set -- $spec
  $PY $S/bench/model.py --model $1 --policy "$2" --out $R/bench/model_${1}_$3.json
done
echo "[queue] bench done $(date -u +%FT%TZ)"
for m in qwen4b llama8b; do
  $PY $S/eval/lmeval.py --model $m --suite representative --policy bf16=bf16 \
    --policy fake_four_over_six=fake:four_over_six --policy fake_n16_k3=fake:map:$S/maps/${m}_seed0_n16_k3.mixfp4map \
    --policy native_four_over_six=native:$S/artifacts/${m}_four_over_six --policy native_n16_k3=native:$S/artifacts/${m}_n16_k3 \
    --out $R/lmeval/$m.json
done
echo "[queue] all done $(date -u +%FT%TZ)"

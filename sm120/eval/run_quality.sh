#!/bin/bash
# Quality evaluations of the native path on one GPU, in sequence (each step writes its own JSON).
#   sm120/eval/run_quality.sh [models...]      default: llama8b mistral7b
# Per model: export NVFP4 / FourOverSix / N16K64-map / N8K64-map artifacts (each checked bit for bit
# against the fake-quant weights), then one perplexity run with BF16, the fake-quant policies and the
# native policies on the same windows. N8K64 runs on the weights-on-B build (n8k64_wB).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
ART=${ART:-$HERE/artifacts}
RES=${RES:-$HERE/results}
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(llama8b mistral7b)
mkdir -p "$RES/ppl" "$ART"
for m in "${MODELS[@]}"; do
  map16="$HERE/maps/${m}_seed0_n16_k3.mixfp4map"
  map8="$HERE/maps/${m}_seed0_n8_k3.mixfp4map"
  for k in nvfp4 four_over_six; do
    [ -f "$ART/${m}_$k/artifact.json" ] || $PY "$HERE/eval/export_artifact.py" --model "$m" --kind "$k" --out "$ART/${m}_$k"
  done
  [ -f "$ART/${m}_n16_k3/artifact.json" ] || $PY "$HERE/eval/export_artifact.py" --model "$m" --map "$map16" --out "$ART/${m}_n16_k3"
  [ -f "$ART/${m}_n8_k3/artifact.json" ] || $PY "$HERE/eval/export_artifact.py" --model "$m" --map "$map8" --out "$ART/${m}_n8_k3"
  $PY "$HERE/eval/ppl.py" --model "$m" \
    --policy bf16=bf16 \
    --policy fake_nvfp4=fake:nvfp4 \
    --policy fake_four_over_six=fake:four_over_six \
    --policy fake_n16_k3=fake:map:"$map16" \
    --policy fake_n8_k3=fake:map:"$map8" \
    --policy native_nvfp4=native:"$ART/${m}_nvfp4" \
    --policy native_four_over_six=native:"$ART/${m}_four_over_six" \
    --policy native_n16_k3=native:"$ART/${m}_n16_k3" \
    --policy native_n8_k3=native:"$ART/${m}_n8_k3":n8k64_wB \
    --out "$RES/ppl/$m.json" > "$RES/ppl/$m.log" 2>&1
  # CLEANUP="model ..." removes that model's non-N16 artifacts afterwards (disk space); they are
  # reproducible from the pinned model and maps with export_artifact.py.
  for c in ${CLEANUP:-}; do [ "$c" = "$m" ] && rm -rf "$ART/${m}_nvfp4" "$ART/${m}_four_over_six" "$ART/${m}_n8_k3"; done
  echo "done $m"
done

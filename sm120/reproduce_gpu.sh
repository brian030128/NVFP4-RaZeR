#!/bin/bash
# One-command reproduction of the SM120 native results on the GPU of this machine.
#
#   PY=~/envs/sm120/bin/python sm120/reproduce_gpu.sh [stage...]
#
# stages (default: all, in this order):
#   build    build + patch + self-test gate every kernel configuration
#   test     pytest suite (numerics, quantizer, kernels incl. decode probe, artifacts, Linear, selection)
#   tune     per-GPU tile-width table -> sm120/configs/<gpu>.json (commit it: it is this GPU's frozen config)
#   bench    kernel-only, decomposition, complete-Linear and full-model benchmarks
#   quality  artifacts + perplexity (bf16 / fake / native) for $MODELS, layerwise and decode checks,
#            downstream accuracy for $LMEVAL_MODELS
# Outputs go to sm120/results/<gpu>/ (the RTX 5090 run of this branch is in sm120/results/ directly).
# Needs: CUDA 13.1 at $CUDA_HOME (default /usr/local/cuda-13.1), the pinned models in the HF cache.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
MODELS=${MODELS:-"qwen4b llama8b mistral7b"}
LMEVAL_MODELS=${LMEVAL_MODELS:-"qwen4b"}
GPU=$($PY -c "import sys; sys.path.insert(0, '$HERE'); from mixfp4_sm120.select import gpu_slug; print(gpu_slug())")
OUT=${OUT:-$HERE/results/$GPU}
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(build test tune bench quality)
mkdir -p "$OUT"/{bench,ppl,lmeval,decode,layerwise}
nvidia-smi > "$OUT/nvidia-smi.txt"
for stage in "${STAGES[@]}"; do
  echo "== $stage ($GPU)"
  case $stage in
    build)   $PY "$HERE/build.py" --all --selftest | tee "$OUT/build.txt" ;;
    test)    (cd "$HERE" && $PY -m pytest -q tests) | tee "$OUT/pytest.txt" ;;
    tune)    $PY "$HERE/bench/tune_tiles.py" ;;
    bench)
      $PY "$HERE/bench/kernel.py" --out "$OUT/bench/kernel.json"
      $PY "$HERE/bench/splitk.py" --out "$OUT/bench/splitk.json"
      $PY "$HERE/bench/linear.py" --out "$OUT/bench/linear.json"
      for m in qwen4b llama8b; do
        $PY "$HERE/eval/export_artifact.py" --model $m --map "$HERE/maps/${m}_seed0_n16_k3.mixfp4map" --out "$HERE/artifacts/${m}_n16_k3" 2>/dev/null || true
        $PY "$HERE/eval/export_artifact.py" --model $m --kind nvfp4 --out "$HERE/artifacts/${m}_nvfp4" 2>/dev/null || true
        $PY "$HERE/bench/model.py" --model $m --policy bf16 --out "$OUT/bench/model_${m}_bf16.json"
        $PY "$HERE/bench/model.py" --model $m --policy "native:$HERE/artifacts/${m}_nvfp4:auto_stock" --out "$OUT/bench/model_${m}_nvfp4.json"
        $PY "$HERE/bench/model.py" --model $m --policy "native:$HERE/artifacts/${m}_n16_k3" --out "$OUT/bench/model_${m}_n16_k3.json"
      done ;;
    quality)
      ART="$HERE/artifacts" RES="$OUT" PY=$PY "$HERE/eval/run_quality.sh" $MODELS
      for m in $MODELS; do
        $PY "$HERE/eval/layerwise.py" --model $m --map "$HERE/maps/${m}_seed0_n16_k3.mixfp4map" \
          --artifact "$HERE/artifacts/${m}_n16_k3" --domain c4 --out "$OUT/layerwise/${m}_n16_k3_c4w0.json"
        $PY "$HERE/eval/decode_check.py" --model $m --artifact "$HERE/artifacts/${m}_n16_k3" \
          --map "$HERE/maps/${m}_seed0_n16_k3.mixfp4map" --out "$OUT/decode/${m}_n16_k3.json"
      done
      for m in $LMEVAL_MODELS; do
        $PY "$HERE/eval/lmeval.py" --model $m --suite representative \
          --policy fake_four_over_six=fake:four_over_six --policy fake_n16_k3=fake:map:"$HERE/maps/${m}_seed0_n16_k3.mixfp4map" \
          --policy native_four_over_six=native:"$HERE/artifacts/${m}_four_over_six" \
          --policy native_n16_k3=native:"$HERE/artifacts/${m}_n16_k3" --out "$OUT/lmeval/${m}.json"
      done ;;
    *) echo "unknown stage $stage" >&2; exit 1 ;;
  esac
done

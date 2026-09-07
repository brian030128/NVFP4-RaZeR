#!/bin/bash
# Two independent single-GPU runs within one Slurm allocation/job slot.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
PAIR_MODE="${1:?panel or teacher}"
case "$PAIR_MODE" in panel|teacher) ;; *) exit 2 ;; esac
IFS=',' read -r -a PAIR_GPUS <<< "${CUDA_VISIBLE_DEVICES:?}"
test "${#PAIR_GPUS[@]}" -eq 2
PAIR_MODELS=(qwen3-4b llama-3.1-8b-local)
run_one() {
    local index="$1"
    export CUDA_VISIBLE_DEVICES="${PAIR_GPUS[$index]}"
    export PATH="$HOME/.local/bin:$PATH"
    export HF_HOME="/tmp/$USER/hf_${SLURM_JOB_ID}_${index}"
    export UV_CACHE_DIR="$HF_HOME/uv-cache"
    export TMPDIR="$HF_HOME/tmp"
    mkdir -p "$HF_HOME" "$TMPDIR"
    trap 'rm -rf "$HF_HOME"' EXIT
    export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
    export OMP_NUM_THREADS=12
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    local python_bin=/home/u4320956/beam_engine-v3/.venv/bin/python
    uv pip install --python "$python_bin" --target "$HF_HOME/joblib" --no-deps \
      transformers==5.16.1 huggingface_hub tokenizers safetensors hf-xet loguru
    export PYTHONPATH="$PWD:$HF_HOME/joblib:${PYTHONPATH:-}"
    if test "$PAIR_MODE" = teacher; then
        "$python_bin" tests/test_domain_teacher.py
    fi
    "$python_bin" "run_domain_${PAIR_MODE}.py" --model "${PAIR_MODELS[$index]}"
}
PAIR_PIDS=()
for index in 0 1; do
    run_one "$index" >"slurm/logs/domain_${PAIR_MODE}_${SLURM_JOB_ID}_${index}.out" \
        2>"slurm/logs/domain_${PAIR_MODE}_${SLURM_JOB_ID}_${index}.err" &
    PAIR_PIDS+=("$!")
done
PAIR_FAILED=0
for pid in "${PAIR_PIDS[@]}"; do
    wait "$pid" || PAIR_FAILED=1
done
exit "$PAIR_FAILED"

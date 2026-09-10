#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=vllm_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Build the serving venv once, on a worker.
#
# Terminal-Bench needs the model behind an OpenAI-compatible endpoint, and nothing on this
# machine has vLLM. Unpacking those wheels is several GB of file writes -- well past what the
# login-node guard allows -- so it is a job. The venv lands on /home because /home is visible
# from every node, which is the whole reason the repo's other venvs live there.
#
#   sbatch slurm/vllm_setup.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs

VENV=${VENV:-/home/u4320956/NVFP4-RaZeR/.venv-vllm}
# The version matters. Unpinned, uv resolves vllm 0.29, which brings torch 2.13.0+cu130 --
# a CUDA *major* version this cluster's driver cannot run:
#   RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
# cu12x is fine on a 12.4 driver through minor-version compatibility (the repo's other venv
# runs torch 2.9.0+cu128), so pin a vllm whose wheels are built against cu128.
VLLM_SPEC=${VLLM_SPEC:-vllm==0.11.0}
# transformers also has to be held back. Unconstrained, uv installs 5.17, whose rewritten
# tokenizer layer vllm 0.11 does not know:
#   AttributeError: TokenizersBackend has no attribute all_special_tokens_extended
# 4.57.3 is the version the repo's other venv already runs, so the served model and the
# in-process evaluations agree on tokenization as well.
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-transformers==4.57.3}
UV=/home/u4320956/.local/bin/uv
export UV_CACHE_DIR="/tmp/$USER/uvcache_$SLURM_JOB_ID"
mkdir -p "$UV_CACHE_DIR"
trap 'rm -rf "$UV_CACHE_DIR"' EXIT

echo "=== vllm setup -> $VENV  spec=$VLLM_SPEC $TRANSFORMERS_SPEC  $(date -Iseconds) on $(hostname) ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader
"$UV" venv --python 3.12 "$VENV"
"$UV" pip install --python "$VENV/bin/python" "$VLLM_SPEC" "$TRANSFORMERS_SPEC"
"$VENV/bin/python" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
# Fail loudly here rather than at serve time if the wheel is for the wrong CUDA major version.
"$VENV/bin/python" -c "
import torch, sys
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
sys.exit(0 if torch.cuda.is_available() else 1)
" || { echo 'FATAL: torch cannot see the GPU -- wrong CUDA build for this driver'; exit 1; }
# --- serve something, briefly -----------------------------------------------------------------
# Two failures so far surfaced only when vLLM actually started an engine, each costing a full
# round trip through the Terminal-Bench job: a cu130 wheel the driver could not load, and a
# transformers 5 tokenizer API vllm 0.11 does not know. Neither is visible to `import vllm`.
# So the setup finishes by serving a small model and calling the endpoint once.
export HF_HOME="/tmp/$USER/hfsmoke_$SLURM_JOB_ID"; mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
SMOKE_MODEL=${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
SMOKE_PORT=${SMOKE_PORT:-18199}
echo "--- serve smoke: $SMOKE_MODEL on :$SMOKE_PORT"
"$VENV/bin/vllm" serve "$SMOKE_MODEL" --served-model-name smoke \
    --host 127.0.0.1 --port "$SMOKE_PORT" --max-model-len 2048 \
    --gpu-memory-utilization 0.30 > "/tmp/$USER/vllm_smoke_$SLURM_JOB_ID.log" 2>&1 &
SMOKE_PID=$!
OK=0
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$SMOKE_PORT/health" > /dev/null; then OK=1; break; fi
    kill -0 "$SMOKE_PID" 2>/dev/null || break
    sleep 10
done
if [[ "$OK" == "1" ]]; then
    curl -sf "http://127.0.0.1:$SMOKE_PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"smoke","prompt":"2+2=","max_tokens":8}' | head -c 400; echo
    echo "SERVE SMOKE OK"
else
    echo "FATAL: vLLM could not serve a model -- see /tmp/$USER/vllm_smoke_$SLURM_JOB_ID.log"
    tail -25 "/tmp/$USER/vllm_smoke_$SLURM_JOB_ID.log"
fi
kill "$SMOKE_PID" 2>/dev/null; wait "$SMOKE_PID" 2>/dev/null
rm -rf "$HF_HOME"
[[ "$OK" == "1" ]] || exit 1

echo "=== done $(date -Iseconds) ==="

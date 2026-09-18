#!/bin/bash
# Source only from a Slurm worker. Keep dependencies and downloads job-local.
: "${SLURM_JOB_ID:?Run through Slurm}"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="/tmp/$USER/reorder_$SLURM_JOB_ID"
export UV_CACHE_DIR="$HF_HOME/uv-cache" TMPDIR="$HF_HOME/tmp"
export PYTHONPYCACHEPREFIX="$HF_HOME/pycache"
mkdir -p "$TMPDIR"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python
if [[ "$MODEL" == qwen27b ]]; then
    uv pip install --python "$PYTHON" --target "$HF_HOME/joblib" --no-deps \
        'transformers==5.16.1' 'huggingface_hub==1.5.0' 'tokenizers==0.23.2' \
        safetensors hf-xet loguru
else
    uv pip install --python "$PYTHON" --target "$HF_HOME/joblib" --no-deps loguru
fi
export PYTHONPATH="$PWD:$HF_HOME/joblib:${PYTHONPATH:-}"

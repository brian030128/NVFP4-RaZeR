#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide_h200
#SBATCH --job-name=tb27_tput
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# What would a Terminal-Bench trial on Qwen3.8-27B actually cost?
#
# There is no fast serving path for that model here. The driver is CUDA 12.4, and every vLLM
# that knows the Qwen3.5 architecture needs CUDA 13:
#
#     vllm 0.11.0   no Qwen3.5 support     torch 2.8.0+cu128    loads here
#     vllm 0.26.0   Qwen3_5 registered     torch 2.11.0 -> nvidia-*-cu13
#     vllm 0.29.0   Qwen3_5 registered     torch 2.13.0+cu130
#
# So the only option is the in-process W4A4 server, measured at about three hours per trial for
# a 4B model. Rather than extrapolate that to 24B, this measures it: serve the real policy and
# time generation at the shape an agent turn actually has -- a few thousand tokens of context,
# a few hundred generated -- and convert that into hours per trial.
#
#   sbatch slurm/tbench_27b_throughput.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs

MODEL=${1:-qwen27b}
POLICY=${POLICY:-four_over_six}
PORT=${PORT:-18137}

PROJECT="$PWD"
SCRATCH="/tmp/$USER/tput_$SLURM_JOB_ID"
mkdir -p "$SCRATCH"/{tmp,cache}
trap 'rm -rf "$SCRATCH"' EXIT
export TMPDIR="$SCRATCH/tmp" UV_CACHE_DIR="$SCRATCH/cache/uv"
export HF_HOME="$SCRATCH/hf"; mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
UV=/home/u4320956/.local/bin/uv
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps" fastapi uvicorn loguru || exit 1
rm -rf "$SCRATCH"/deps/{transformers,tokenizers,torch,nvidia,triton}
PYPATH="$PROJECT:$SCRATCH/deps"
if [[ "$MODEL" == "qwen27b" ]]; then
    "$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps27" --no-deps \
      transformers==5.16.1 'huggingface_hub==1.5.0' tokenizers safetensors hf-xet loguru || exit 1
    PYPATH="$PROJECT:$SCRATCH/deps27:$SCRATCH/deps"
fi

OUT="$PROJECT/results/terminal_bench/tput_$SLURM_JOB_ID"
mkdir -p "$OUT"

CALIB="$SCRATCH/calibration_$MODEL"
echo "=== throughput probe $MODEL/$POLICY  $(date -Iseconds) on $(hostname) ==="
env PYTHONPATH="$PYPATH" "$PYTHON" run_math_code_calibration.py \
    --model "$MODEL" --out "$CALIB" --allow-source-drift \
    > "$OUT/calibration.log" 2>&1 || { tail -20 "$OUT/calibration.log"; exit 1; }

env PYTHONPATH="$PYPATH" "$PYTHON" serve_quantized.py --model "$MODEL" --calib "$CALIB" \
    --policy "$POLICY" --port "$PORT" --allow-map-drift \
    --provenance "$OUT/provenance.json" > "$OUT/serve.log" 2>&1 &
SRV=$!
for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
    kill -0 "$SRV" 2>/dev/null || { echo "server died"; tail -20 "$OUT/serve.log"; exit 1; }
    sleep 10
done
echo "server ready after ${i}0s"

env PYTHONPATH="$PYPATH" "$PYTHON" "$PROJECT/scripts/measure_agent_throughput.py" \
    --port "$PORT" --out "$OUT" 2>&1 | tee "$OUT/throughput.txt"

kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
echo "=== done $(date -Iseconds) ==="

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
# Serving is in-process (serve_quantized.py), which is the faithful route: it keeps the W4A4
# activation hooks that no checkpoint export carries. The question here is whether it is fast
# enough to be worth using, rather than extrapolating the 4B's three-hours-per-trial by
# parameter count. It times generation at the shape an agent turn actually has -- a few
# thousand tokens of prompt, a few hundred generated -- and converts that into hours/trial.
#
# Answer, measured: 1.85-2.0 tok/s, so a 400-token turn is 3.6 min and a 10-turn trial 0.6 h.
# Fast enough, and no inference server is needed. If one ever is, use sglang.
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

echo "=== throughput probe $MODEL/$POLICY  $(date -Iseconds) on $(hostname) ==="

# A throughput probe of four_over_six needs no election, so it needs no scoring pass: the
# shipped calibration directory supplies the report.json that serve_quantized reads, and the
# scores it lacks are only used for k3. That skips ~25 minutes, and it avoids the 27B
# calibration's device_map='balanced' over two GPUs, which failed here with
# "CUDA error: invalid device ordinal" on a single-GPU allocation.
SHIPPED_CALIB=${SHIPPED_CALIB:-results/math_code_adaptive/calibration_333787_qwen27b}
if [[ "$POLICY" == "k3" ]]; then
    CALIB="$SCRATCH/calibration_$MODEL"
    env PYTHONPATH="$PYPATH" "$PYTHON" run_math_code_calibration.py \
        --model "$MODEL" --out "$CALIB" --allow-source-drift \
        > "$OUT/calibration.log" 2>&1 || { tail -20 "$OUT/calibration.log"; exit 1; }
else
    CALIB="$PROJECT/$SHIPPED_CALIB"
    echo "using the shipped calibration (no election needed for $POLICY): $CALIB"
fi

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

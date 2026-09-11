#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide_h200
#SBATCH --job-name=tbench_kse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# MIXFP4_REPORT.md's policies on Terminal-Bench.
#
# Perplexity and multiple choice both score the model on a fixed corpus. Terminal-Bench scores
# what it DOES: an agent works in a container and the trial passes only if the task's own tests
# pass. A quantization error has to survive a long chain of the model's own decisions, which is
# the failure mode neither other measurement can see.
#
# The model is served in-process by serve_quantized.py rather than exported to a checkpoint,
# because these policies cannot survive save_pretrained: MixFP4 (k=3) is a per-tile map, and
# every policy here is W4A4 whose activation half lives in forward hooks. A checkpoint would
# quietly become W4A16 -- a different method from the one being reported.
#
# Expect low absolute scores. Terminal-Bench 4.0 is built for frontier agents, and a capped
# 4B-class model is near the floor; the point is the comparison BETWEEN policies at an
# identical budget, and a floor that swallows all three is itself the finding.
#
#   sbatch slurm/tbench_kse.sbatch qwen4b
#   NTASKS=30 POLICIES="four_over_six,k3" sbatch slurm/tbench_kse.sbatch qwen4b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs

MODEL=${1:-qwen4b}
POLICIES=${POLICIES:-"bf16,four_over_six,k3"}
DATASET=${DATASET:-terminal-bench/terminal-bench@4.0.0}
NTASKS=${NTASKS:-15}
AGENT=${AGENT:-terminus-2}
MAX_TURNS=${MAX_TURNS:-20}
AGENT_TMULT=${AGENT_TMULT:-0.25}
PORT=${PORT:-18127}

PROJECT="$PWD"
SCRATCH="/tmp/$USER/tbkse_$SLURM_JOB_ID"
mkdir -p "$SCRATCH"/{sif,cache,tmp,jobs}
trap 'rm -rf "$SCRATCH"' EXIT

export UV_CACHE_DIR="$SCRATCH/cache/uv" TMPDIR="$SCRATCH/tmp"
export SINGULARITY_CACHEDIR="$SCRATCH/cache/singularity" APPTAINER_CACHEDIR="$SCRATCH/cache/singularity"
export SINGULARITY_TMPDIR="$SCRATCH/tmp" APPTAINER_TMPDIR="$SCRATCH/tmp"
export HARBOR_HOME="$SCRATCH/harbor"
export HF_HOME="$SCRATCH/hf"; mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"

# See scripts/singularity-no-fakeroot: harbor's --fakeroot cannot work for this account.
SHIM="$SCRATCH/bin"; mkdir -p "$SHIM"
ln -sf "$PROJECT/scripts/singularity-no-fakeroot" "$SHIM/singularity"
export PATH="$SHIM:$PATH"

PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
UV=/home/u4320956/.local/bin/uv
"$UV" pip install --python "$PYTHON" --target "$SCRATCH/deps" \
  fastapi uvicorn loguru || exit 1
rm -rf "$SCRATCH"/deps/{transformers,tokenizers,torch,nvidia,triton}
export PYTHONPATH="$PROJECT:$SCRATCH/deps"

OUT="$PROJECT/results/terminal_bench/kse_job_$SLURM_JOB_ID"
mkdir -p "$OUT"

echo "=== tbench_kse $MODEL  policies=$POLICIES  tasks=$NTASKS  $(date -Iseconds) on $(hostname) ==="
singularity --version; harbor --version

# --- the calibration the k=3 election needs (its shipped score shards are gone) --------------
CALIB="$SCRATCH/calibration_$MODEL"
echo "--- calibration -> $CALIB"
"$PYTHON" run_math_code_calibration.py --model "$MODEL" --out "$CALIB" --allow-source-drift \
    > "$OUT/calibration_$MODEL.log" 2>&1 || { tail -20 "$OUT/calibration_$MODEL.log"; exit 1; }
"$PYTHON" "$PROJECT/scripts/check_map_digest.py" "$CALIB" "$MODEL" || \
    echo 'NOTE: maps.json differs; serve_quantized.py checks the frozen map itself.'

IFS=',' read -ra LIST <<< "$POLICIES"
for POLICY in "${LIST[@]}"; do
    echo "########## policy $POLICY"
    "$PYTHON" serve_quantized.py --model "$MODEL" --calib "$CALIB" --policy "$POLICY" \
        --port "$PORT" --served-name qmodel \
        --provenance "$OUT/provenance_$POLICY.json" ${SERVE_ARGS:-} \
        > "$OUT/serve_$POLICY.log" 2>&1 &
    SRV=$!
    READY=0
    for i in $(seq 1 90); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && { READY=1; break; }
        kill -0 "$SRV" 2>/dev/null || { echo "server died"; break; }
        sleep 10
    done
    if [[ "$READY" != "1" ]]; then
        echo "FATAL: server never became ready for $POLICY"; tail -25 "$OUT/serve_$POLICY.log"
        kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; continue
    fi
    echo "server ready for $POLICY after ${i}0s"

    # harbor resolves the provider key and base URL from the host environment before forwarding.
    export OPENAI_API_KEY=local
    export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
    export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1"

    timeout "${HARBOR_TIMEOUT:-43200}" harbor run \
        -d "$DATASET" -e singularity \
        --ek "singularity_image_cache_dir=$SCRATCH/sif" \
        -a "$AGENT" -m "openai/qmodel" \
        --ak "max_turns=$MAX_TURNS" \
        --agent-timeout-multiplier "$AGENT_TMULT" \
        --ae "OPENAI_API_KEY=local" \
        --ae "OPENAI_API_BASE=http://127.0.0.1:$PORT/v1" \
        --ae "OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1" \
        --allow-agent-host 127.0.0.1 \
        -l "$NTASKS" -k 1 -n 1 -q \
        -o "$SCRATCH/jobs" --job-name "$POLICY" -y \
        > "$OUT/harbor_$POLICY.log" 2>&1
    echo "harbor exit=$? for $POLICY"
    tail -20 "$OUT/harbor_$POLICY.log"

    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    cp -r "$SCRATCH/jobs/$POLICY" "$OUT/jobs_$POLICY" 2>/dev/null
done

echo "=== done $(date -Iseconds) ==="

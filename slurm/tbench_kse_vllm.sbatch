#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide_h200
#SBATCH --job-name=tbench_vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Terminal-Bench on MIXFP4_REPORT.md's policies, served by vLLM.
#
# READ THIS BEFORE QUOTING ANY NUMBER FROM THIS JOB: these rows are **W4A16**, not the W4A4 the
# report measures. The element-type election is a weight rewrite and bakes into the exported
# checkpoint; the activation quantizer lives in forward hooks that save_pretrained drops. So
# this measures the weight half of each policy -- which is the half the k-SE election decides --
# and not the reported configuration.
#
# The faithful alternative was measured and rejected on cost: serving the real W4A4 path
# in-process (slurm/tbench_kse.sbatch) ran at about three hours per Terminal-Bench trial for a
# 4B model, so three policies over fifteen tasks is roughly 135 GPU-hours. vLLM is an order of
# magnitude faster and cannot run the hooks.
#
# Qwen3-4B is the target because vllm 0.11.0 predates the Qwen3.5 architecture and cannot serve
# Qwen3.8-27B at all, and because Qwen3-4B is where the method's accuracy effect is largest
# (+0.0135, p = 4e-10), so it is the model most likely to show a difference here.
#
#   sbatch slurm/tbench_kse_vllm.sbatch qwen4b
#   NTASKS=25 POLICIES="four_over_six,k3" sbatch slurm/tbench_kse_vllm.sbatch qwen4b

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
PORT=${PORT:-18131}

# Three of the 66 tasks declare a GPU inside the container (scripts/list_gpu_tasks.py reads it
# out of each task.toml). harbor's singularity environment has no GPU capability, so they raise
#     RuntimeError: Task requires 1 GPU(s) but EnvironmentType.SINGULARITY ...
# before the agent runs. They fail identically for every policy and say nothing about
# quantization, so they are excluded by name rather than discovered one failure at a time.
GPU_TASKS=${GPU_TASKS:-"fp8-rmsnorm-gemm jax-speedrun-gpu math-eval-grader"}
EXCLUDE_ARGS=()
for t in $GPU_TASKS; do EXCLUDE_ARGS+=(-x "$t"); done

# -n 1 by default. At -n 2 the 15-task run lost 7 of 13 trials to container startup rather
# than to the model: "Failed to start Singularity FastAPI server after 3 port attempts" and
# "Server process died on port". Each container extracts a multi-GB SIF (the unshare -r path
# extracts rather than loop-mounts), so two at once contend for ports and scratch.
PROJECT="$PWD"
SCRATCH="/tmp/$USER/tbv_$SLURM_JOB_ID"
mkdir -p "$SCRATCH"/{sif,cache,tmp,jobs,ckpt}
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

SHIM="$SCRATCH/bin"; mkdir -p "$SHIM"
ln -sf "$PROJECT/scripts/singularity-no-fakeroot" "$SHIM/singularity"
export PATH="$SHIM:$PATH"

PYTHON=${PYTHON:-/home/u4320956/beam_engine-v3/.venv/bin/python}
VLLM_VENV=${VLLM_VENV:-/home/u4320956/NVFP4-RaZeR/.venv-vllm}
# Never exported: these are built for Python 3.11 and harbor runs on its own 3.13, which dies on
# a foreign pydantic_core. Applied per command instead.
PYPATH="$PROJECT"

OUT="$PROJECT/results/terminal_bench/vllm_job_$SLURM_JOB_ID"
mkdir -p "$OUT"

echo "=== tbench_vllm $MODEL  policies=$POLICIES  tasks=$NTASKS  $(date -Iseconds) on $(hostname) ==="
echo "PRECISION NOTE: exported checkpoints are W4A16; the report's policies are W4A4."
singularity --version; harbor --version
[[ -x "$VLLM_VENV/bin/vllm" ]] || { echo "FATAL: no vLLM venv at $VLLM_VENV"; exit 1; }

CALIB="$SCRATCH/calibration_$MODEL"
echo "--- calibration -> $CALIB"
env PYTHONPATH="$PYPATH" "$PYTHON" run_math_code_calibration.py \
    --model "$MODEL" --out "$CALIB" --allow-source-drift \
    > "$OUT/calibration_$MODEL.log" 2>&1 || { tail -20 "$OUT/calibration_$MODEL.log"; exit 1; }
env PYTHONPATH="$PYPATH" "$PYTHON" "$PROJECT/scripts/check_map_digest.py" "$CALIB" "$MODEL" || \
    echo 'NOTE: maps.json differs; the exporter checks the frozen map itself.'

IFS=',' read -ra LIST <<< "$POLICIES"
for POLICY in "${LIST[@]}"; do
    echo "########## policy $POLICY"
    CKPT="$SCRATCH/ckpt/$POLICY"
    env PYTHONPATH="$PYPATH" "$PYTHON" export_kse_checkpoint.py \
        --model "$MODEL" --calib "$CALIB" --policy "$POLICY" --out "$CKPT" \
        ${EXPORT_ARGS:-} > "$OUT/export_$POLICY.log" 2>&1
    if [[ $? -ne 0 ]]; then echo "export failed for $POLICY"; tail -20 "$OUT/export_$POLICY.log"; continue; fi
    cp "$CKPT/quantization_provenance.json" "$OUT/provenance_$POLICY.json" 2>/dev/null

    "$VLLM_VENV/bin/vllm" serve "$CKPT" --served-model-name qmodel \
        --host 127.0.0.1 --port "$PORT" --max-model-len "${MAXLEN:-32768}" \
        --gpu-memory-utilization "${GPUFRAC:-0.85}" \
        > "$OUT/vllm_$POLICY.log" 2>&1 &
    SRV=$!
    READY=0
    for i in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && { READY=1; break; }
        kill -0 "$SRV" 2>/dev/null || { echo "vllm died"; break; }
        sleep 10
    done
    if [[ "$READY" != "1" ]]; then
        echo "FATAL: vllm never became ready for $POLICY"; tail -25 "$OUT/vllm_$POLICY.log"
        kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; rm -rf "$CKPT"; continue
    fi
    echo "vllm ready for $POLICY after ${i}0s"

    export OPENAI_API_KEY=local
    export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
    export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1"

    timeout "${HARBOR_TIMEOUT:-36000}" harbor run \
        -d "$DATASET" -e singularity \
        --ek "singularity_image_cache_dir=$SCRATCH/sif" \
        -a "$AGENT" -m "openai/qmodel" \
        --ak "max_turns=$MAX_TURNS" \
        --agent-timeout-multiplier "$AGENT_TMULT" \
        --ae "OPENAI_API_KEY=local" \
        --ae "OPENAI_API_BASE=http://127.0.0.1:$PORT/v1" \
        --ae "OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1" \
        --allow-agent-host 127.0.0.1 \
        "${EXCLUDE_ARGS[@]}" \
        -l "$NTASKS" -k 1 -n "${NCONCURRENT:-1}" -q \
        -o "$SCRATCH/jobs" --job-name "$POLICY" -y \
        > "$OUT/harbor_$POLICY.log" 2>&1
    echo "harbor exit=$? for $POLICY"
    tail -20 "$OUT/harbor_$POLICY.log"

    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
    rm -rf "$CKPT"
    cp -r "$SCRATCH/jobs/$POLICY" "$OUT/jobs_$POLICY" 2>/dev/null
done

echo "=== done $(date -Iseconds) ==="

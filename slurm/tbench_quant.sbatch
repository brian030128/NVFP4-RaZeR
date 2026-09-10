#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide_h200
#SBATCH --job-name=tbench_quant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Terminal-Bench on locally served, fake-quantized models.
#
# Every other evaluation in this repo scores the model on a fixed corpus. Terminal-Bench scores
# what the model DOES: it drives an agent in a container, and the trial passes only if the task's
# own tests pass afterwards. That makes it the one measurement here where a quantization error
# has to survive a long chain of the model's own decisions rather than one next-token
# distribution -- which is exactly the failure mode perplexity and multiple choice both miss.
#
# The chain: export_quant_checkpoint.py bakes the quantized weights into an ordinary BF16
# checkpoint, vLLM serves it on localhost, and harbor's agent reaches that endpoint because the
# singularity backend shares the host network namespace.
#
# IMPORTANT -- this is W4A16, not the W4A4 of MIXFP4_REPORT §1. The activation quantizer lives in
# models/qmodule_*.py and does not survive save_pretrained, so only the weights are quantized
# here. Do not put a number from this job in the same table as a §1 number.
#
#   sbatch slurm/tbench_quant.sbatch llama-3.1-8b-ins-local
#   NTASKS=20 CONFIGS="fp16,nvfp4" sbatch slurm/tbench_quant.sbatch qwen3-8b

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs

MODEL=${1:-llama-3.1-8b-ins-local}
# bf16 reference, plain NVFP4 (§1's reference), and §1's recommended election rule.
CONFIGS=${CONFIGS:-"fp16,nvfp4,mix_4_6_clipa1_hess_impg16_h10@8x64"}
DATASET=${DATASET:-terminal-bench/terminal-bench@4.0.0}
NTASKS=${NTASKS:-10}
AGENT=${AGENT:-terminus-2}
KATTEMPTS=${KATTEMPTS:-1}
PORT=${PORT:-18123}

PROJECT="$PWD"
SCRATCH="/tmp/$USER/tbq_$SLURM_JOB_ID"
mkdir -p "$SCRATCH"/{sif,cache,tmp,jobs,ckpt}
trap 'rm -rf "$SCRATCH"' EXIT

export UV_CACHE_DIR="$SCRATCH/cache/uv"
export TMPDIR="$SCRATCH/tmp"
export SINGULARITY_CACHEDIR="$SCRATCH/cache/singularity"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$SCRATCH/tmp" APPTAINER_TMPDIR="$SCRATCH/tmp"
export HARBOR_HOME="$SCRATCH/harbor"
export HF_HOME="$SCRATCH/hf"; mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"

# See scripts/singularity-no-fakeroot: harbor's --fakeroot is unusable for this account.
SHIM="$SCRATCH/bin"; mkdir -p "$SHIM"
ln -sf "$PROJECT/scripts/singularity-no-fakeroot" "$SHIM/singularity"
export PATH="$SHIM:$PATH"

QUANT_PY=${QUANT_PY:-/home/u4320956/beam_engine-v3/.venv/bin/python}
VLLM_VENV=${VLLM_VENV:-/home/u4320956/NVFP4-RaZeR/.venv-vllm}
OUT="$PROJECT/results/terminal_bench/quant_job_$SLURM_JOB_ID"
mkdir -p "$OUT"

echo "=== tbench_quant $MODEL  $(date -Iseconds) on $(hostname) ==="
echo "configs: $CONFIGS   dataset: $DATASET   tasks: $NTASKS   agent: $AGENT"
singularity --version; harbor --version
[[ -x "$VLLM_VENV/bin/vllm" ]] || { echo "FATAL: no vLLM venv at $VLLM_VENV -- run slurm/vllm_setup.sbatch"; exit 1; }

IFS=',' read -ra CFG_LIST <<< "$CONFIGS"
for CFG in "${CFG_LIST[@]}"; do
    TAG=$(echo "$CFG" | tr '/@' '__')
    CKPT="$SCRATCH/ckpt/$TAG"
    echo "########## $CFG -> $TAG"

    # 1. bake the quantized weights into a servable checkpoint
    "$QUANT_PY" export_quant_checkpoint.py --model_name "$MODEL" --config "$CFG" \
        --out "$CKPT" > "$OUT/export_$TAG.log" 2>&1
    if [[ $? -ne 0 ]]; then echo "export FAILED for $CFG, see $OUT/export_$TAG.log"; tail -20 "$OUT/export_$TAG.log"; continue; fi
    cp "$CKPT/quantization_provenance.json" "$OUT/provenance_$TAG.json" 2>/dev/null

    # 2. serve it
    "$VLLM_VENV/bin/vllm" serve "$CKPT" \
        --served-model-name qmodel \
        --host 127.0.0.1 --port "$PORT" \
        --max-model-len "${MAXLEN:-32768}" \
        --gpu-memory-utilization "${GPUFRAC:-0.85}" \
        > "$OUT/vllm_$TAG.log" 2>&1 &
    VLLM_PID=$!
    echo "vllm pid $VLLM_PID, waiting for /health ..."
    READY=0
    for i in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then READY=1; break; fi
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "vllm died early"; break; }
        sleep 10
    done
    if [[ "$READY" != "1" ]]; then
        echo "FATAL: vllm never became ready for $CFG"; tail -30 "$OUT/vllm_$TAG.log"
        kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; continue
    fi
    echo "vllm ready after ${i}0s"

    # 3. run the benchmark against it. The agent runs inside the container but singularity
    #    shares the host network namespace, so 127.0.0.1 is the same loopback vLLM bound to.
    timeout "${HARBOR_TIMEOUT:-36000}" harbor run \
        -d "$DATASET" \
        -e singularity \
        --ek "singularity_image_cache_dir=$SCRATCH/sif" \
        -a "$AGENT" \
        -m "openai/qmodel" \
        --ae "OPENAI_API_KEY=local" \
        --ae "OPENAI_API_BASE=http://127.0.0.1:$PORT/v1" \
        --ae "OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1" \
        --allow-agent-host 127.0.0.1 \
        -l "$NTASKS" -k "$KATTEMPTS" -n 1 -q \
        -o "$SCRATCH/jobs" --job-name "$TAG" -y \
        > "$OUT/harbor_$TAG.log" 2>&1
    echo "harbor exit=$? for $CFG"
    tail -25 "$OUT/harbor_$TAG.log"

    kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null
    rm -rf "$CKPT"          # tens of GB; the provenance file is what we keep
    cp -r "$SCRATCH/jobs/$TAG" "$OUT/jobs_$TAG" 2>/dev/null
done

echo "=== done $(date -Iseconds) ==="

#!/bin/bash
# The flipquant parity check (results/topk_cal/flipquant_parity/NOTE.md): calibration time and memory of flipquant's
# calibration.train_map_razer (main b1c4123) against NVFP4-RaZeR's run_train_map.py (topk-cal), one run at a time on an
# idle GPU, every run under measure.py. Llama-3.1-8B: flipquant A' (full, 20 epochs) and C' (top-256, 5 epochs), compared
# with topk-cal's runs A and C of the same day. Phi-4: a back-to-back pair, flipquant then RaZeR (the paper's command).
set -u
FQ=/home/dev/n16k64_campaign/fqport/wt        # flipquant, a worktree at b1c4123 (= main)
RZ=$(cd "$(dirname "$0")/../../.." && pwd)     # NVFP4-RaZeR, topk-cal
O=/home/dev/n16k64_campaign/fqparity
PY=/home/dev/.conda/envs/n16k64/bin/python
MEASURE="$PY $RZ/results/topk_cal/flipquant_parity/measure.py"
FQDATA=/home/dev/n16k64_campaign/fqport/razer_data
export HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=0 PYTHONDONTWRITEBYTECODE=1
unset SM120_BUILD_DIR PYTORCH_CUDA_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG
mkdir -p $O
log() { echo "$(date -u +%FT%TZ) $*" >> $O/commands.log; }
idle() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy before $1"; exit 1; }; return 0; }
run() { local name=$1 cwd=$2; shift 2; idle $name; log "START $name (cwd $cwd): $*"
        $MEASURE --out $O/$name.measure.json --log $O/$name.log --cwd $cwd -- "$@"; local rc=$?; log "END $name rc=$rc"
        [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
[ "$(git -C $FQ rev-parse HEAD)" = b1c41236ed72771bdd5dea12a809c0f2c6aee55a ] && [ -z "$(git -C $FQ status --porcelain)" ] \
  || { log "STOP: the flipquant worktree is not clean at b1c4123"; exit 1; }
log "BEGIN flipquant b1c4123 vs NVFP4-RaZeR $(git -C $RZ rev-parse --short HEAD)"
run fq_llama_A $FQ $PY -m calibration.train_map_razer --model llama3.1-8b --unit 16x64 --data-root $FQDATA --out $O/fq_llama_A
run fq_llama_C $FQ $PY -m calibration.train_map_razer --model llama3.1-8b --unit 16x64 --teacher-topk 256 --epochs 5 \
    --data-root $FQDATA --out $O/fq_llama_C
run fq_phi4 $FQ $PY -m calibration.train_map_razer --model phi4-14b --unit 16x64 --data-root $FQDATA --out $O/fq_phi4
run rz_phi4 $RZ env PYTHONPATH=$RZ CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY run_train_map.py --model phi4 --data-root /home/dev/n16k64_campaign/multimodel/data --unit 16x64 --tm-opt --tile-grad-tc \
    --param ste --lr 0.02 --init-logit -1 --epochs 20 --eval-every 2 --no-dev --no-eval --out $O/rz_phi4
[ -z "$(git -C $FQ status --porcelain)" ] && log "flipquant worktree still clean" || log "NOTE: the flipquant worktree changed"
log "DONE"

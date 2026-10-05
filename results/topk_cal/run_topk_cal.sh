#!/bin/bash
# The calibration cost of TM-OPT+TC with flipquant's top-K teacher (results/topk_cal/PROTOCOL.md): Llama-3.1-8B 16x64,
# deterministic, paper settings, --no-dev --no-eval, one run at a time:
#   A  full-vocabulary teacher, 20 epochs (the paper setting; its map must equal the paper's 16x64 map, else STOP)
#   B  full-vocabulary teacher, 5 epochs
#   C  top-256 teacher (--teacher-topk 256), 5 epochs
# Run with `bash` from the topk-cal worktree. Outputs in /home/dev/n16k64_campaign/topk/runs.
set -u
W=$(cd "$(dirname "$0")/../.." && pwd)
D=/home/dev/n16k64_campaign/topk
R=$D/runs
PY=/home/dev/.conda/envs/n16k64/bin/python
LOG=$D/commands.log
cd $W
export HF_HUB_OFFLINE=0 PYTHONPATH=$W CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
mkdir -p $R $D/logs
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run() { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $D/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
        [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
TC="--model llama8b --data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation --unit 16x64 --tm-opt
    --tile-grad-tc --param ste --lr 0.02 --init-logit -1 --eval-every 2 --no-dev --no-eval"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy"; exit 1; }
log "BEGIN topk_cal at $(git rev-parse HEAD)"
run A_full_e20 $PY run_train_map.py $TC --epochs 20 --teacher-topk 0 --out $R/A_full_e20
new=$(sha256sum $R/A_full_e20/map.pt | cut -d' ' -f1)
ref=$(sha256sum /home/dev/n16k64_campaign/paper/maps/llama8b_16x64/map.pt | cut -d' ' -f1)
[ "$new" = "$ref" ] && log "BITWISE OK A = the paper's 16x64 map ($new)" || { log "STOP A's map $new differs from the paper's $ref"; exit 1; }
run B_full_e5 $PY run_train_map.py $TC --epochs 5 --teacher-topk 0 --out $R/B_full_e5
run C_top256_e5 $PY run_train_map.py $TC --epochs 5 --teacher-topk 256 --out $R/C_top256_e5
log "DONE topk_cal"

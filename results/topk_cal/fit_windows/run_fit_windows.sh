#!/bin/bash
# Calibration cost with more fit windows and the top-1000 teacher (results/topk_cal/fit_windows/NOTE.md): Llama-3.1-8B
# 16x64, TM-OPT+TC paper settings otherwise, deterministic, --no-dev --no-eval, one run at a time on an idle GPU.
#   check  the default path is unchanged: --teacher-topk 256 --epochs 5 must give topk-cal run C's map (af6c7515...)
#   E128   --teacher-topk 1000 --epochs 5                    (the K = 1000 reference, 80 steps)
#   E256   --fit-windows 256 --teacher-topk 1000 --epochs 5  (160 steps)
#   E512   --fit-windows 512 --teacher-topk 1000 --epochs 5  (320 steps, as the paper's 128 x 20)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd)
D=/home/dev/n16k64_campaign/topk
R=$D/fit_windows
PY=/home/dev/.conda/envs/n16k64/bin/python
cd $W
export HF_HUB_OFFLINE=0 PYTHONPATH=$W CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
mkdir -p $R/logs
log() { echo "$(date -u +%FT%TZ) $*" >> $R/commands.log; }
run() { local name=$1; shift; nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "STOP: GPU busy before $name"; exit 1; }
        log "START $name $*"; "$@" < /dev/null > $R/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
        [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
TC="--model llama8b --data-root /home/dev/n16k64_campaign/cost_comparison/data --transformers-deviation --unit 16x64 --tm-opt
    --tile-grad-tc --param ste --lr 0.02 --init-logit -1 --eval-every 2 --no-dev --no-eval"
log "BEGIN fit_windows at $(git rev-parse --short HEAD) (+ uncommitted --fit-windows)"
run check_C $PY run_train_map.py $TC --epochs 5 --teacher-topk 256 --out $R/check_C
new=$(sha256sum $R/check_C/map.pt | cut -d' ' -f1)
[ "$new" = af6c751500cc59fa070702b61e913a0e1cfa0d46eef80cfecf9cd462d4cd1784 ] && log "CHECK OK: run C's map ($new)" \
  || { log "STOP: check_C's map $new is not run C's"; exit 1; }
run E128 $PY run_train_map.py $TC --epochs 5 --teacher-topk 1000 --out $R/E128
run E256 $PY run_train_map.py $TC --epochs 5 --teacher-topk 1000 --fit-windows 256 --out $R/E256
run E512 $PY run_train_map.py $TC --epochs 5 --teacher-topk 1000 --fit-windows 512 --out $R/E512
log "DONE fit_windows"

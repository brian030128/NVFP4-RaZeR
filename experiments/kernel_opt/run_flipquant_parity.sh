#!/bin/bash
# flipquant GEMM parity (results/kernel_opt/flipquant_parity/NOTE.md): the paired timing run of one model, then the
# binaries check. Stops at the first failure. Nothing in the flipquant worktree is modified. Run with `bash`.
#   bash experiments/kernel_opt/run_flipquant_parity.sh llama8b
set -e
MODEL=${1:-llama8b}
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTHONPATH PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
OUT=/home/dev/n16k64_campaign/kernel_opt/fq_parity
FQ=/home/dev/n16k64_campaign/fqport/wt
mkdir -p $OUT/logs
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN flipquant GEMM parity $MODEL at $(git rev-parse HEAD) (flipquant $(git -C $FQ rev-parse HEAD))"
step TIMING_$MODEL $PY experiments/kernel_opt/flipquant_parity_gemm.py --model $MODEL --fq-root $FQ \
     --rz-build /home/dev/n16k64_campaign/kernel_opt/build_V --out $OUT/gemm_$MODEL.json
step BINARIES_$MODEL $PY experiments/kernel_opt/flipquant_parity_binaries.py --timing $OUT/gemm_$MODEL.json \
     --out $OUT/binaries_$MODEL.json
log "DONE $MODEL"

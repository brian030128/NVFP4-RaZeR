#!/bin/bash
# flipquant GEMM parity (results/kernel_opt/flipquant_parity/NOTE.md): one paired timing run of a model, then the
# binaries check. Stops at the first failure. Nothing in the flipquant worktree is modified. Run with `bash`.
#   bash experiments/kernel_opt/run_flipquant_parity.sh llama8b            # run 1 -> gemm_llama8b.json
#   bash experiments/kernel_opt/run_flipquant_parity.sh llama8b r2         # a repeat with fresh processes
#   bash experiments/kernel_opt/run_flipquant_parity.sh llama8b aa aa      # the A/A control (two RaZeR processes)
set -e
MODEL=${1:-llama8b}
TAG=${2:+_$2}
CONTROL=${3:-none}
cd /home/dev/NVFP4-RaZeR
unset SM120_BUILD_DIR PYTHONPATH PYTORCH_CUDA_ALLOC_CONF
export CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1
PY=/home/dev/.conda/envs/n16k64/bin/python
OUT=/home/dev/n16k64_campaign/kernel_opt/fq_parity
FQ=/home/dev/n16k64_campaign/fqport/wt
RUN=$MODEL$TAG
mkdir -p $OUT/logs
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> $OUT/commands.log; }
step() { local name=$1; shift; log "START $name"; set +e; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; set -e; log "END $name rc=$rc"; [ $rc -eq 0 ] || { log "STOP: $name failed"; exit 1; }; }
log "BEGIN flipquant GEMM parity $RUN (control $CONTROL) at $(git rev-parse HEAD) (flipquant $(git -C $FQ rev-parse HEAD))"
step TIMING_$RUN $PY experiments/kernel_opt/flipquant_parity_gemm.py --model $MODEL --fq-root $FQ \
     --rz-build /home/dev/n16k64_campaign/kernel_opt/build_V --control $CONTROL --out $OUT/gemm_$RUN.json
step BINARIES_$RUN $PY experiments/kernel_opt/flipquant_parity_binaries.py --timing $OUT/gemm_$RUN.json \
     --out $OUT/binaries_$RUN.json
log "DONE $RUN"

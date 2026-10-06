#!/bin/bash
# The remaining dev-KL runs after Phi-4 N: Llama N, Llama O, Phi-4 O (one at a time on an idle GPU).
set -u
B=/home/dev/n16k64_campaign/fqrel; PY=/home/dev/.conda/envs/n16k64/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> $B/queue.log; }
for run in "llama8b N" "llama8b O" "phi4 O"; do
  set -- $run
  until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && ! pgrep -f "[r]un_train_map.py" >/dev/null; do sleep 30; done
  log "START devkl $1 $2"
  ( cd $B && env -u PYTHONPATH -u SM120_BUILD_DIR -u PYTORCH_CUDA_ALLOC_CONF PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=0 $PY devkl_run.py $1 $2 ) < /dev/null > $B/devkl/logs/$1_$2.log 2>&1
  log "END devkl $1 $2 rc=$?"
done
log "DONE devkl"

#!/bin/bash
# Qwen3-1.7B dev-KL runs (N, O), then the PPL of the N epoch-10 16x64 maps (Llama, Phi-4, Qwen3-1.7B) and of the
# lowest-dev-KL N maps where that is not epoch 10 (Phi-4 epoch 7; Qwen3-1.7B decided after its run). One job at a time.
set -u
B=/home/dev/n16k64_campaign/fqrel; W=/home/dev/n16k64_campaign/fqopt/wt; PY=/home/dev/.conda/envs/n16k64/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> $B/queue.log; }
idle() { until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && ! pgrep -f "[r]un_train_map.py" >/dev/null; do sleep 15; done; }
ENV="env -u PYTHONPATH -u SM120_BUILD_DIR -u PYTORCH_CUDA_ALLOC_CONF PYTHONDONTWRITEBYTECODE=1 HF_HOME=/home/dev/.cache/huggingface HF_HUB_OFFLINE=0"
for s in N O; do
  idle; log "START devkl qwen3_1p7b $s"
  ( cd $B && $ENV $PY devkl_run.py qwen3_1p7b $s ) < /dev/null > $B/devkl/logs/qwen3_1p7b_$s.log 2>&1
  log "END devkl qwen3_1p7b $s rc=$?"
done
mkdir -p $B/devkl/ppl
ppl() {  # model registry key, trainer key, epoch
  local f=$(printf "%s/devkl/%s_16x64_N/map_epoch%03d.pt" $B $2 $3) out=$(printf "%s/devkl/ppl/%s_N_ep%03d.json" $B $1 $3)
  [ -f "$out" ] && return 0
  idle; log "START devkl ppl $1 ep$3"
  ( cd $W && $ENV $PY -m evaluation.ppl --model $1 --mode native --weight mixfp4 --map $f --unit 16x64 --paper-convention \
    --out $out ) < /dev/null > $B/devkl/logs/ppl_$1_ep$3.log 2>&1
  log "END devkl ppl $1 ep$3 rc=$?"
}
ppl llama3.1-8b llama8b 10
ppl phi4-14b phi4 10
ppl qwen3-1.7b qwen3_1p7b 10
ppl phi4-14b phi4 7
best=$(python3 -c "
import json; r=json.load(open('$B/devkl/qwen3_1p7b_16x64_N/report.json'))
print(min((e for e in r['epochs'] if 'dev_kl' in e), key=lambda e: e['dev_kl'])['epoch'] + 1)")
[ "$best" != "10" ] && ppl qwen3-1.7b qwen3_1p7b $best
log "DONE devkl3"

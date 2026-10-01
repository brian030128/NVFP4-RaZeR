#!/bin/bash
# flipquant-maps gate R: re-train the paper's Mistral-7B-v0.3 8x64 map with the paper's command; it must equal the
# committed map bitwise (sha256 8aacdd77..., experiments/paper/maps.sha256.json)
set -e
WT=/home/dev/n16k64_campaign/fqmaps/wt
cd $WT
unset SM120_BUILD_DIR
export PYTHONPATH=$WT/sm120:$WT CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131 HF_HOME=/home/dev/.cache/huggingface \
       HF_HUB_OFFLINE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/home/dev/n16k64_campaign/fqmaps/gateR
echo "$(date -u +%FT%T+00:00) START gateR at $(git rev-parse HEAD)" >> $G/commands.log
set +e
/home/dev/.conda/envs/n16k64/bin/python run_train_map.py --model mistral7b --data-root /home/dev/n16k64_campaign/multimodel/data \
  --unit 8x64 --tm-opt --tile-grad-tc --param ste --lr 0.02 --init-logit -1 --epochs 20 --eval-every 2 --no-dev --no-eval \
  --out $G/train > $G/train.log 2>&1
rc=$?
set -e
got=$(sha256sum $G/train/map.pt 2>/dev/null | cut -d' ' -f1)
want=8aacdd7706ca96d6fc23702f8da7eb03ae36eb55247fbf0524d5f7d232ab132c
echo "$(date -u +%FT%T+00:00) END gateR rc=$rc map_sha256=$got equal=$([ "$got" = "$want" ] && echo yes || echo NO)" >> $G/commands.log

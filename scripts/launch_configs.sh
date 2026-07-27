#!/bin/bash
# Launch run_ppl_sweep.py with an explicit --configs list, one shard per FREE GPU.
#
# Unlike launch_sweep.sh this takes the config list on the command line, so exploring a new
# selection rule does not need a code edit. Configs are round-robined across shards by index, and
# every shard writes its own JSON, so a shard that dies loses only its own configs.
#
# This is a shared machine: a GPU is only used if it currently has no compute process and
# essentially no memory in use.
#
# Usage: scripts/launch_configs.sh <model> <w4a16|w4a4> <configs> <outdir> <logdir> [datasets]

set -u
MODEL=${1:?model name}
SWEEP=${2:?w4a16 or w4a4}
CONFIGS=${3:?comma-separated config list}
OUTDIR=${4:?output dir}
LOGDIR=${5:?log dir}
DATASETS=${6:-wikitext,c4}

PYTHON=${PYTHON:-/home/brain_l/.conda/envs/razer/bin/python}
MEM_FREE_MIB=${MEM_FREE_MIB:-500}
export HF_HOME=${HF_HOME:-/share2/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

mkdir -p "$OUTDIR" "$LOGDIR"

BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)

FREE=()
while IFS=, read -r idx used uuid; do
    idx=$(echo "$idx" | tr -d ' ')
    used=$(echo "$used" | tr -d ' MiB')
    uuid=$(echo "$uuid" | tr -d ' ')
    if [[ -n "$BUSY" ]] && grep -q "$uuid" <<< "$BUSY"; then
        echo "gpu$idx: in use by another process, skipping"
        continue
    fi
    if (( used > MEM_FREE_MIB )); then
        echo "gpu$idx: ${used}MiB already allocated, skipping"
        continue
    fi
    FREE+=("$idx")
done < <(nvidia-smi --query-gpu=index,memory.used,gpu_uuid --format=csv,noheader)

N=${#FREE[@]}
if (( N == 0 )); then
    echo "No free GPUs. Nothing launched."
    exit 1
fi
echo "Free GPUs: ${FREE[*]}  ->  $N shards"

for i in "${!FREE[@]}"; do
    gpu=${FREE[$i]}
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" run_ppl_sweep.py \
        --model_name "$MODEL" --sweep "$SWEEP" --configs "$CONFIGS" \
        --datasets "$DATASETS" --seq_len 2048 \
        --shard_id "$i" --num_shards "$N" \
        --output "$OUTDIR/${MODEL}_${SWEEP}.shard${i}.json" \
        > "$LOGDIR/${MODEL}_${SWEEP}_s${i}.log" 2>&1 &
    echo "  gpu$gpu -> shard $i/$N (pid $!)"
done
wait
echo "All shards finished."

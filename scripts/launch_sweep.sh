#!/bin/bash
# Launch run_ppl_sweep.py shards, one per FREE GPU.
#
# This is a shared machine: a GPU is only used if it currently has no compute process and
# essentially no memory in use. Shards are assigned to free GPUs only, and the number of shards
# equals the number of free GPUs found, so nothing is ever placed on someone else's card.
#
# Usage: scripts/launch_sweep.sh <model_name> <sweep> <output_dir> <log_dir> [shard_id_offset]

set -u
MODEL=${1:?model name}
SWEEP=${2:?sweep name}
OUTDIR=${3:?output dir}
LOGDIR=${4:?log dir}
OFFSET=${5:-0}

PYTHON=${PYTHON:-/home/brain_l/.conda/envs/razer/bin/python}
MEM_FREE_MIB=${MEM_FREE_MIB:-500}

mkdir -p "$OUTDIR" "$LOGDIR"

# GPUs with a running compute process are off limits regardless of memory reported
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
    sid=$((OFFSET + i))
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" run_ppl_sweep.py \
        --model_name "$MODEL" --sweep "$SWEEP" \
        --datasets wikitext,c4 --seq_len 2048 \
        --shard_id "$i" --num_shards "$N" \
        --output "$OUTDIR/${MODEL}_${SWEEP}.shard${sid}.json" \
        > "$LOGDIR/${MODEL}_${SWEEP}_s${sid}.log" 2>&1 &
    echo "  gpu$gpu -> shard $i/$N (pid $!)"
done

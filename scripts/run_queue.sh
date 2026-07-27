#!/bin/bash
# Run a queue of sweeps back to back, each taking every free GPU.
#
# Each line of the queue file is:   <model> <w4a16|w4a4> <outdir> <configs-file>
# Blank lines and lines starting with # are skipped.
#
# Unlike chaining launch_configs.sh by hand, this waits for the previous sweep to actually finish
# before starting the next, so a queue can be left running unattended.
#
# Usage: scripts/run_queue.sh <queue-file> <log-root>

set -u
QUEUE=${1:?queue file}
LOGROOT=${2:?log root}
HERE=$(dirname "$0")

mkdir -p "$LOGROOT"

while read -r model sweep outdir cfgfile; do
    [[ -z "${model:-}" || "$model" == \#* ]] && continue
    [[ -f "$cfgfile" ]] || { echo "!! missing config file $cfgfile, skipping"; continue; }

    # wait for any sweep still running, including one launched outside this queue
    while pgrep -f '[r]un_ppl_sweep.py --model_name' > /dev/null; do sleep 60; done

    echo "=== $(date +%H:%M:%S)  $model / $sweep -> $outdir"
    CONFIGS=$(cat "$cfgfile")
    "$HERE/launch_configs.sh" "$model" "$sweep" "$CONFIGS" \
        "$outdir" "$LOGROOT/$(basename "$outdir")" wikitext,c4

    # launch_configs.sh ends with `wait`, so control returns only once every shard has exited
    echo "=== $(date +%H:%M:%S)  done $outdir"
done < "$QUEUE"

echo "Queue complete."

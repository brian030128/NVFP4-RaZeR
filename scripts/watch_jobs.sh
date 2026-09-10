#!/bin/bash
# Emit one line per NEW interesting log line across a set of Slurm jobs, then exit when every
# job has left the queue. Used as the command of a Monitor, so each new line is one event.
#
#   scripts/watch_jobs.sh <state-file> <jobid> [jobid ...]
set -uo pipefail
STATE=$1; shift
JOBS=("$@")
IDS=$(IFS=,; echo "${JOBS[*]}")
: > "$STATE"

while true; do
    NOW=$(mktemp)
    for j in "${JOBS[@]}"; do
        for f in slurm/logs/*_"$j".out slurm/logs/*_"$j".err; do
            [ -f "$f" ] || continue
            grep -haE "mean [0-9]|Done\. Results|SKIP |CUDA out of memory|Traceback|Error:|FATAL|=== done|Trials|RuntimeError|Mean:" "$f" 2>/dev/null \
                | tr -d '\r' | cut -c1-200 | sed -e "s|^|[$j] |"
        done
    done | sort -u > "$NOW"
    comm -13 "$STATE" "$NOW"
    mv "$NOW" "$STATE"

    LEFT=$(sacct -j "$IDS" --format=JobID,State --noheader 2>/dev/null \
           | grep -v '\.' | grep -cE 'RUNNING|PENDING')
    if [ "$LEFT" = "0" ]; then
        echo "ALL JOBS TERMINAL: $IDS"
        break
    fi
    sleep 240
done

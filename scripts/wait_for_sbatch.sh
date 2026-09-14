#!/bin/bash
# sbatch started failing with "ERROR: Oops! Something went wrong! get_api_token" while squeue
# and sacct kept working, so this is the cluster's submission auth rather than anything in the
# job scripts. Poll until it recovers, then say so.
set -uo pipefail
for i in $(seq 1 "${TRIES:-40}"); do
    out=$(sbatch --wrap="echo ok" --job-name=authtest --time=00:01:00 \
          --partition=taide --account=gov113008 --gres=gpu:H100:1 2>&1)
    case "$out" in
        *"Submitted batch job"*)
            echo "SBATCH RECOVERED after about $((i * 2)) min"
            echo "$out"
            exit 0
            ;;
    esac
    sleep "${INTERVAL:-120}"
done
echo "SBATCH STILL FAILING: $out"

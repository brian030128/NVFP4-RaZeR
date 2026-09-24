#!/bin/bash
# Usage: launch.sh NAME (multiround|distill|pool|prep) ARGS...   -- runs one job in the foreground, logging the command.
set -u
NAME=$1; KIND=$2; shift 2
REPO=/home/dev/NVFP4-RaZeR
RUNS=/home/dev/n16k64_campaign/cost_comparison/runs
PY=/home/dev/.conda/envs/n16k64/bin/python
cd $REPO
export HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
case $KIND in
  multiround) export PYTHONPATH=$REPO; SCRIPT=run_multiround.py ;;
  distill)    export PYTHONPATH=$REPO:/home/dev/n16k64_campaign/cost_comparison/pydeps; SCRIPT=run_cost_distill.py ;;
  pool)       export PYTHONPATH=$REPO; SCRIPT=prepare_distill_pool.py ;;
esac
echo "$(date -u +%FT%TZ) START $NAME PYTHONPATH=$PYTHONPATH $PY $SCRIPT $*" >> $RUNS/commands.log
$PY $SCRIPT "$@" > $RUNS/logs/$NAME.log 2>&1
RC=$?
echo "$(date -u +%FT%TZ) END $NAME rc=$RC" >> $RUNS/commands.log
exit $RC

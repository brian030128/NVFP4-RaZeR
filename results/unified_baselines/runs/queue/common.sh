# Shared by the unified-baselines queues (results/unified_baselines/PROTOCOL.md). Sourced, not run.
D=/home/dev/n16k64_campaign/unified_baselines
R=$D/runs
A=$D/artifacts
DE=/home/dev/n16k64_campaign/deploy_eval/artifacts
SA=/home/dev/n16k64_campaign/scale_additivity
PY=/home/dev/.conda/envs/n16k64/bin/python
REPO=/home/dev/NVFP4-RaZeR
LOG=$D/commands.log
cd $REPO
export HF_HUB_OFFLINE=0 PYTHONPATH=$REPO CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HOME=/home/dev/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=/home/dev/.conda/envs/mixfp4-cuda131
QPATH=$REPO:/home/dev/n16k64_campaign/cost_comparison/pydeps
mkdir -p $R $A $D/logs
log () { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
run () { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $D/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"
         [ $rc -eq 0 ] || { log "STOP run failed: $name"; exit 1; }; }
probe () { local name=$1; shift; log "START $name $*"; "$@" < /dev/null > $D/logs/$name.log 2>&1; local rc=$?; log "END $name rc=$rc"; }
root () { [ $1 = llama8b ] && echo /home/dev/n16k64_campaign/cost_comparison/data || echo /home/dev/n16k64_campaign/multimodel/data; }
deviation () { [ $1 = llama8b ] && echo --transformers-deviation; }
UNIFIED="--budget c1 --epochs 20 --act-rows --deterministic"
QAT="--arm qat --optimizer adamw_fp32 --micro-batch 8 --checkpointing"
SCALE="--arm scale --micro-batch 8"
# the learning-rate selection of one arm: run the rates choose_lr.py asks for, then echo the chosen rate
select_lr () {
  local M=$1 ARM=$2 FLAGS=$3 next
  mkdir -p $R/$M
  while true; do
    next=$($PY results/unified_baselines/choose_lr.py next $R/$M $ARM) || { log "STOP choose_lr failed: $M $ARM"; exit 1; }
    case "$next" in
      DONE*) log "CHOSEN $M $ARM ${next#DONE }"; CHOSEN=${next#DONE }; return 0 ;;
      STOP*) log "STOP $M $ARM: ${next#STOP }"; exit 1 ;;
      *) run ${M}_${ARM}_dev_lr$next env PYTHONPATH=$QPATH $PY run_cost_distill.py --model $M --data-root $(root $M) $(deviation $M) \
             $FLAGS $UNIFIED --lr $next --out $R/$M/${ARM}_dev_lr$next ;;
    esac
  done
}
# the deployed (--no-dev) run must repeat the chosen development run's per-step training KL exactly
check_repeat () {
  local M=$1 ARM=$2
  $PY -c "import json, sys; a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); sys.exit(0 if [x['kl'] for x in a['log']] == [x['kl'] for x in b['log']] else 1)" \
      $R/$M/${ARM}_nodev/report.json $R/$M/${ARM}_dev_lr$CHOSEN/report.json \
    && log "REPEAT OK $M $ARM (the deployed run's per-step KL = ${ARM}_dev_lr$CHOSEN's)" \
    || { log "STOP $M $ARM: the deployed run does not repeat the chosen development run"; exit 1; }
}

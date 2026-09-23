# source repro_local/env.sh  -- shared environment for local N16K64 reproduction runs
REPRO_REPO=/home/dev/NVFP4-RaZeR-n16k64
export CAMPAIGN_ROOT=/home/dev/n16k64_campaign
export HF_HOME=/home/dev/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$REPRO_REPO/research/n16k64/software/primary/support:$REPRO_REPO/research/n16k64/software/primary:$REPRO_REPO
export PY=/home/dev/.conda/envs/n16k64/bin/python
export FREEZE=$CAMPAIGN_ROOT/freeze/PROTOCOL_FREEZE.json
export FREEZE_SHA=$(sha256sum $FREEZE | cut -d' ' -f1)

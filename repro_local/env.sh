# source repro_local/env.sh  -- shared environment for local N16K64 reproduction runs
# The checkout this file lives in, wherever it is (no hardcoded clone path).
REPRO_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-/home/dev/n16k64_campaign}
# The campaign code expects the frozen protocol under $CAMPAIGN_ROOT/freeze; point it at this checkout.
mkdir -p "$CAMPAIGN_ROOT/freeze" "$CAMPAIGN_ROOT/runs"
for f in PROTOCOL_FREEZE.json PROTOCOL_FREEZE.sha256; do
  ln -sfn "$REPRO_REPO/research/n16k64/campaigns/primary/$f" "$CAMPAIGN_ROOT/freeze/$f"
done
export HF_HOME=/home/dev/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$REPRO_REPO/research/n16k64/software/primary/support:$REPRO_REPO/research/n16k64/software/primary:$REPRO_REPO
export PY=/home/dev/.conda/envs/n16k64/bin/python
export FREEZE=$CAMPAIGN_ROOT/freeze/PROTOCOL_FREEZE.json
export FREEZE_SHA=$(sha256sum $FREEZE | cut -d' ' -f1)

#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=llama_new_ppl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64G
#SBATCH --time=00:12:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL=llama8b
source slurm/task_reorder_env.sh
ROOT=/work/u4320956/task_reorder/transfer_20260920/llama8b
CONFIRM="$ROOT/${1:?confirmation directory}"
OUT="$ROOT/${2:?new output directory}"
LAYOUTS=$("$PYTHON" - "$CONFIRM" <<'PY'
from pathlib import Path
import json,sys
from scripts.ce_confirmation_gate import eligible
p=Path(sys.argv[1]);assert eligible(p), 'Fresh frozen CE gate failed; do not promote to PPL'
plan=json.loads((p/'plan.json').read_text())
print(plan['candidate'])
PY
)
# Published FourOverSix, fine8x64, and supplied raw256 controls are reused.
"$PYTHON" run_task_reorder_eval.py --model llama8b --calib "$ROOT/calibration" \
  --layouts "refined=$LAYOUTS" --background raw256 --policies refined --out "$OUT"

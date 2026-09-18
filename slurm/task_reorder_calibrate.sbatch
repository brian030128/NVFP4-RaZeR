#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=reorder_cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL="${1:?model}"; OUTPUT="${2:?output}"
case "$MODEL" in
    llama8b) MODULES='^model\.layers\.31\.mlp\.(gate|up|down)_proj$' ;;
    qwen27b) MODULES='^model\.language_model\.layers\.63\.mlp\.(gate|up|down)_proj$' ;;
    *) exit 2 ;;
esac
source slurm/task_reorder_env.sh
"$PYTHON" tests/test_task_reorder.py
"$PYTHON" run_math_code_calibration.py --model "$MODEL" --out "$OUTPUT" \
    --reorder-modules "$MODULES" --allow-source-drift

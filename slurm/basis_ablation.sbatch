#!/bin/bash
# Does the gain come from E0M3, or from task-gradient selection over any legal
# 4-bit knob? Three bases (see quantize/basis.py), identical scoring, election,
# counts and evaluation. Arm e0m3 is the reported direction, re-run through the
# same new code path so the arms are directly comparable.
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=basis_ablation
#SBATCH --array=0-5%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --time=03:00:00
#SBATCH --mem=80G
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="/tmp/$USER/hf_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON=/home/u4320956/beam_engine-v3/.venv/bin/python

"$PYTHON" tests/test_basis.py
"$PYTHON" tests/test_adaptive_prefix.py
"$PYTHON" tests/test_causal_four_over_six.py

MODELS=(qwen4b qwen4b qwen4b llama8b llama8b llama8b)
BASES=(e0m3 alpha type_pure e0m3 alpha type_pure)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
BASIS="${BASES[$SLURM_ARRAY_TASK_ID]}"
STEM="results/basis_ablation/${SLURM_ARRAY_JOB_ID}_${MODEL}_${BASIS}"
echo "=== ARM model=$MODEL basis=$BASIS stem=$STEM $(date -Iseconds) ==="

"$PYTHON" run_math_code_calibration.py --model "$MODEL" --basis "$BASIS" \
  --out "$STEM/calibration"

"$PYTHON" run_math_code_evaluation.py \
  --calibration "$STEM/calibration" --out "$STEM/evaluation" --no-reuse

echo "=== done model=$MODEL basis=$BASIS $(date -Iseconds) ==="

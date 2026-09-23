#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=reorder_objective_ablation_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL="${REORDER_MODEL:-llama8b}"
source slurm/task_reorder_env.sh
"$PYTHON" -m py_compile run_reorder_objective_ablation.py \
  summarize_reorder_objective_ablation.py quantize/task_reorder.py run_task_reorder.py
# New ablation tests first, so a pre-existing failure cannot mask them.
"$PYTHON" tests/test_reorder_objective_ablation.py
# The deployed search must be unchanged by this branch.
"$PYTHON" tests/test_task_reorder.py

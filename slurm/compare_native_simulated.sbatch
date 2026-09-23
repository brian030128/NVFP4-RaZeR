#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=compare_native_sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Pair two perplexity reports window by window. Pointing both at the same native
# report with different policies gives that run's own gain of one policy over
# another, paired across windows rather than as a difference of two aggregates.
#
#   sbatch slurm/compare_native_simulated.sbatch NATIVE SIM [extra args]
#
# Extra args are forwarded, so --native-policy, --simulated-policy and --out all
# work.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MODEL="${REORDER_MODEL:-llama8b}"
source slurm/task_reorder_env.sh
NATIVE="${1:?Usage: sbatch ... NATIVE SIM [extra args]}"
SIM="${2:?Usage: sbatch ... NATIVE SIM [extra args]}"
shift 2
"$PYTHON" scripts/compare_native_simulated_ppl.py --native "$NATIVE" --simulated "$SIM" "$@"

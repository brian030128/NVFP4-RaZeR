#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=reorder_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# CPU test suites plus the diagnostics summary. Everything here is CPU-only; the GPU is requested
# solely because this cluster rations cores at 12 per GPU and has no CPU-only partition.
#
#   sbatch slurm/reorder_tests.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="$HOME/.local/bin:$PATH"
PYTHON=${PYTHON:-/home/u4320956/DiRotQ/.venv/bin/python}
export PYTHONUNBUFFERED=1

mkdir -p slurm/logs

echo "############ tests/test_reorder.py ############"
"$PYTHON" tests/test_reorder.py
echo "reorder tests exit=$?"

echo
echo "############ tests/test_mixfp4.py (regression: quantizer.py was touched) ############"
"$PYTHON" tests/test_mixfp4.py
echo "mixfp4 tests exit=$?"

echo
echo "############ diagnostics summary vs the i.i.d. noise expectation ############"
for f in results/reorder/diag_*.csv; do
    [[ -e "$f" ]] || continue
    "$PYTHON" analyze_reorder.py "$f"
    echo
done

echo "=== done $(date -Iseconds) ==="

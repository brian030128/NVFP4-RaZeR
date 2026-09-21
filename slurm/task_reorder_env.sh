#!/bin/bash
# Source only from a Slurm worker. Environments/cache are shared across stages.
: "${SLURM_JOB_ID:?Run through Slurm}"
export HF_HOME="${REORDER_HF_HOME:-$PWD/.joblib/huggingface}"
export TMPDIR="/tmp/$USER/reorder_$SLURM_JOB_ID/tmp"
export PYTHONPYCACHEPREFIX="/tmp/$USER/reorder_$SLURM_JOB_ID/pycache"
mkdir -p "$TMPDIR"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHON="${PYTHON:-$PWD/.joblib/${MODEL:-llama8b}/bin/python}"
test -x "$PYTHON" || { echo 'Run slurm/task_reorder_setup.sbatch first' >&2; exit 1; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

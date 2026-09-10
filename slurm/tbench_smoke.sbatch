#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=tbench_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Can Terminal-Bench 4.0 run on this cluster at all?
#
# Terminal-Bench is driven by the `harbor` CLI and defaults to a Docker sandbox. This node has
# the docker CLI but no reachable daemon and the account is in no docker group, so that path is
# closed; the cloud sandboxes (modal, daytona, e2b, ...) all need an external account. What is
# available is `singularity`, and harbor ships a first-class `singularity` environment written
# for exactly this situation -- it converts each task's docker_image to a .sif and talks to a
# FastAPI server inside the container over HTTP.
#
# This job answers the prerequisite question before any model is involved: does the harness
# stand up here -- dataset download, unprivileged docker->sif conversion, container start,
# command execution, verifier -- using the `nop` agent, which does nothing at all. Every task
# must FAIL its tests under `nop`; the result of interest is whether trials reach the verifier
# rather than erroring in setup.
#
#   sbatch slurm/tbench_smoke.sbatch
#   NTASKS=5 sbatch slurm/tbench_smoke.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs

DATASET=${DATASET:-terminal-bench/terminal-bench@4.0.0}
NTASKS=${NTASKS:-3}
AGENT=${AGENT:-nop}

PROJECT="$PWD"
# Everything the harness touches -- SIF images, the dataset, uv's cache -- is large and churny,
# so it lives on the node-local RAID and is discarded with the job (see /home/u4320956/CLAUDE.md).
SCRATCH="/tmp/$USER/tbench_$SLURM_JOB_ID"
mkdir -p "$SCRATCH"/{sif,cache,tmp,jobs}
# Converted .sif images are large and are re-pulled per job by default. SIF_CACHE points them
# at a node-local directory that outlives the job, so an iteration on the same node reuses them
# (still node-local scratch, never /home or /work -- see /home/u4320956/CLAUDE.md).
SIF_CACHE=${SIF_CACHE:-$SCRATCH/sif}
mkdir -p "$SIF_CACHE"
trap 'rm -rf "$SCRATCH"' EXIT

export UV_CACHE_DIR="$SCRATCH/cache/uv"
export TMPDIR="$SCRATCH/tmp"
export SINGULARITY_CACHEDIR="$SCRATCH/cache/singularity"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$SCRATCH/tmp"
export APPTAINER_TMPDIR="$SCRATCH/tmp"
export HARBOR_HOME="$SCRATCH/harbor"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"

# harbor hardcodes --fakeroot when it starts a container, which this account cannot use (no
# /etc/subuid mapping) and does not need (singularity here is setuid). A shim named
# `singularity`, first on PATH, strips that one flag and passes everything else through.
# See scripts/singularity-no-fakeroot for the full reasoning.
SHIM="$SCRATCH/bin"
mkdir -p "$SHIM"
ln -sf "$PROJECT/scripts/singularity-no-fakeroot" "$SHIM/singularity"
export PATH="$SHIM:$PATH"

OUT="$PROJECT/results/terminal_bench/job_$SLURM_JOB_ID"
mkdir -p "$OUT"

echo "=== tbench smoke  $(date -Iseconds) on $(hostname) ==="
singularity --version
harbor --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Prerequisite 1: can this account convert a public docker image to a .sif without root?
echo "--- unprivileged docker->sif conversion"
timeout 900 singularity pull --force --dir "$SCRATCH/sif" alpine.sif docker://alpine:3.20 \
  > "$OUT/singularity_pull.log" 2>&1
echo "singularity pull exit=$? ; $(ls -la "$SCRATCH/sif" 2>/dev/null | tail -2)"

# Prerequisite 2: can a container actually execute a command here?
echo "--- container exec"
timeout 300 singularity exec "$SCRATCH/sif/alpine.sif" /bin/sh -c 'echo hello from $(cat /etc/alpine-release)' \
  2>&1 | tee "$OUT/singularity_exec.log"

# Prerequisite 3: the harness itself, with an agent that does nothing.
echo "--- harbor run ($AGENT, $NTASKS tasks, $DATASET)"
timeout 5400 harbor run \
    -d "$DATASET" \
    -e singularity \
    --ek "singularity_image_cache_dir=$SIF_CACHE" \
    -a "$AGENT" \
    -l "$NTASKS" \
    -n "${NCONCURRENT:-1}" \
    -q \
    -o "$SCRATCH/jobs" \
    --job-name "smoke_$SLURM_JOB_ID" \
    -y 2>&1 | tee "$OUT/harbor_run.log"
echo "harbor run exit=${PIPESTATUS[0]}"

# Keep the results, not the images: copy the job tree (small JSON/logs) back to /home.
if [[ -d "$SCRATCH/jobs" ]]; then
    cp -r "$SCRATCH/jobs" "$OUT/jobs" 2>/dev/null
    echo "--- results tree"
    find "$OUT/jobs" -maxdepth 3 | head -40
fi

echo "=== done $(date -Iseconds) ==="

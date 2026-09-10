#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=taide
#SBATCH --job-name=tbench_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Why does harbor's bootstrap die with "cannot install /usr/bin/python3" in a Terminal-Bench
# task image?
#
# The container itself now starts (scripts/singularity-no-fakeroot), and DNS works inside it,
# so the remaining failure is about the image's own contents: bootstrap.sh insists on the exact
# path /usr/bin/python3 and, when it is missing, falls through an apt/apk/dnf/yum chain that is
# apparently finding none of them -- or finding apt and failing anyway. This looks at the image
# instead of guessing: which interpreters exist, which package manager exists, and whether
# apt-get can actually install python3 from inside.
#
#   sbatch slurm/tbench_probe_image.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p slurm/logs

IMAGE=${IMAGE:-harborframework/terminal-bench:layout-config-recreation2-environment-9caa9d6614e4ae31@sha256:c08989cd9fc6858e7ec7488acf65b2840ece53581fb4327b3e667ff15da23724}
PROJECT="$PWD"
SCRATCH="/tmp/$USER/tbprobe_$SLURM_JOB_ID"
mkdir -p "$SCRATCH"/{sif,cache,tmp}
trap 'rm -rf "$SCRATCH"' EXIT

export SINGULARITY_CACHEDIR="$SCRATCH/cache" APPTAINER_CACHEDIR="$SCRATCH/cache"
export SINGULARITY_TMPDIR="$SCRATCH/tmp" APPTAINER_TMPDIR="$SCRATCH/tmp"
export TMPDIR="$SCRATCH/tmp"
SHIM="$SCRATCH/bin"; mkdir -p "$SHIM"
ln -sf "$PROJECT/scripts/singularity-no-fakeroot" "$SHIM/singularity"
export PATH="$SHIM:$HOME/.local/bin:$PATH"

OUT="$PROJECT/results/terminal_bench/probe_$SLURM_JOB_ID"
mkdir -p "$OUT"

echo "=== probing $IMAGE  $(date -Iseconds) on $(hostname) ==="
SIF="$SCRATCH/sif/task.sif"
timeout 2400 singularity pull --force "$SIF" "docker://$IMAGE" > "$OUT/pull.log" 2>&1
echo "pull exit=$? size=$(du -sh "$SIF" 2>/dev/null | cut -f1)"
[[ -f "$SIF" ]] || { echo "FATAL: no image"; tail -20 "$OUT/pull.log"; exit 1; }

echo "--- what is in the image"
timeout 600 singularity exec --writable-tmpfs --containall --pid --fakeroot "$SIF" /bin/sh -c '
  echo "id=$(id -u)";
  echo "os: $(cat /etc/os-release 2>/dev/null | head -2 | tr "\n" " ")";
  for p in /usr/bin/python3 /usr/local/bin/python3 /usr/bin/python /opt/conda/bin/python3; do
      [ -x "$p" ] && echo "have $p -> $($p -V 2>&1)";
  done
  echo "which python3: $(command -v python3 || echo none)";
  for m in apt-get apk dnf yum tmux curl; do
      echo "$m: $(command -v $m || echo none)";
  done
' 2>&1 | tee "$OUT/contents.log" | tail -20

echo "--- can apt-get install python3 from inside?"
timeout 900 singularity exec --writable-tmpfs --containall --pid --fakeroot "$SIF" /bin/sh -c '
  if command -v apt-get >/dev/null 2>&1; then
      apt-get update 2>&1 | tail -5;
      DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv 2>&1 | tail -8;
      echo "after: $([ -x /usr/bin/python3 ] && /usr/bin/python3 -V || echo still-missing)";
  else
      echo "no apt-get in this image";
  fi
' 2>&1 | tee "$OUT/aptget.log" | tail -20

echo "--- would a /usr/bin/python3 symlink to the image python satisfy bootstrap?"
timeout 600 singularity exec --writable-tmpfs --containall --pid --fakeroot "$SIF" /bin/sh -c '
  SRC=$(command -v python3);
  if [ -n "$SRC" ] && [ ! -x /usr/bin/python3 ]; then
      ln -sf "$SRC" /usr/bin/python3 && echo "symlinked /usr/bin/python3 -> $SRC";
      /usr/bin/python3 -m venv --without-pip /opt/harbor-server && echo "venv ok" || echo "venv FAILED";
      /opt/harbor-server/bin/python3 -m ensurepip --default-pip >/dev/null 2>&1 && echo "ensurepip ok" || echo "ensurepip failed";
  else
      echo "nothing to symlink (SRC=$SRC)";
  fi
' 2>&1 | tee "$OUT/symlink.log" | tail -10

echo "=== done $(date -Iseconds) ==="

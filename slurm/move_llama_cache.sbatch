#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=dev
#SBATCH --job-name=move_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Frees /home by relocating the base Llama-3.1-8B HF cache entry to /work and
# symlinking it back, so every HF_HOME path keeps resolving unchanged.
set -euo pipefail
SRC=/home/u4320956/NVFP4-RaZeR/.joblib/huggingface/hub/models--meta-llama--Llama-3.1-8B
DST=/work/u4320956/hf_relocated/models--meta-llama--Llama-3.1-8B
[[ -L "$SRC" ]] && { echo "already relocated"; exit 0; }
mkdir -p "$(dirname "$DST")"
cp -a "$SRC" "$DST"
# Verify byte-for-byte before removing the original.
diff -r "$SRC" "$DST"
rm -rf "$SRC"
ln -s "$DST" "$SRC"
ls -la "$SRC"
df -h /home/u4320956 /work/u4320956

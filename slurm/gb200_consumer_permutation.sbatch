#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_consumer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=12G
#SBATCH --time=00:03:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
OUT="$PWD/results/task_reorder/transfer_20260920/consumer_$SLURM_JOB_ID"
BUILD="/work/u4320956/b200/consumer-$SLURM_JOB_ID"
mkdir -p "$OUT" "$BUILD"
cp native/consumer_permutation.cu "$OUT/"
sha256sum native/consumer_permutation.cu /home/u4320956/mixfp4/src/gemm_benchmark.hpp > "$OUT/source.sha256"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_100a -I /home/u4320956/mixfp4/src native/consumer_permutation.cu -o "$BUILD/bench"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$OUT/hardware.csv"
export MIXFP4_BENCH_GRAPH=1
"$BUILD/bench" "$PWD/results/task_reorder/transfer_20260920/permutations" > "$OUT/results.txt"

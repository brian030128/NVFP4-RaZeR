#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_perm_chk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
BUILD="/work/u4320956/b200/permutation-check-$SLURM_JOB_ID"
OUT="$PWD/results/task_reorder/transfer_20260920/gb200_check_$SLURM_JOB_ID"
mkdir -p "$BUILD" "$OUT"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_100a -I native native/check_permutation_output.cu -o "$BUILD/check"
for proj in up_proj down_proj; do
 if [[ "$proj" == up_proj ]]; then N=17408; else N=5120; fi
 for M in 1 16 128 512 2048; do
  "$BUILD/check" "$PWD/results/task_reorder/transfer_20260920/permutations/$proj" "$M" "$N" >> "$OUT/checks.txt"
 done
done
cat "$OUT/checks.txt"

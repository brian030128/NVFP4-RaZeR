#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_fused
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=24G
#SBATCH --time=00:08:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH=/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
unset MIXFP4_SPARSE_OUTPUT MIXFP4_PERM_PREFIX
OUT="$PWD/results/task_reorder/transfer_20260920/fused_$SLURM_JOB_ID"
BUILD="/work/u4320956/b200/fused-permutation-$SLURM_JOB_ID"
mkdir -p "$OUT" "$BUILD"
python3 scripts/build_fused_permutation.py "$BUILD"
nvcc -O3 -std=c++17 -arch=sm_100a --expt-relaxed-constexpr -Xcompiler=-fopenmp \
 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
 -I "$BUILD/override" -I /home/u4320956/mixfp4/3rdparty/cutlass/include \
 -I /home/u4320956/mixfp4/3rdparty/cutlass/tools/util/include \
 -I /home/u4320956/mixfp4/3rdparty/cutlass/examples/common \
 -I /home/u4320956/mixfp4/src -I "$PWD/native" -I "$BUILD" \
 "$BUILD/fused_permutation.cu" -lcuda -o "$BUILD/bench"
cp "$BUILD/source.json" "$BUILD/permutation_store.cuh" "$BUILD/fused_permutation.cu" "$OUT/"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$OUT/hardware.csv"
"$BUILD/bench" 128 256 256 --check --iters=10 > "$OUT/smoke.txt"
export MIXFP4_BENCH_GRAPH=1
for proj in up_proj down_proj; do
 if [[ "$proj" == up_proj ]]; then N=17408;K=5120;else N=5120;K=17408;fi
 export MIXFP4_PERM_PREFIX="$PWD/results/task_reorder/transfer_20260920/permutations/$proj"
 for M in 1 128 512 2048; do
  "$BUILD/bench" "$M" "$N" "$K" --no-check --iters=100 > "$OUT/${proj}_m${M}.txt"
 done
done
printf 'COMPLETED %s\n' "$OUT"

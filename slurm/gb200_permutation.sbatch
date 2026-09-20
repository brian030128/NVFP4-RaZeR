#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_perm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH=/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MIXFP4_SPARSE_OUTPUT="${MIXFP4_SPARSE_OUTPUT:-}"
if [[ -z "$MIXFP4_SPARSE_OUTPUT" ]]; then unset MIXFP4_SPARSE_OUTPUT; fi
OUT="$PWD/results/task_reorder/transfer_20260920/gb200_${SLURM_JOB_ID}"
BUILD="/work/u4320956/b200/permutation-${SLURM_JOB_ID}"
mkdir -p "$OUT" "$BUILD"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$OUT/hardware.csv"
uname -a > "$OUT/platform.txt"
python3 scripts/build_permutation_benchmark.py /home/u4320956/mixfp4 "$BUILD"
nvcc -O3 -std=c++17 -arch=sm_100a --expt-relaxed-constexpr -Xcompiler=-fopenmp \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
  -I /home/u4320956/mixfp4/3rdparty/cutlass/include \
  -I /home/u4320956/mixfp4/3rdparty/cutlass/tools/util/include \
  -I /home/u4320956/mixfp4/3rdparty/cutlass/examples/common \
  -I /home/u4320956/mixfp4/src -I "$PWD/native" \
  "$BUILD/permutation_gemm.cu" -lcuda -o "$BUILD/permutation_gemm"
cp "$BUILD/source_manifest.json" "$OUT/"
"$BUILD/permutation_gemm" 128 256 256 --check --iters=10 > "$OUT/gemm_check.txt"
for mode in 1 0; do
  export MIXFP4_BENCH_GRAPH="$mode"
  for proj in up_proj down_proj; do
    if [[ "$proj" == up_proj ]]; then N=17408; K=5120; else N=5120; K=17408; fi
    export MIXFP4_PERM_PREFIX="$PWD/results/task_reorder/transfer_20260920/permutations/$proj"
    for M in 1 16 128 512 2048; do
      "$BUILD/permutation_gemm" "$M" "$N" "$K" --no-check --iters=100 > "$OUT/${proj}_m${M}_graph${mode}.txt"
    done
  done
done
printf 'COMPLETED %s\n' "$OUT"

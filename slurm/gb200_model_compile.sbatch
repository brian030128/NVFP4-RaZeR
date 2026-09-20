#!/bin/bash
#SBATCH --account=gov113008
#SBATCH --partition=gb200-dev
#SBATCH --job-name=gb200_modelcompile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GB200:1
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
BUILD=/home/u4320956/.cache/mixfp4-model-runtime/build
mkdir -p "$BUILD"
python3 scripts/build_native_model_runtime.py "$BUILD"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_100a --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -Xcompiler=-fopenmp \
 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
 -I "$BUILD" -I /home/u4320956/mixfp4/3rdparty/cutlass/include \
 -I /home/u4320956/mixfp4/3rdparty/cutlass/tools/util/include \
 -I /home/u4320956/mixfp4/3rdparty/cutlass/examples/common -I /home/u4320956/mixfp4/src \
 native/model_runtime.cu -lcuda -o "$BUILD/libmixfp4_model.so"
OUT="$PWD/results/task_reorder/full_model_20260920/kernel_$SLURM_JOB_ID"
mkdir -p "$OUT"
cp "$BUILD/source.json" "$OUT/"
/home/u4320956/.cache/mixfp4-model-runtime/venv/bin/python scripts/check_native_model_runtime.py "$BUILD/libmixfp4_model.so" "$OUT"

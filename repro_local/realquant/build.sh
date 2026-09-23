#!/bin/bash
# Build the mixfp4 SM120 mixed E2M1/E0M3 GEMM for one weight-granule configuration, as a
# standalone self-test executable (target=exe) or as the ctypes shared library (target=lib),
# then install the E0M3 formats with the repo's own SASS patcher.
#
#   repro_local/realquant/build.sh wt_as_A exe|lib    # N16K64: weights on A, 16 rows x 64 K
#   repro_local/realquant/build.sh b8x64   exe|lib    # N8K64:  weights on B,  8 cols x 64 K
#
# Mirrors /home/dev/mixfp4/scripts/build_mixed.sh (flags, SASS-only sm_120a) with the conda
# CUDA 13.1 toolchain in place of /usr/local/cuda-13.1, and builds from a per-config snapshot of
# the mixfp4 sources so the generated blob header never touches the mixfp4 checkout.
set -euo pipefail
CFG="$1"; TARGET="$2"
E=/home/dev/.conda/envs/mixfp4-cuda131
RQ=/home/dev/n16k64_campaign/realquant
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAP="$RQ/$CFG"
export PATH="$E/bin:$PATH"

case "$CFG" in
  wt_as_A)   # docs/mixed_nvfp4_report.md section 6: "WEIGHTS AS THE A OPERAND", stock 4x2 layout
    GEN="MMA_M=2 MMA_N=8 A_ATOMS=1 B_ATOMS=8"
    EXTRA="-DMIXFP4_BLOB=1 -DMIXFP4_JOINT_KA=1 -DMIXFP4_B_ALL_E2M1=1 -DMIXFP4_A_ATOMS_PER_GRANULE=1 -DMIXFP4_B_ATOMS_PER_GRANULE=8" ;;
  b8x64)     # section 6: "B at its 8-column floor with a 64-element K granule: 1x8 arrangement"
    GEN="MMA_M=8 MMA_N=2 A_ATOMS=8 B_ATOMS=1"
    EXTRA="-DMIXFP4_BLOB=1 -DMIXFP4_JOINT_KB=1 -DMIXFP4_A_ALL_E2M1=1 -DMIXFP4_ATOM_M=1 -DMIXFP4_PERM_N=128 -DMIXFP4_A_ATOMS_PER_GRANULE=8 -DMIXFP4_B_ATOMS_PER_GRANULE=1" ;;
  *) echo "unknown config $CFG" >&2; exit 1 ;;
esac

env $GEN python3 "$SNAP/scripts/gen_mixed_mma_blob.py"

COMMON=(-O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_120a,code=[sm_120a]"
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_ENABLE_GDC_FOR_SM100=1
  --expt-relaxed-constexpr -ftemplate-backtrace-limit=0
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DCUTLASS_SM100_FAMILY_ARCHS_ENABLED
  -Xcompiler=-fno-strict-aliasing
  -ccbin "$E/bin/x86_64-conda-linux-gnu-g++"
  -I"$SNAP/3rdparty/cutlass/examples/common" -I"$SNAP/src"
  -I"$SNAP/3rdparty/cutlass/include" -I"$RQ/cmake_cfg/3rdparty/cutlass/include"
  -I"$SNAP/3rdparty/cutlass/tools/util/include")
mkdir -p "$RQ/bin"
if [ "$TARGET" = exe ]; then
  OUT="$RQ/bin/${CFG}_exe"
  nvcc $EXTRA "${COMMON[@]}" -Xcompiler=-fopenmp "$SNAP/src/mixed_nvfp4_gemm.cu" -o "$OUT.unpatched" \
    -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lgomp
else
  OUT="$RQ/bin/lib${CFG}.so"
  nvcc $EXTRA "${COMMON[@]}" -Xcompiler=-fPIC -Xcompiler=-fopenmp -shared -Dmain=mixfp4_selftest_main \
    -I"$SNAP/src" "$HERE/realquant_gemm.cu" -o "$OUT.unpatched" -lcudart_static -lrt -lpthread -ldl -lgomp
fi
python3 "$SNAP/scripts/patch_mixed_nvfp4_gemm.py" --cuobjdump "$E/bin/cuobjdump" --allow-missing-sites \
  "$OUT.unpatched" "$OUT"
echo "built $OUT"

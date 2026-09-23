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
#
# Latency-study builds (library only, never patched -- they contain no E0M3 site):
#   stock           src/nvfp4_gemm.cu, the stock CUTLASS SM120 NVFP4 GEMM (no format dispatch)
#   wt_as_A_nodisp  wt_as_A with -DMIXFP4_NO_DISPATCH=1: same tile/warp arrangement, dispatch
#   b8x64_nodisp    b8x64   with -DMIXFP4_NO_DISPATCH=1   compiled out (the in-source ceiling)
#
# Output-layout variant (patched, E0M3 sites installed):
#   wt_as_A_colD    wt_as_A with C/D stored column-major (-DMIXFP4_D_COLMAJOR=1, a two-line guard
#                   added to the snapshot copy $RQ/wt_as_A_colD/src/mixed_nvfp4_gemm.cu). Its D is
#                   the row-major [tokens, out] tensor, so the weights-on-A output needs no transpose.
set -euo pipefail
CFG="$1"; TARGET="$2"
E=/home/dev/.conda/envs/mixfp4-cuda131
RQ=/home/dev/n16k64_campaign/realquant
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${CFG%_nodisp}"
SNAP="$RQ/$BASE"
[ "$CFG" = stock ] && SNAP="$RQ/wt_as_A"
COLD=0
if [ "$CFG" = wt_as_A_colD ]; then BASE=wt_as_A; SNAP="$RQ/wt_as_A_colD"; COLD=1; fi
export PATH="$E/bin:$PATH"
PATCH=1
SOURCE_DEF=""

case "$BASE" in
  wt_as_A)   # docs/mixed_nvfp4_report.md section 6: "WEIGHTS AS THE A OPERAND", stock 4x2 layout
    GEN="MMA_M=2 MMA_N=8 A_ATOMS=1 B_ATOMS=8"
    EXTRA="-DMIXFP4_BLOB=1 -DMIXFP4_JOINT_KA=1 -DMIXFP4_B_ALL_E2M1=1 -DMIXFP4_A_ATOMS_PER_GRANULE=1 -DMIXFP4_B_ATOMS_PER_GRANULE=8" ;;
  b8x64)     # section 6: "B at its 8-column floor with a 64-element K granule: 1x8 arrangement"
    GEN="MMA_M=8 MMA_N=2 A_ATOMS=8 B_ATOMS=1"
    EXTRA="-DMIXFP4_BLOB=1 -DMIXFP4_JOINT_KB=1 -DMIXFP4_A_ALL_E2M1=1 -DMIXFP4_ATOM_M=1 -DMIXFP4_PERM_N=128 -DMIXFP4_A_ATOMS_PER_GRANULE=8 -DMIXFP4_B_ATOMS_PER_GRANULE=1" ;;
  stock)
    GEN=""; EXTRA=""; SOURCE_DEF="-DRQ_STOCK=1"; PATCH=0 ;;
  *) echo "unknown config $CFG" >&2; exit 1 ;;
esac
[ "$COLD" = 1 ] && EXTRA="$EXTRA -DMIXFP4_D_COLMAJOR=1"

# One-time setup, so a fresh machine can rebuild from /home/dev/mixfp4 alone:
#  * a per-config source snapshot (the blob generator writes into it, never into the checkout);
#  * the column-major-D guard, a two-line patch applied only to the wt_as_A_colD snapshot;
#  * CUTLASS's generated version header (cmake configure only, no build).
MIXFP4=/home/dev/mixfp4
if [ ! -d "$SNAP" ]; then
  mkdir -p "$SNAP"
  cp -r "$MIXFP4/src" "$MIXFP4/scripts" "$MIXFP4/CMakeLists.txt" "$SNAP/"
  ln -s "$MIXFP4/3rdparty" "$SNAP/3rdparty"
  git -C "$MIXFP4" rev-parse HEAD > "$RQ/MIXFP4_COMMIT"
fi
if [ "$COLD" = 1 ] && ! grep -q MIXFP4_D_COLMAJOR "$SNAP/src/mixed_nvfp4_gemm.cu"; then
  python3 - "$SNAP/src/mixed_nvfp4_gemm.cu" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
old = "using LayoutCTag = cutlass::layout::RowMajor;\nusing LayoutDTag = cutlass::layout::RowMajor;\n"
new = ("// repro_local patch: -DMIXFP4_D_COLMAJOR=1 stores C/D column-major. With the weights on A the\n"
       "// kernel computes D = W X^T (out x tokens); column-major D is then the row-major [tokens, out]\n"
       "// tensor the model consumes, so the output needs no transpose. Mainloop and dispatch unchanged.\n"
       "#if defined(MIXFP4_D_COLMAJOR) && MIXFP4_D_COLMAJOR\n"
       "using LayoutCTag = cutlass::layout::ColumnMajor;\nusing LayoutDTag = cutlass::layout::ColumnMajor;\n"
       "#else\n" + old + "#endif\n")
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
fi
if [ ! -f "$RQ/cmake_cfg/3rdparty/cutlass/include/cutlass/version_extended.h" ]; then
  "$E/bin/cmake" -S "$SNAP" -B "$RQ/cmake_cfg" -DCMAKE_CUDA_COMPILER="$E/bin/nvcc" \
    -DCMAKE_CXX_COMPILER="$E/bin/x86_64-conda-linux-gnu-g++" -DCMAKE_C_COMPILER="$E/bin/x86_64-conda-linux-gnu-gcc" \
    -DCUTLASS_NVCC_ARCHS=120a -DCMAKE_CUDA_ARCHITECTURES=120a > "$RQ/cmake_cfg.log" 2>&1
fi
if [[ "$CFG" == *_nodisp ]]; then
  # JOINT_KA defaults MIXFP4_PIPE_FLAGS=1, whose pipelined loop calls dispatch_pattern directly
  # and bypasses NO_DISPATCH; the generic dispatch lambda (PIPE_FLAGS=0) honours it.
  EXTRA="$EXTRA -DMIXFP4_NO_DISPATCH=1 -DMIXFP4_PIPE_FLAGS=0"; PATCH=0
  [ "$TARGET" = lib ] || { echo "$CFG is a library-only latency build" >&2; exit 1; }
fi

[ -n "$GEN" ] && env $GEN python3 "$SNAP/scripts/gen_mixed_mma_blob.py"

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
  nvcc $EXTRA $SOURCE_DEF "${COMMON[@]}" -Xcompiler=-fPIC -Xcompiler=-fopenmp -shared -Dmain=mixfp4_selftest_main \
    -I"$SNAP/src" "$HERE/realquant_gemm.cu" -o "$OUT.unpatched" -lcudart_static -lrt -lpthread -ldl -lgomp
fi
if [ "$PATCH" = 1 ]; then
  python3 "$SNAP/scripts/patch_mixed_nvfp4_gemm.py" --cuobjdump "$E/bin/cuobjdump" --allow-missing-sites \
    "$OUT.unpatched" "$OUT"
else
  cp "$OUT.unpatched" "$OUT"   # no E0M3 site to install
fi
echo "built $OUT"

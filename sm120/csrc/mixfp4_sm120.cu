// C ABI around the SM120 mixed E2M1/E0M3 block-scaled GEMM, for the NVFP4-RaZeR native path.
//
// The GEMM is the vendored mixfp4 kernel (sm120/kernel/src/mixed_nvfp4_gemm.cu, included verbatim
// with its self-test driver compiled out), built for one configuration per shared library by
// sm120/build.py. This file adds only:
//   * the fused epilogue  D[m, n] = bf16( (s_m[m] * s_n[n]) * acc[m, n] + bias )
//     where s_m / s_n are FP32 vectors along M / N (a null pointer broadcasts a per-call scalar),
//     and bias is a bf16 vector along the weight operand's output-channel axis (null = 0);
//   * the kernel's scale-factor layout as a flat offset table, so Python can place scale bytes;
//   * the kernel's format-granule map, derived from its own TiledMma;
//   * a JSON description of the compiled configuration, checked by build.py and at load time.
//
// Configurations (all set by build.py, see sm120/mixfp4_sm120/configs.py):
//   SM120_STOCK=1        stock CUTLASS SM120 NVFP4 mainloop (E2M1 only, no format dispatch), the
//                        baseline; same tile, same epilogue as the mixed builds.
//   SM120_BIAS_ON_N=1    bias is indexed by N (weights on operand B); default M (weights on A).
//   MIXFP4_*             forwarded to the vendored kernel (granule, arrangement, D layout, ...).
//
// Every mixed build must be run through sm120/kernel/scripts/patch_mixed_nvfp4_gemm.py before use:
// PTX cannot spell E0M3, so unpatched the E0M3 dispatch sites silently compute E2M1.

#include <cstdint>
#include <cstdio>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/version.h"
#include "cutlass/numeric_conversion.h"
// The collective builder pulls in the SM90/SM120 visitor (EVT) nodes in their required order.
#include "cutlass/epilogue/collective/collective_builder.hpp"

// ------------------------------------------------------------------------------------------------
// Fused epilogue (an EVT the SM120 TMA epilogue builder accepts in place of a fusion operation)
// ------------------------------------------------------------------------------------------------
namespace sm120_epi {
using namespace cute;
namespace fu = cutlass::epilogue::fusion;

// Alignment 1 for every broadcast: N is the token count, which is arbitrary.
template <class CtaTile>
using ScaleM = fu::Sm90ColBroadcast<0, CtaTile, float, float, Stride<_1, _0, int64_t>, 1>;
template <class CtaTile>
using ScaleN = fu::Sm90RowBroadcast<0, CtaTile, float, float, Stride<_0, _1, int64_t>, 1>;
template <class CtaTile>
using BiasM = fu::Sm90ColBroadcast<0, CtaTile, cutlass::bfloat16_t, float, Stride<_1, _0, int64_t>, 1>;
template <class CtaTile>
using BiasN = fu::Sm90RowBroadcast<0, CtaTile, cutlass::bfloat16_t, float, Stride<_0, _1, int64_t>, 1>;

template <class CtaTile>
using ScaleMN = fu::Sm90EVT<
    fu::Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
    ScaleM<CtaTile>, ScaleN<CtaTile>>;

#if defined(SM120_BIAS_ON_N) && SM120_BIAS_ON_N
template <class CtaTile>
using Bias = BiasN<CtaTile>;
#else
template <class CtaTile>
using Bias = BiasM<CtaTile>;
#endif

// D = (s_m * s_n) * acc + bias, one FP32 fma, one rounding to bf16.
template <class CtaTile>
using Fusion = fu::Sm90EVT<
    fu::Sm90Compute<cutlass::homogeneous_multiply_add, cutlass::bfloat16_t, float,
                    cutlass::FloatRoundStyle::round_to_nearest>,
    ScaleMN<CtaTile>, fu::Sm90AccFetch, Bias<CtaTile>>;
}  // namespace sm120_epi

#if defined(SM120_STOCK) && SM120_STOCK
// ------------------------------------------------------------------------------------------------
// Stock CUTLASS SM120 NVFP4 GEMM: same element types, tile and epilogue as the mixed kernel.
// ------------------------------------------------------------------------------------------------
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
#if defined(MIXFP4_D_COLMAJOR) && MIXFP4_D_COLMAJOR
using LayoutCTag = cutlass::layout::ColumnMajor;
using LayoutDTag = cutlass::layout::ColumnMajor;
#else
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
#endif
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    sm120_epi::Fusion<ThreadBlockShape>>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

#else
// ------------------------------------------------------------------------------------------------
// The vendored mixed kernel, verbatim, with our epilogue substituted through its fusion hook.
// ------------------------------------------------------------------------------------------------
#define MIXFP4_EPILOGUE_FUSION_T sm120_epi::Fusion
#ifndef MIXFP4_NO_SELFTEST
#define MIXFP4_NO_SELFTEST 1
#endif
#include "mixed_nvfp4_gemm.cu"
#endif

namespace {

using SfConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using DataA = typename ElementA::DataType;
using DataB = typename ElementB::DataType;
using SF = typename ElementA::ScaleFactorType;

template <class LayoutSF>
int64_t sf_cosize(LayoutSF const &layout) {
  return int64_t(size(filter_zeros(layout)));
}

struct Call {
  void const *a, *sfa, *b, *sfb;
  void *d;
  int m, n, k;
  float const *scale_m, *scale_n;
  float scale_m_default, scale_n_default;
  void const *bias;
};

typename Gemm::Arguments make_arguments(Call const &c) {
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {c.m, c.k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {c.n, c.k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {c.m, c.n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {c.m, c.n, 1});
  auto shape = make_shape(c.m, c.n, c.k, 1);
  LayoutSFA layout_sfa = SfConfig::tile_atom_to_shape_SFA(shape);
  LayoutSFB layout_sfb = SfConfig::tile_atom_to_shape_SFB(shape);

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {c.m, c.n, c.k, 1},
      {static_cast<DataA const *>(c.a), stride_a, static_cast<DataB const *>(c.b), stride_b,
       static_cast<SF const *>(c.sfa), layout_sfa, static_cast<SF const *>(c.sfb), layout_sfb},
      {{},  // fusion arguments, filled below
       // The EVT never reads C (no source node), but the epilogue's TMA descriptor for C is built
       // regardless, so it is given D's valid address.
       static_cast<ElementC const *>(c.d), stride_c, static_cast<ElementD *>(c.d), stride_d}};

  auto &fusion = args.epilogue.thread;
  // An EVT node's arguments are its children's, then its own: fusion = {op_0 ScaleMN, op_1 AccFetch,
  // op_2 Bias, op_3 fma}; ScaleMN = {op_0 ScaleM, op_1 ScaleN, op_2 mul}.
  auto &scale_mn = fusion.op_0;
  auto &scale_m = scale_mn.op_0;
  auto &scale_n = scale_mn.op_1;
  scale_m.ptr_col = c.scale_m;
  scale_m.null_default = c.scale_m_default;
  scale_m.dCol = {};
  scale_n.ptr_row = c.scale_n;
  scale_n.null_default = c.scale_n_default;
  scale_n.dRow = {};
  auto &bias = fusion.op_2;
#if defined(SM120_BIAS_ON_N) && SM120_BIAS_ON_N
  bias.ptr_row = static_cast<cutlass::bfloat16_t const *>(c.bias);
  bias.dRow = {};
#else
  bias.ptr_col = static_cast<cutlass::bfloat16_t const *>(c.bias);
  bias.dCol = {};
#endif
  bias.null_default = cutlass::bfloat16_t(0.0f);
  return args;
}

#ifndef SM120_CONFIG_NAME
#define SM120_CONFIG_NAME "unnamed"
#endif
#define SM120_STR2(x) #x
#define SM120_STR(x) SM120_STR2(x)

}  // namespace

extern "C" {

// Configuration this library was compiled for, as a JSON object. Returns the full length; writes
// at most `capacity` bytes (NUL-terminated when it fits).
int sm120_describe(char *out, int capacity) {
  int a_rows = 0, a_k = 0, b_cols = 0, b_k = 0, pin_a = 1, pin_b = 1, stock = 1;
  int blob_mma_m = 0, blob_mma_n = 0;
#if !(defined(SM120_STOCK) && SM120_STOCK)
  stock = 0;
  a_rows = MIXFP4_A_ATOMS_PER_GRANULE * 16;
  b_cols = MIXFP4_B_ATOMS_PER_GRANULE * 8;
#if defined(MIXFP4_K_GRANULE_A)
  a_k = MIXFP4_K_GRANULE_A;
#elif defined(MIXFP4_K_GRANULE)
  a_k = MIXFP4_K_GRANULE;
#else
  a_k = int(size<2>(ThreadBlockShape{}));
#endif
#if defined(MIXFP4_K_GRANULE_B)
  b_k = MIXFP4_K_GRANULE_B;
#elif defined(MIXFP4_K_GRANULE)
  b_k = MIXFP4_K_GRANULE;
#else
  b_k = int(size<2>(ThreadBlockShape{}));
#endif
#if defined(MIXFP4_A_ALL_E2M1) && MIXFP4_A_ALL_E2M1
  pin_a = 1;
#else
  pin_a = 0;
#endif
#if defined(MIXFP4_B_ALL_E2M1) && MIXFP4_B_ALL_E2M1
  pin_b = 1;
#else
  pin_b = 0;
#endif
#if defined(MIXFP4_BLOBGEN_MMA_M)
  blob_mma_m = MIXFP4_BLOBGEN_MMA_M;
  blob_mma_n = MIXFP4_BLOBGEN_MMA_N;
#endif
#endif
  int d_colmajor = 0;
#if defined(MIXFP4_D_COLMAJOR) && MIXFP4_D_COLMAJOR
  d_colmajor = 1;
#endif
  int bias_on_n = 0;
#if defined(SM120_BIAS_ON_N) && SM120_BIAS_ON_N
  bias_on_n = 1;
#endif
  using TM = typename Gemm::GemmKernel::CollectiveMainloop::TiledMma;
  int const atom_m = int(size<1>(typename TM::ThrLayoutVMNK{}));
  int const atom_n = int(size<2>(typename TM::ThrLayoutVMNK{}));
  int const stages = int(Gemm::GemmKernel::CollectiveMainloop::DispatchPolicy::Stages);
  int const smem = int(sizeof(typename GemmKernel::SharedStorage));
  char buf[1024];
  int len = std::snprintf(
      buf, sizeof(buf),
      "{\"config\":\"%s\",\"stock\":%d,\"tile_mnk\":[%d,%d,%d],\"warp_atoms_mn\":[%d,%d],"
      "\"granule_a\":[%d,%d],\"granule_b\":[%d,%d],\"pinned_e2m1\":[%d,%d],"
      "\"blobgen_mma_mn\":[%d,%d],\"d_colmajor\":%d,\"bias_on_n\":%d,\"mainloop_stages\":%d,"
      "\"shared_storage_bytes\":%d,\"cutlass_version\":\"%d.%d.%d\",\"epilogue\":\"bf16((s_m*s_n)*acc+bias)\"}",
      SM120_CONFIG_NAME, stock, int(size<0>(ThreadBlockShape{})), int(size<1>(ThreadBlockShape{})),
      int(size<2>(ThreadBlockShape{})), atom_m, atom_n, a_rows, a_k, b_cols, b_k, pin_a, pin_b,
      blob_mma_m, blob_mma_n, d_colmajor, bias_on_n, stages, smem, CUTLASS_MAJOR, CUTLASS_MINOR,
      CUTLASS_PATCH);
  if (out != nullptr && capacity > 0) {
    int const n = len < capacity - 1 ? len : capacity - 1;
    for (int i = 0; i < n; ++i) { out[i] = buf[i]; }
    out[n] = '\0';
  }
  return len;
}

// Number of scale-factor bytes the kernel reads for operand A (operand 0, rows = m) or B
// (operand 1, rows = n), including the layout's padding.
int64_t sm120_sf_size(int operand, int m, int n, int k) {
  auto shape = make_shape(m, n, k, 1);
  if (operand == 0) { return sf_cosize(SfConfig::tile_atom_to_shape_SFA(shape)); }
  return sf_cosize(SfConfig::tile_atom_to_shape_SFB(shape));
}

// out[r * (k / 16) + kb] = byte offset of the scale factor of row r, K block kb.
int sm120_sf_offsets(int operand, int m, int n, int k, int64_t *out) {
  if (k % 16 != 0 || out == nullptr) { return -1; }
  auto shape = make_shape(m, n, k, 1);
  int const rows = operand == 0 ? m : n;
  int const kblocks = k / 16;
  auto fill = [&](auto layout) {
    for (int r = 0; r < rows; ++r) {
      for (int kb = 0; kb < kblocks; ++kb) {
        out[int64_t(r) * kblocks + kb] = int64_t(layout(r, kb * 16, 0));
      }
    }
  };
  if (operand == 0) { fill(SfConfig::tile_atom_to_shape_SFA(shape)); }
  else              { fill(SfConfig::tile_atom_to_shape_SFB(shape)); }
  return 0;
}

// Granule representative of every row (operand 0) or column (operand 1) of one CTA tile, as the
// kernel's own build_granule_map derives it from its TiledMma. Returns the tile extent, 0 for the
// stock kernel (no format granule), or -extent if `capacity` is too small.
int sm120_granule_map(int operand, int *out, int capacity) {
#if defined(SM120_STOCK) && SM120_STOCK
  (void) operand; (void) out; (void) capacity;
  return 0;
#else
  std::vector<int> map = operand == 0 ? build_granule_map<true>(MIXFP4_A_ATOMS_PER_GRANULE)
                                      : build_granule_map<false>(MIXFP4_B_ATOMS_PER_GRANULE);
  if (int(map.size()) > capacity) { return -int(map.size()); }
  for (size_t i = 0; i < map.size(); ++i) { out[i] = map[i]; }
  return int(map.size());
#endif
}

size_t sm120_workspace_size(int m, int n, int k) {
  Call c{};
  c.m = m; c.n = n; c.k = k;
  c.scale_m_default = c.scale_n_default = 1.0f;
  return Gemm::get_workspace_size(make_arguments(c));
}

// D[m, n] = bf16( (s_m[m] * s_n[n]) * sum_k decode(A[m, k]) decode(B[n, k]) + bias ).
// A: [m, k/2] bytes, B: [n, k/2] bytes (K-contiguous nibbles, element 2j in the low nibble);
// sfa / sfb: scale bytes in the kernel layout (sm120_sf_offsets), bit 7 = E0M3 format tag.
// D: bf16, row-major [m, n] or, for column-major-D builds, column-major (= row-major [n, m]).
// scale_m / scale_n / bias may be null: the scale then is the given default, the bias zero.
// Returns 0 on success; negative codes identify the failing step.
int sm120_gemm(void const *a, void const *sfa, void const *b, void const *sfb, void *d,
               int m, int n, int k,
               float const *scale_m, float scale_m_default,
               float const *scale_n, float scale_n_default,
               void const *bias,
               void *workspace, size_t workspace_size, void *stream) {
  Call c{a, sfa, b, sfb, d, m, n, k, scale_m, scale_n, scale_m_default, scale_n_default, bias};
  auto args = make_arguments(c);
  if (Gemm::get_workspace_size(args) > workspace_size) { return -1000; }
  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) { return -2000 - int(status); }
  status = gemm_op.initialize(args, workspace, static_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) { return -3000 - int(status); }
  status = gemm_op.run(static_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) { return -4000 - int(status); }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { return -5000 - int(err); }
  return 0;
}

}  // extern "C"

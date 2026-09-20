// Mixed E0M3/E2M1 block-scaled NVFP4 GEMM on datacenter Blackwell (sm_100a).
//
// This is the sm_100 counterpart of src/mixed_nvfp4_gemm.cu, and it is deliberately much smaller,
// because the hardware problem that dominated the sm_120 work does not exist here.
//
// On sm_120 the operand format is two bits of the *SASS opcode* of `mma.sync...block_scale`.
// Choosing it at runtime therefore meant branching between distinct instructions, ptxas
// if-converted those branches, and a predicated-off OMMA still consumed a tensor-pipe issue slot
// -- costing 2x before any of the mitigation work. PTX cannot even spell E0M3, so the binary had
// to be patched after compilation.
//
// On sm_100 the format lives in the UMMA *instruction descriptor*, which `tcgen05.mma` takes in a
// general-purpose register:
//
//     asm volatile("tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
//                  "[%0], %1, %2, %3, [%5], [%6], p;"
//                  :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
//                     "r"(uint32_t(idescE>>32)),    // <-- the format bits, in a register
//                     ...);
//                                     (cute/arch/mma_sm100_umma.hpp)
//
// So selecting E0M3 is a register write. No branch, no if-conversion, no wasted issue slot, and
// no cubin patching -- the same compiled instruction does either format. CuTe already models this:
// `UMMA::InstrDescriptorBlockScaled idesc_` is a non-static member of the MMA_Traits, and the
// mainloop already rewrites it at runtime for CUTLASS's own type-erased dtype feature.
//
// What this binary establishes, in order:
//   1. that format enum 0 on the mxf4nvf4 path is the evenly-spaced INT4/E0M3 codebook, by
//      checking the result against a host reference built on that codebook; and
//   2. what forcing it costs, against src/nvfp4_gemm_sm100.cu on the same shape.
//
// Uniform over the launch. Per-granule selection is the next step and rides on the same
// descriptor write.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "collective/dispatch_policy_sm100_mixed.hpp"
#include "collective/sm100_blockscaled_mma_mixed.hpp"

#include "helper.h"
#include "gemm_benchmark.hpp"
#include "sm100_gemm_launch.hpp"

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

#ifndef MIXFP4_TILE_M
#define MIXFP4_TILE_M 256
#endif
#ifndef MIXFP4_TILE_N
#define MIXFP4_TILE_N 256
#endif
#ifndef MIXFP4_CLUSTER_M
#define MIXFP4_CLUSTER_M 2
#endif
using MmaTileShape = Shape<Int<MIXFP4_TILE_M>, Int<MIXFP4_TILE_N>, _256>;
using ClusterShape = Shape<Int<MIXFP4_CLUSTER_M>, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    Shape<_128, Int<(MIXFP4_TILE_N == 256 ? 64 : 32)>>,
    ElementAccumulator, ElementAccumulator,
    void, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cute::conditional_t<MIXFP4_TILE_M == 256,
        cutlass::epilogue::TmaWarpSpecialized2Sm,
        cutlass::epilogue::TmaWarpSpecialized1Sm>>::CollectiveOp;

// CUTLASS's own mainloop, using the same epilogue as the stock *_fast configuration. We never launch it;
// we only harvest the twenty-odd template arguments CollectiveBuilder derived, so that the mixed
// mainloop differs from the stock one in the dispatch policy and nothing else.
using StdMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

// Swap the dispatch policy for our structurally identical tag, which is what routes the type to
// the forked mainloop specialization in collective/sm100_blockscaled_mma_mixed.hpp. Pattern
// matching on the whole parameter pack means a CUTLASS bump that adds a parameter is a compile
// error here rather than a silent divergence.
template <class T>
struct RebindToMixed;

template <int Stages, int SchedPipe, int AccumPipe, class Cluster, class Arch, class... Rest>
struct RebindToMixed<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<Stages, SchedPipe, AccumPipe,
                                                                  Cluster, Arch>,
    Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaledMixed<Stages, SchedPipe,
                                                                         AccumPipe, Cluster, Arch>,
      Rest...>;
};

using CollectiveMainloop = typename RebindToMixed<StdMainloop>::type;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

template <typename T>
auto make_iterator(T *ptr) {
  return cute::recast_ptr<T>(ptr);
}

template <typename Element, typename Layout>
void initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  constexpr int bits = cutlass::sizeof_bits<Element>::value;
  double scope_max, scope_min;
  if constexpr (bits <= 6) {
    scope_max = 6; scope_min = -6;
  } else if constexpr (cute::is_same_v<Element, cutlass::float_ue4m3_t> ||
                       cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
    scope_max = 4; scope_min = 1;
  } else {
    scope_max = 4; scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, bits <= 6 ? 1 : 0);
}

// E2M1 and E0M3 index the *same* 4-bit nibble through different codebooks:
//   E2M1 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6
//   E0M3 magnitudes: 0, 1,   2, 3,   4, 5, 6, 7   (evenly spaced signed integers, i.e. INT4)
// both with a sign bit. Recovering the nibble from an E2M1-decoded value and re-reading it under
// the E0M3 codebook therefore reproduces what an E0M3 instruction computes, without needing raw
// sub-byte access. (Same construction as the sm_120 driver, where it is checked against hardware.)
static constexpr float kE2m1Magnitudes[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static float reinterpret_e2m1_as_e0m3(float v) {
  float const mag = std::fabs(v);
  int nibble = 0;
  for (int i = 0; i < 8; ++i) {
    if (mag == kE2m1Magnitudes[i]) { nibble = i; break; }
  }
  return std::signbit(v) ? -float(nibble) : float(nibble);
}

static float decode_scale(uint8_t raw) {
  cutlass::float_ue4m3_t s{};
  s.raw() = uint8_t(raw & 0x7fu);
  return float(s);
}

static uint8_t parse_fmt(char const *s, uint8_t dflt) {
  if (s == nullptr) { return dflt; }
  std::string v = s;
  if (v == "e2m1" || v == "1") { return cutlass::gemm::collective::MixedFmt::kE2M1; }
  if (v == "e0m3" || v == "int4" || v == "0") { return cutlass::gemm::collective::MixedFmt::kE0M3; }
  std::cerr << "unknown format '" << v << "' (want e2m1 or e0m3)" << std::endl;
  std::exit(2);
}

static char const *fmt_name(uint8_t f) {
  return f == cutlass::gemm::collective::MixedFmt::kE0M3 ? "E0M3" : "E2M1";
}

int run(int m, int n, int k, uint8_t fmt_a, uint8_t fmt_b, int warmup_iters, int bench_iters,
        bool check, int raster, int swizzle) {
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_a = make_layout(make_shape(m, k, 1), stride_a);
  auto layout_b = make_layout(make_shape(n, k, 1), stride_b);
  auto layout_c = make_layout(make_shape(m, n, 1), stride_c);
  auto layout_d = make_layout(make_shape(m, n, 1), stride_d);
  LayoutSFA layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_a;
  cutlass::HostTensor<typename ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_sfa;
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_b;
  cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_sfb;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_c;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_d;

  block_a.reset(cutlass::make_Coord(size(layout_a)));
  block_b.reset(cutlass::make_Coord(size(layout_b)));
  block_c.reset(cutlass::make_Coord(size(layout_c)));
  block_d.reset(cutlass::make_Coord(size(layout_d)));
  block_sfa.reset(cutlass::make_Coord(size(filter_zeros(layout_sfa))));
  block_sfb.reset(cutlass::make_Coord(size(filter_zeros(layout_sfb))));

  if (check) {
    initialize_block(block_a.host_view(), 2021);
    initialize_block(block_b.host_view(), 2022);
    initialize_block(block_c.host_view(), 2023);
    initialize_block(block_sfa.host_view(), 2024);
    initialize_block(block_sfb.host_view(), 2025);

    block_a.sync_device();
    block_b.sync_device();
    block_c.sync_device();
    block_sfa.sync_device();
    block_sfb.sync_device();
  } else {
    CUDA_CHECK(cudaMemset(block_a.device_data(), 0x22, size_t(m) * k / 2));
    CUDA_CHECK(cudaMemset(block_b.device_data(), 0x22, size_t(n) * k / 2));
    CUDA_CHECK(cudaMemset(block_c.device_data(), 0, size_t(m) * n * sizeof(ElementC)));
    CUDA_CHECK(cudaMemset(block_sfa.device_data(), 0x3c, block_sfa.capacity()));
    CUDA_CHECK(cudaMemset(block_sfb.device_data(), 0x3c, block_sfb.capacity()));
  }

  float alpha = 1.0f, beta = 0.0f;

  typename Gemm::GemmKernel::MainloopArguments mainloop_args{};
  mainloop_args.ptr_A = block_a.device_data();
  mainloop_args.dA = stride_a;
  mainloop_args.ptr_B = block_b.device_data();
  mainloop_args.dB = stride_b;
  mainloop_args.ptr_SFA = block_sfa.device_data();
  mainloop_args.layout_SFA = layout_sfa;
  mainloop_args.ptr_SFB = block_sfb.device_data();
  mainloop_args.layout_SFB = layout_sfb;
  mainloop_args.fmt_a = fmt_a;
  mainloop_args.fmt_b = fmt_b;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      mainloop_args,
      {{alpha, beta}, block_c.device_data(), stride_c, block_d.device_data(), stride_d}};

  using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;
  arguments.scheduler.raster_order = raster == 1 ? Raster::AlongM
                                    : raster == 2 ? Raster::AlongN : Raster::Heuristic;
  arguments.scheduler.max_swizzle_size = swizzle;

  Gemm gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  mixfp4::Sm100GemmLaunch<Gemm> launch(gemm_op);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  if (check) {
    // Decode both operands to scaled floats under the format the kernel was told to use, then do
    // a plain float GEMM. Decoding up front keeps the O(M*N*K) inner loop free of sub-byte and
    // scale-factor address arithmetic.
    auto tensor_a = make_tensor(make_iterator(block_a.host_data()), layout_a);
    auto tensor_sfa = make_tensor(block_sfa.host_data(), layout_sfa);
    auto tensor_b = make_tensor(make_iterator(block_b.host_data()), layout_b);
    auto tensor_sfb = make_tensor(block_sfb.host_data(), layout_sfb);
    auto tensor_c = make_tensor(make_iterator(block_c.host_data()), layout_c);

    std::vector<float> a_dec(size_t(m) * size_t(k));
    std::vector<float> b_dec(size_t(n) * size_t(k));
    // Negative control. "All four combinations PASSED" only means something if a WRONG codebook
    // makes the check fail -- otherwise the test could be passing because neither the kernel nor
    // the reference is honouring the format at all. MIXFP4_REF_FMT_A/B decouple the reference's
    // codebook from the kernel's so a deliberate mismatch can be demanded to fail.
    uint8_t const ref_fmt_a = parse_fmt(std::getenv("MIXFP4_REF_FMT_A"), fmt_a);
    uint8_t const ref_fmt_b = parse_fmt(std::getenv("MIXFP4_REF_FMT_B"), fmt_b);
    if (ref_fmt_a != fmt_a || ref_fmt_b != fmt_b) {
      std::printf("  [negative control: reference decodes A=%s B=%s while the kernel was told "
                  "A=%s B=%s -- this SHOULD fail]\n",
                  fmt_name(ref_fmt_a), fmt_name(ref_fmt_b), fmt_name(fmt_a), fmt_name(fmt_b));
    }
    bool const a_is_e0m3 = (ref_fmt_a == cutlass::gemm::collective::MixedFmt::kE0M3);
    bool const b_is_e0m3 = (ref_fmt_b == cutlass::gemm::collective::MixedFmt::kE0M3);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
      for (int kk = 0; kk < k; ++kk) {
        uint8_t const sf = tensor_sfa(i, kk, 0).raw();
        float v = float(static_cast<cutlass::float_e2m1_t>(tensor_a(i, kk, 0)));
        if (a_is_e0m3) { v = reinterpret_e2m1_as_e0m3(v); }
        a_dec[size_t(i) * size_t(k) + size_t(kk)] = v * decode_scale(sf);
      }
    }
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j) {
      for (int kk = 0; kk < k; ++kk) {
        uint8_t const sf = tensor_sfb(j, kk, 0).raw();
        float v = float(static_cast<cutlass::float_e2m1_t>(tensor_b(j, kk, 0)));
        if (b_is_e0m3) { v = reinterpret_e2m1_as_e0m3(v); }
        b_dec[size_t(j) * size_t(k) + size_t(kk)] = v * decode_scale(sf);
      }
    }

    block_d.sync_host();
    auto tensor_d = make_tensor(make_iterator(block_d.host_data()), layout_d);

    // Relative Frobenius norm, not element-wise relative error: the GPU sums K in a different
    // order and rounds to bfloat16, so near-cancelling elements can show a large relative error
    // while the result is right. The norm ratio is insensitive to that but still moves by order 1
    // if an operand decodes under the wrong codebook.
    double num = 0.0, den = 0.0, max_abs_err = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : num, den) reduction(max : max_abs_err)
    for (int i = 0; i < m; ++i) {
      float const *ap = &a_dec[size_t(i) * size_t(k)];
      for (int j = 0; j < n; ++j) {
        float const *bp = &b_dec[size_t(j) * size_t(k)];
        float acc = 0.f;
        for (int kk = 0; kk < k; ++kk) { acc += ap[kk] * bp[kk]; }
        float const ref = alpha * acc +
                          beta * float(static_cast<cutlass::bfloat16_t>(tensor_c(i, j, 0)));
        float const got = float(static_cast<cutlass::bfloat16_t>(tensor_d(i, j, 0)));
        double const d = double(got) - double(ref);
        num += d * d;
        den += double(ref) * double(ref);
        max_abs_err = std::max(max_abs_err, std::fabs(d));
      }
    }
    double const rel = (den > 0.0) ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
    bool const passed = (rel < 2e-2);
    std::printf("Correctness (A=%s B=%s): %s  rel_fro=%.3e  max_abs=%.3e\n",
                fmt_name(fmt_a), fmt_name(fmt_b), passed ? "PASSED" : "FAILED", rel, max_abs_err);
    if (!passed) { return -1; }
  }

  std::vector<ElementD> checked_output;
  if (check) {
    checked_output.assign(block_d.host_data(), block_d.host_data() + size(layout_d));
  }
  double ms = mixfp4::benchmark_ms(
      [&](cudaStream_t stream) {
        launch(stream);
      },
      warmup_iters, bench_iters);
  if (check) {
    block_d.sync_host();
    if (!std::equal(checked_output.begin(), checked_output.end(), block_d.host_data())) {
      std::fprintf(stderr, "Correctness: FAILED after repeated launches\n");
      return 1;
    }
  }

  double const tflops =
      2.0 * double(m) * double(n) * double(k) / (ms / 1.0e3) / 1.0e12;

  std::printf("Problem size: %dx%dx%d  (tile %dx%dx256, cluster %dx1x1)\n", m, n, k,
              MIXFP4_TILE_M, MIXFP4_TILE_N, MIXFP4_CLUSTER_M);
  std::printf("Formats: A=%s B=%s\n", fmt_name(fmt_a), fmt_name(fmt_b));
  std::printf("Scheduler: raster=%d swizzle=%d; timing=%s; PDL=%s\n", raster, swizzle,
              mixfp4::benchmark_uses_graph() ? "graph" : "launch",
              mixfp4::sm100_uses_pdl() ? "on" : "off");
  std::printf("Avg latency: %f ms\n", ms);
  std::printf("Throughput: %.1f TFLOP/s (mixed sm100)\n", tflops);
  return 0;
}

int main(int argc, char const **argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major != 10) {
    std::cerr << "This kernel targets datacenter Blackwell (sm_100); detected compute capability "
              << props.major << "." << props.minor << std::endl;
    return 0;
  }

  int m = 4096, n = 4096, k = 4096;
  int warmup_iters = 10, bench_iters = 50;
  int raster = 0, swizzle = -1;
  bool check = (std::getenv("MIXFP4_SKIP_REF") == nullptr);
  uint8_t fmt_a = parse_fmt(std::getenv("MIXFP4_FMT_A"), cutlass::gemm::collective::MixedFmt::kE2M1);
  uint8_t fmt_b = parse_fmt(std::getenv("MIXFP4_FMT_B"), cutlass::gemm::collective::MixedFmt::kE2M1);

  std::vector<char const *> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--fmt-a=", 0) == 0) { fmt_a = parse_fmt(a.c_str() + 8, fmt_a); }
    else if (a.rfind("--fmt-b=", 0) == 0) { fmt_b = parse_fmt(a.c_str() + 8, fmt_b); }
    else if (a == "--no-check") { check = false; }
    else if (a == "--check") { check = true; }
    else if (a.rfind("--iters=", 0) == 0) { bench_iters = std::atoi(a.c_str() + 8); }
    else if (a.rfind("--raster=", 0) == 0) { raster = std::atoi(a.c_str() + 9); }
    else if (a.rfind("--swizzle=", 0) == 0) { swizzle = std::atoi(a.c_str() + 10); }
    else if (a.rfind("--", 0) == 0) {
      std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
      return 2;
    }
    else { pos.push_back(argv[i]); }
  }
  if (pos.size() >= 3) { m = std::atoi(pos[0]); n = std::atoi(pos[1]); k = std::atoi(pos[2]); }

  if ((!pos.empty() && pos.size() != 3) || m <= 0 || n <= 0 || k <= 0 || bench_iters <= 0 ||
      raster < 0 || raster > 2 || swizzle < -1 || swizzle > 8 ||
      (swizzle > 0 && (swizzle & (swizzle - 1)))) {
    std::fprintf(stderr, "Invalid dimensions, iteration count or scheduler options.\n");
    return 2;
  }
  if (swizzle < 0) { swizzle = (m >= 8192 || n >= 8192 || k >= 8192) ? 2 : 0; }
  return run(m, n, k, fmt_a, fmt_b, warmup_iters, bench_iters, check, raster, swizzle);
}

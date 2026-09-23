// ctypes-callable wrapper around the mixfp4 SM120 mixed E2M1/E0M3 block-scaled GEMM.
//
// The kernel, its tile shape, warp arrangement, granule configuration and scale-factor layout are
// taken verbatim from /home/dev/mixfp4/src/mixed_nvfp4_gemm.cu by including that file (its main()
// is renamed at compile time with -Dmain=mixfp4_selftest_main). Nothing about the GEMM is
// re-derived here: this file only exposes
//   * the kernel's own scale-factor layout, as a flat offset table, so Python can place scale bytes;
//   * the granule map (which rows/columns of a CTA tile share one format flag);
//   * one GEMM launch  D[m, n] = alpha * sum_k decode(A[m, k]) * decode(B[n, k])   (bf16 output),
//     with A and B packed K-contiguous e2m1/e0m3 nibbles and the format flag in bit 7 of each
//     UE4M3 scale byte, exactly as the self-test executable feeds it.
// The resulting shared library must be run through scripts/patch_mixed_nvfp4_gemm.py, like the
// executable, or every E0M3 site silently computes E2M1.

// -DRQ_STOCK=1 wraps src/nvfp4_gemm.cu instead: the stock CUTLASS SM120 NVFP4 GEMM (E2M1 only, no
// format dispatch), exposed through the same entry points for latency comparisons.
#if defined(RQ_STOCK) && RQ_STOCK
#include "nvfp4_gemm.cu"
#else
#include "mixed_nvfp4_gemm.cu"
#endif

namespace {

using SfConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

template <class LayoutSF>
int64_t sf_cosize(LayoutSF const &layout) {
  return int64_t(size(filter_zeros(layout)));
}

}  // namespace

extern "C" {

// Number of scale-factor bytes the kernel reads for operand A (operand == 0, rows = m) or
// B (operand == 1, rows = n), including the layout's padding.
int64_t rq_sf_size(int operand, int m, int n, int k) {
  auto shape = make_shape(m, n, k, 1);
  if (operand == 0) { return sf_cosize(SfConfig::tile_atom_to_shape_SFA(shape)); }
  return sf_cosize(SfConfig::tile_atom_to_shape_SFB(shape));
}

// out[r * (k / 16) + kb] = byte offset of the scale factor of row r, K block kb.
int rq_sf_offsets(int operand, int m, int n, int k, int64_t *out) {
  if (k % 16 != 0) { return -1; }
  auto shape = make_shape(m, n, k, 1);
  int const rows = operand == 0 ? m : n;
  int const kblocks = k / 16;
  auto fill = [&](auto layout) {
#pragma omp parallel for
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

#if defined(RQ_STOCK) && RQ_STOCK
// The stock kernel has no format granule and decodes both operands as E2M1.
int rq_granule_map(int, int *, int) { return 0; }
void rq_granule_shape(int *out4) { out4[0] = out4[1] = out4[2] = out4[3] = 0; }
void rq_pinned(int *out2) { out2[0] = out2[1] = 1; }
#else
// Granule representative of every row (operand 0) or column (operand 1) of one CTA tile, as the
// kernel's own build_granule_map derives it from the TiledMma. Returns the tile extent.
int rq_granule_map(int operand, int *out, int capacity) {
  std::vector<int> map = operand == 0 ? build_granule_map<true>(MIXFP4_A_ATOMS_PER_GRANULE)
                                      : build_granule_map<false>(MIXFP4_B_ATOMS_PER_GRANULE);
  if (int(map.size()) > capacity) { return -int(map.size()); }
  for (size_t i = 0; i < map.size(); ++i) { out[i] = map[i]; }
  return int(map.size());
}

// Granule geometry this library was compiled for: rows of A, K of A, cols of B, K of B.
void rq_granule_shape(int *out4) {
  out4[0] = MIXFP4_A_ATOMS_PER_GRANULE * 16;
  out4[2] = MIXFP4_B_ATOMS_PER_GRANULE * 8;
#if defined(MIXFP4_K_GRANULE_A)
  out4[1] = MIXFP4_K_GRANULE_A;
#else
  out4[1] = int(size<2>(ThreadBlockShape{}));
#endif
#if defined(MIXFP4_K_GRANULE_B)
  out4[3] = MIXFP4_K_GRANULE_B;
#else
  out4[3] = int(size<2>(ThreadBlockShape{}));
#endif
}

// 1 if operand A (resp. B) is compiled pinned to E2M1, i.e. its flags must never be set.
void rq_pinned(int *out2) {
#if defined(MIXFP4_A_ALL_E2M1) && MIXFP4_A_ALL_E2M1
  out2[0] = 1;
#else
  out2[0] = 0;
#endif
#if defined(MIXFP4_B_ALL_E2M1) && MIXFP4_B_ALL_E2M1
  out2[1] = 1;
#else
  out2[1] = 0;
#endif
}
#endif  // RQ_STOCK

static typename Gemm::Arguments make_arguments(void const *a, void const *sfa, void const *b,
                                               void const *sfb, void *d, int m, int n, int k,
                                               float alpha) {
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  auto shape = make_shape(m, n, k, 1);
  LayoutSFA layout_sfa = SfConfig::tile_atom_to_shape_SFA(shape);
  LayoutSFB layout_sfb = SfConfig::tile_atom_to_shape_SFB(shape);
  using DataA = typename ElementA::DataType;
  using DataB = typename ElementB::DataType;
  using SF = typename ElementA::ScaleFactorType;
  // beta = 0: C is never read, but it is given D's (valid) address so the epilogue's source
  // descriptor is well formed.
  return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {static_cast<DataA const *>(a), stride_a, static_cast<DataB const *>(b), stride_b,
       static_cast<SF const *>(sfa), layout_sfa, static_cast<SF const *>(sfb), layout_sfb},
      {{alpha, 0.0f}, static_cast<ElementC const *>(d), stride_c, static_cast<ElementD *>(d),
       stride_d}};
}

size_t rq_workspace_size(int m, int n, int k) {
  auto args = make_arguments(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, 1.0f);
  return Gemm::get_workspace_size(args);
}

// Returns 0 on success; a negative cutlass::Status / cudaError code otherwise.
int rq_gemm(void const *a, void const *sfa, void const *b, void const *sfb, void *d,
            int m, int n, int k, float alpha, void *workspace, size_t workspace_size,
            void *stream) {
  auto args = make_arguments(a, sfa, b, sfb, d, m, n, k, alpha);
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

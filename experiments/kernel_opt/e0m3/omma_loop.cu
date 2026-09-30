// E0M3 investigation, test A: a register-resident OMMA loop, one kernel per format site.
//
// Each kernel issues the same instruction stream: per loop iteration, U independent
//   prmt.b32 sf, sfa, sfa, SEL(site);  mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 ... {sf} ...
// pairs, on operands loaded once into registers (no shared memory, no dispatch, no branches in the loop body).
// PTX cannot spell E0M3, so every mma.sync compiles as E2M1 x E2M1; the kernel's patcher
// (sm120/kernel/scripts/patch_mixed_nvfp4_gemm.py) then sets the OMMA format bits of each kernel from its PRMT
// selector: site 0 -> E2M1 x E2M1 (left native), 1 -> E0M3 x E2M1, 2 -> E2M1 x E0M3, 3 -> E0M3 x E0M3.
// So the four kernels differ only in those format bits (checked by the patcher's census).
//
// Built once per site (-DOMMA_SITE=0..3): the patcher expects a single kernel per binary, and four textually identical
// kernels in one library would give it four equally valid text offsets.
// C ABI (ctypes): omma_loop(grid, block, iters, a, b, sf, out, stream) -> cudaError_t; omma_site() -> OMMA_SITE.
//   a: uint32 [grid*block*4]   A fragments (4 regs per thread)
//   b: uint32 [grid*block*2]   B fragments (2 regs per thread)
//   sf: uint32 [grid*block*2]  SFA / SFB words (4 ue4m3 scales each)
//   out: float [grid*block]    a checksum per thread (keeps the loop live)
#include <cstdint>
#include <cuda_runtime.h>

#ifndef OMMA_U
#define OMMA_U 8  // independent accumulator chains per thread
#endif

template <int SITE>
struct Sel;
template <> struct Sel<0> { static constexpr uint32_t v = 0x3210; };
template <> struct Sel<1> { static constexpr uint32_t v = 0x3214; };
template <> struct Sel<2> { static constexpr uint32_t v = 0x3254; };
template <> struct Sel<3> { static constexpr uint32_t v = 0x3654; };

template <int SITE>
__global__ void __launch_bounds__(256) omma_kernel(int iters, uint32_t const* __restrict__ a, uint32_t const* __restrict__ b,
                                                   uint32_t const* __restrict__ sf, float* __restrict__ out) {
  int const t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a0 = a[4 * t], a1 = a[4 * t + 1], a2 = a[4 * t + 2], a3 = a[4 * t + 3];
  uint32_t b0 = b[2 * t], b1 = b[2 * t + 1];
  uint32_t sfa = sf[2 * t], sfb = sf[2 * t + 1];
  float acc[OMMA_U][4];
#pragma unroll
  for (int u = 0; u < OMMA_U; ++u) {
#pragma unroll
    for (int v = 0; v < 4; ++v) { acc[u][v] = 0.0f; }
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < OMMA_U; ++u) {
      asm volatile(
          "{\n"
          "  .reg .b32 sfs;\n"
          "  prmt.b32 sfs, %8, %8, %10;\n"
          "  mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%11,%12}, {%0,%1,%2,%3}, {sfs}, {0, 0}, {%9}, {0, 0};\n"
          "}\n"
          : "+f"(acc[u][0]), "+f"(acc[u][1]), "+f"(acc[u][2]), "+f"(acc[u][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(sfa), "r"(sfb), "n"(Sel<SITE>::v), "r"(b0), "r"(b1));
    }
  }
  float s = 0.0f;
#pragma unroll
  for (int u = 0; u < OMMA_U; ++u) {
#pragma unroll
    for (int v = 0; v < 4; ++v) { s += acc[u][v]; }
  }
  out[t] = s;
}

#ifndef OMMA_SITE
#error "build with -DOMMA_SITE=0..3"
#endif

extern "C" int omma_loop(int grid, int block, int iters, void const* a, void const* b, void const* sf, void* out, void* stream) {
  omma_kernel<OMMA_SITE><<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      iters, static_cast<uint32_t const*>(a), static_cast<uint32_t const*>(b), static_cast<uint32_t const*>(sf),
      static_cast<float*>(out));
  return int(cudaGetLastError());
}

extern "C" int omma_site() { return OMMA_SITE; }

extern "C" int omma_per_iter() { return OMMA_U; }

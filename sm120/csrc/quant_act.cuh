// Per-token activation quantization + packing into the SM1xx block-scaled layout (CUDA C++).
//
// Same arithmetic as mixfp4_sm120/quant_act.py (Triton), which is bit-identical to the PyTorch
// reference quantizers; tests/test_quant_act.py checks this implementation against both.
//   MODE 0  nvfp4_rows          scale = e4m3(clamp(bmax * fp32(1/6))); _quant_e2m1 rounding
//   MODE 1  four_over_six_rows  scales e4m3(peak * fp32(1/6)), e4m3(peak * 0.25); codes by bucketize
//                               against the E2M1 midpoints; the /4 candidate iff its squared error
//                               (16-term adjacent pairwise tree, every op separately rounded) is lower
// Every multiply/add/divide is an explicitly rounded intrinsic, so the compiler cannot contract or
// reassociate them. One CTA per padded token row; rows >= T and K blocks >= K/16 inside the last
// 64-K atom get zero scale bytes (the GEMM reads them; 0x7F would be an E4M3 NaN).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace sm120_quant {

constexpr int kThreads = 128;

__device__ __forceinline__ float e4m3_round(float v) {
  v = fminf(fmaxf(v, 0.001953125f), 448.0f);
  __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  __half_raw h = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3);
  return __half2float(__half(h));
}

__device__ __forceinline__ uint8_t e4m3_byte(float v) {   // v is already E4M3-exact
  return uint8_t(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float sgn(float v) { return v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f); }

__device__ __forceinline__ int bucketize(float r) {   // count of E2M1 midpoints strictly below r
  return int(r > 0.25f) + int(r > 0.75f) + int(r > 1.25f) + int(r > 1.75f) + int(r > 2.5f) + int(r > 3.5f) + int(r > 5.0f);
}

__device__ __forceinline__ float level(int i) {
  return i <= 4 ? float(i) * 0.5f : (i == 5 ? 3.f : (i == 6 ? 4.f : 6.f));
}

__device__ __forceinline__ float tree16(float const *v) {   // torch's adjacent pairwise tree
  float a[8], b[4], c[2];
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = __fadd_rn(v[2 * i], v[2 * i + 1]);
#pragma unroll
  for (int i = 0; i < 4; ++i) b[i] = __fadd_rn(a[2 * i], a[2 * i + 1]);
#pragma unroll
  for (int i = 0; i < 2; ++i) c[i] = __fadd_rn(b[2 * i], b[2 * i + 1]);
  return __fadd_rn(c[0], c[1]);
}

__device__ __forceinline__ int64_t sf_offset(int64_t row, int kb, int k_atoms) {
  return ((row / 128) * k_atoms + kb / 4) * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 + kb % 4;
}

template <int MODE>
__global__ void __launch_bounds__(kThreads)
quant_rows_kernel(__nv_bfloat16 const *__restrict__ x, int64_t x_stride, uint8_t *__restrict__ packed,
                  uint8_t *__restrict__ sf, float *__restrict__ gs_out, int T, int K, float inv2688, float inv6) {
  int64_t const row = blockIdx.x;
  int const KB = K / 16;
  int const KB_PAD = (KB + 3) / 4 * 4;
  int const k_atoms = KB_PAD / 4;
  if (row >= T) {
    for (int kb = threadIdx.x; kb < KB_PAD; kb += kThreads) { sf[sf_offset(row, kb, k_atoms)] = 0; }
    return;
  }
  __nv_bfloat16 const *xr = x + row * x_stride;
  // pass 1: row amax (max is exact, order-independent)
  float amax = 0.f;
  for (int i = threadIdx.x; i < K; i += kThreads) { amax = fmaxf(amax, fabsf(__bfloat162float(xr[i]))); }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
  __shared__ float red[kThreads / 32];
  if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = amax; }
  __syncthreads();
  amax = red[0];
#pragma unroll
  for (int w = 1; w < kThreads / 32; ++w) { amax = fmaxf(amax, red[w]); }
  float const gs = fmaxf(__fmul_rn(amax, inv2688), 1.17549435e-38f);
  if (threadIdx.x == 0) { gs_out[row] = gs; }
  // pass 2: one 16-element scale block per thread iteration
  for (int kb = threadIdx.x; kb < KB; kb += kThreads) {
    float s[16];
    float peak = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      s[j] = __fdiv_rn(__bfloat162float(xr[kb * 16 + j]), gs);
      peak = fmaxf(peak, fabsf(s[j]));
    }
    int idx[16];
    float scale;
    if (MODE == 1) {
      float const sc6 = e4m3_round(__fmul_rn(peak, inv6));
      float const sc4 = e4m3_round(__fmul_rn(peak, 0.25f));
      int i6[16], i4[16];
      float q6[16], q4[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        float const sg = sgn(s[j]);
        i6[j] = bucketize(fabsf(__fdiv_rn(s[j], sc6)));
        i4[j] = bucketize(fabsf(__fdiv_rn(s[j], sc4)));
        float const c6 = __fmul_rn(level(i6[j]), sg);
        float const c4 = __fmul_rn(level(i4[j]), sg);
        float const d6 = __fsub_rn(__fmul_rn(c6, sc6), s[j]);
        float const d4 = __fsub_rn(__fmul_rn(c4, sc4), s[j]);
        q6[j] = __fmul_rn(d6, d6);
        q4[j] = __fmul_rn(d4, d4);
      }
      bool const take4 = tree16(q4) < tree16(q6);
      scale = take4 ? sc4 : sc6;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        int const i = take4 ? i4[j] : i6[j];
        // sign bit: the code is level * sign, negative iff s < 0 and level > 0
        idx[j] = i | ((s[j] < 0.f && i > 0) ? 8 : 0);
      }
    } else {
      scale = e4m3_round(__fmul_rn(peak, inv6));
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        float const xs = __fdiv_rn(s[j], scale);
        float const ax = fabsf(xs);
        float const e = fmaxf(floorf(log2f(__fadd_rn(ax, xs == 0.f ? 1.f : 0.f))), 0.f);
        float const p = exp2f(e);
        float xm = __fmul_rn(__fdiv_rn(xs, p), 2.0f);
        xm = __fmul_rn(sgn(xm), floorf(__fadd_rn(fabsf(xm), 0.5f)));
        float const code = fminf(fmaxf(__fmul_rn(__fmul_rn(xm, p), 0.5f), -6.f), 6.f);
        int const t2 = int(__fmul_rn(fabsf(code), 2.0f));
        int const i = t2 <= 4 ? t2 : (t2 == 6 ? 5 : (t2 == 8 ? 6 : 7));
        idx[j] = i | ((code < 0.f) ? 8 : 0);
      }
    }
    uint32_t w0 = 0, w1 = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) { w0 |= uint32_t(idx[j] & 15) << (4 * j); }
#pragma unroll
    for (int j = 0; j < 8; ++j) { w1 |= uint32_t(idx[8 + j] & 15) << (4 * j); }
    uint2 *dst = reinterpret_cast<uint2 *>(packed + row * (K / 2) + kb * 8);
    *dst = make_uint2(w0, w1);
    sf[sf_offset(row, kb, k_atoms)] = e4m3_byte(scale);
  }
  for (int kb = KB + threadIdx.x; kb < KB_PAD; kb += kThreads) { sf[sf_offset(row, kb, k_atoms)] = 0; }
}

inline cudaError_t quant_rows(void const *x, int64_t x_stride, int T, int K, int mode, void *packed, void *sf,
                              float *gs, cudaStream_t stream) {
  int const rows_pad = (T + 127) / 128 * 128;
  // torch divides by a Python scalar as a multiply by its FP32 reciprocal
  float const inv2688 = 1.0f / 2688.0f;
  float const inv6 = 1.0f / 6.0f;
  auto const *xb = static_cast<__nv_bfloat16 const *>(x);
  auto *p = static_cast<uint8_t *>(packed);
  auto *s = static_cast<uint8_t *>(sf);
  if (mode == 1) {
    quant_rows_kernel<1><<<rows_pad, kThreads, 0, stream>>>(xb, x_stride, p, s, gs, T, K, inv2688, inv6);
  } else {
    quant_rows_kernel<0><<<rows_pad, kThreads, 0, stream>>>(xb, x_stride, p, s, gs, T, K, inv2688, inv6);
  }
  return cudaGetLastError();
}

}  // namespace sm120_quant

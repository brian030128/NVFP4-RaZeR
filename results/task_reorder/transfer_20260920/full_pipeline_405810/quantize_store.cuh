#include <cuda_fp8.h>
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
__global__ void amax_input(const B*x,size_t n,unsigned*maximum){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<n)atomicMax(maximum,__float_as_uint(fabsf(__bfloat162float(x[t]))));}
__device__ unsigned code(float x){float a=fabsf(x);unsigned c=(a>.25f)+(a>.75f)+(a>1.25f)+(a>1.75f)+(a>2.5f)+(a>3.5f)+(a>5.f);return c|((x<0.f)?8:0);}
__device__ float decode(unsigned c){const float v[8]={0,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};return (c&8)?-v[c&7]:v[c&7];}
template<bool Mapped,class L>__global__ void quantize(const B*x,uint64_t*out,uint8_t*sf,const unsigned*maximum,const int*inv,L layout,int m,int k){
 size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;size_t count=size_t(m)*k;if(t>=count)return;
 int lane=threadIdx.x&15;size_t group=t/16;int row=group/(k/16),g=group%(k/16);
 float global=__uint_as_float(*maximum)/(6.f*448.f);float w=__bfloat162float(x[t])/global;float mx=fabsf(w);
 for(int mask=8;mask;mask>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,mask,16));
 __nv_fp8_e4m3 s4(fminf(448.f,fmaxf(0x1p-9f,mx/4.f))),s6(fminf(448.f,fmaxf(0x1p-9f,mx/6.f)));
 float a=float(s4),b=float(s6);unsigned c4=code(w/a),c6=code(w/b);
 float e4=__fsub_rn(__fmul_rn(decode(c4),a),w),e6=__fsub_rn(__fmul_rn(decode(c6),b),w);e4=__fmul_rn(e4,e4);e6=__fmul_rn(e6,e6);
 for(int mask=8;mask;mask>>=1){e4+=__shfl_xor_sync(0xffffffff,e4,mask,16);e6+=__shfl_xor_sync(0xffffffff,e6,mask,16);}
 unsigned c=e4<e6?c4:c6;unsigned lo=lane<8?c<<(lane*4):0,hi=lane>=8?c<<((lane-8)*4):0;
 for(int mask=8;mask;mask>>=1){lo|=__shfl_xor_sync(0xffffffff,lo,mask,16);hi|=__shfl_xor_sync(0xffffffff,hi,mask,16);}
 if(lane==0){int dest=g;if constexpr(Mapped)dest=inv[g];out[size_t(row)*(k/16)+dest]=uint64_t(lo)|(uint64_t(hi)<<32);sf[layout(row,dest*16,0)]=e4<e6?s4.__x:s6.__x;}
}
template<class L>__global__ void packed_gather(const uint64_t*a,const uint8_t*sa,uint64_t*b,uint8_t*sb,const int*q,L layout,int m,int k){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t>=size_t(m)*(k/16))return;int r=t/(k/16),g=t%(k/16);b[t]=a[size_t(r)*(k/16)+q[g]];sb[layout(r,g*16,0)]=sa[layout(r,q[g]*16,0)];}
template<class L>__global__ void compare(const uint64_t*a,const uint8_t*sa,const uint64_t*b,const uint8_t*sb,L layout,int m,int k,int*errors){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t>=size_t(m)*(k/16))return;int r=t/(k/16),g=t%(k/16);if(a[t]!=b[t]||sa[layout(r,g*16,0)]!=sb[layout(r,g*16,0)])atomicAdd(errors,1);}

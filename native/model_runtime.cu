// Native full-model runtime: host C ABI, packed 4-bit operands, actual per-tile formats.
#include "model_types.cuh"
#include <cuda_bf16.h>
#include <memory>
#include <stdexcept>
using B=__nv_bfloat16;
#include "model_quant.cuh"
namespace model_native {
thread_local std::string error;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void status(cutlass::Status e){if(e!=cutlass::Status::kSuccess)throw std::runtime_error(cutlassGetStatusString(e));}
template<class T>T* allocate(size_t n){T*p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));return p;}
__global__ void reduce_amax(const B*x,size_t n,unsigned*maximum){
 unsigned v=0;for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)v=max(v,__float_as_uint(fabsf(__bfloat162float(x[i]))));
 for(int s=16;s;s>>=1)v=max(v,__shfl_xor_sync(0xffffffff,v,s));
 __shared__ unsigned sm[8];if((threadIdx.x&31)==0)sm[threadIdx.x/32]=v;__syncthreads();
 if(threadIdx.x<32){v=threadIdx.x<8?sm[threadIdx.x]:0;for(int s=16;s;s>>=1)v=max(v,__shfl_xor_sync(0xffffffff,v,s));if(!threadIdx.x)atomicMax(maximum,v);}
}
__global__ void alpha_value(const unsigned*mx,float*alpha,float gw){if(!threadIdx.x)*alpha=fmaxf(__uint_as_float(*mx)*(1.f/(6.f*448.f)),1.17549435e-38f)*gw;}
template<class L>__global__ void swizzle_scales(const uint8_t*in,uint8_t*out,L layout,int n,int k){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<size_t(n)*k/16){int r=t/(k/16),g=t%(k/16);out[layout(r,g*16,0)]=in[t];}}
template<class L>__global__ void unswizzle_scales(const uint8_t*in,uint8_t*out,L layout,int m,int k){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<size_t(m)*k/16){int r=t/(k/16),g=t%(k/16);out[t]=in[layout(r,g*16,0)];}}
__global__ void restore(const B*x,B*y,const int*inv,size_t count,int n){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<count)y[t]=x[t/n*n+inv[t%n]];}
struct Plan {
 int m,n,k;float gw;uint64_t*a=nullptr;uint8_t*sa=nullptr;unsigned*mx=nullptr;float*alpha=nullptr;void*workspace=nullptr;B*raw=nullptr;B*out;const int*cols;const int*rows;
 LayoutSFA la;Gemm gemm;std::unique_ptr<mixfp4::Sm100GemmLaunch<Gemm>> launch;
 Plan(int M,int N,int K,const void*wb,const void*sb,const uint8_t*map,const int*ci,const int*ri,void*y,float global,cudaStream_t stream):m(M),n(N),k(K),gw(global),out((B*)y),cols(ci),rows(ri),la(cutlass::detail::Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFA(make_shape(m,n,k,1))){
  if(m<=0||n%256||k%256)throw std::runtime_error("requires N,K multiples of256");
  a=allocate<uint64_t>(size_t(m)*k/16);sa=allocate<uint8_t>(size(filter_zeros(la)));mx=allocate<unsigned>(1);alpha=allocate<float>(1);raw=rows?allocate<B>(size_t(m)*n):out;
  check(cudaMemsetAsync(sa,0,size(filter_zeros(la)),stream));
  auto lb=cutlass::detail::Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFB(make_shape(m,n,k,1));
  typename Gemm::GemmKernel::MainloopArguments ma{};
  ma.ptr_A=(ElementA::DataType*)a;ma.dA=cutlass::make_cute_packed_stride(StrideA{},make_shape(m,k,1));
  ma.ptr_B=(ElementB::DataType const*)wb;ma.dB=cutlass::make_cute_packed_stride(StrideB{},make_shape(n,k,1));
  ma.ptr_SFA=(ElementA::ScaleFactorType*)sa;ma.layout_SFA=la;ma.ptr_SFB=(ElementB::ScaleFactorType const*)sb;ma.layout_SFB=lb;
  ma.fmt_a=1;ma.fmt_b=1;ma.type_map=map;ma.type_stride=k/64;
  auto sd=cutlass::make_cute_packed_stride(StrideD{},make_shape(m,n,1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},ma,{{1.f,0.f},nullptr,sd,(ElementD*)raw,sd}};
  args.epilogue.thread.alpha_ptr=alpha;args.scheduler.max_swizzle_size=0;
  status(gemm.can_implement(args));size_t bytes=Gemm::get_workspace_size(args);if(bytes)check(cudaMalloc(&workspace,bytes));
  status(gemm.initialize(args,workspace,stream));launch=std::make_unique<mixfp4::Sm100GemmLaunch<Gemm>>(gemm);
 }
 ~Plan(){for(void*p:{(void*)a,(void*)sa,(void*)mx,(void*)alpha,workspace})if(p)cudaFree(p);if(rows&&raw)cudaFree(raw);}
 void run(const B*x,cudaStream_t stream,bool only_gemm){
  if(!only_gemm){size_t count=size_t(m)*k;check(cudaMemsetAsync(mx,0,4,stream));
   reduce_amax<<<std::min(size_t(512),(count+255)/256),256,0,stream>>>(x,count,mx);alpha_value<<<1,32,0,stream>>>(mx,alpha,gw);
   if(cols)quantize<true><<<(count+255)/256,256,0,stream>>>(x,a,sa,mx,cols,la,m,k);else quantize<false><<<(count+255)/256,256,0,stream>>>(x,a,sa,mx,nullptr,la,m,k);
  }
  (*launch)(stream);if(rows&&!only_gemm)restore<<<(size_t(m)*n+255)/256,256,0,stream>>>(raw,out,rows,size_t(m)*n,n);check(cudaGetLastError());
 }
};
}
using namespace model_native;
extern "C" {
const char* mf_error(){return error.c_str();}
size_t mf_sf_size(int n,int k){auto l=cutlass::detail::Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFB(make_shape(1,n,k,1));return size(filter_zeros(l));}
int mf_swizzle(const void*in,void*out,int n,int k,void*st){try{auto l=cutlass::detail::Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFB(make_shape(1,n,k,1));size_t count=size_t(n)*k/16;check(cudaMemsetAsync(out,0,size(filter_zeros(l)),(cudaStream_t)st));swizzle_scales<<<(count+255)/256,256,0,(cudaStream_t)st>>>((uint8_t*)in,(uint8_t*)out,l,n,k);check(cudaGetLastError());return 0;}catch(std::exception const&e){error=e.what();return -1;}}
void* mf_create(int m,int n,int k,void*wb,void*sb,void*map,void*ci,void*ri,void*out,float gw,void*st){try{return new Plan(m,n,k,wb,sb,(uint8_t*)map,(int*)ci,(int*)ri,out,gw,(cudaStream_t)st);}catch(std::exception const&e){error=e.what();return nullptr;}}
int mf_run(void*p,void*x,void*st,int only_gemm){try{((Plan*)p)->run((B*)x,(cudaStream_t)st,only_gemm);return 0;}catch(std::exception const&e){error=e.what();return -1;}}
int mf_dump(void*vp,void*codes,void*scales,void*maximum,void*st){try{Plan*p=(Plan*)vp;auto s=(cudaStream_t)st;size_t count=size_t(p->m)*p->k;check(cudaMemcpyAsync(codes,p->a,count/2,cudaMemcpyDeviceToDevice,s));check(cudaMemcpyAsync(maximum,p->mx,4,cudaMemcpyDeviceToDevice,s));unswizzle_scales<<<(count/16+255)/256,256,0,s>>>(p->sa,(uint8_t*)scales,p->la,p->m,p->k);check(cudaGetLastError());return 0;}catch(std::exception const&e){error=e.what();return -1;}}
void mf_destroy(void*p){delete (Plan*)p;}
}

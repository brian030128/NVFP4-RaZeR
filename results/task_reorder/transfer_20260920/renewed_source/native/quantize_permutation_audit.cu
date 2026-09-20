// FourOverSix producer fusion: store each whole16 code/scale group in target order.
#define main consumer_benchmark_unused
#include "consumer_permutation.cu"
#undef main
#include <cuda_fp8.h>
#include <cstring>
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
__global__ void amax_input(const B*x,size_t n,unsigned*maximum){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<n)atomicMax(maximum,__float_as_uint(fabsf(__bfloat162float(x[t]))));}
__device__ unsigned code(float x){float a=fabsf(x);unsigned c=(a>.25f)+(a>.75f)+(a>1.25f)+(a>1.75f)+(a>2.5f)+(a>3.5f)+(a>5.f);return c|((x<0.f)?8:0);}
__host__ __device__ float decode(unsigned c){const float v[8]={0,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};return (c&8)?-v[c&7]:v[c&7];}
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
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("permutation directory required");std::string dir=argv[1];const char* audit=std::getenv("MIXFP4_QUANT_AUDIT_DIR");
for(int k:{5120,17408}){auto p=read(dir+(k==5120?"/up_proj_col.txt":"/down_proj_col.txt"),k);std::vector<int>q(k/16),inv(k/16);for(int g=0;g<k/16;++g){q[g]=p[g*16]/16;for(int i=0;i<16;++i)if(p[g*16+i]!=q[g]*16+i)throw std::runtime_error("split group");inv[q[g]]=g;}int*dq=device(q),*di=device(inv);
for(int m:{1,128,512,2048}){if(audit && m!=1)continue;size_t count=size_t(m)*k,groups=count/16;auto layout=cutlass::detail::Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFA(cute::make_shape(m,256,k,1));size_t sfcount=cute::size(cute::filter_zeros(layout));B*x=alloc<B>(count),*unused=alloc<B>(count);uint64_t*base=alloc<uint64_t>(groups),*ref=alloc<uint64_t>(groups),*fused=alloc<uint64_t>(groups);uint8_t*sbase=alloc<uint8_t>(sfcount),*sref=alloc<uint8_t>(sfcount),*sfused=alloc<uint8_t>(sfcount);unsigned*mx=alloc<unsigned>(1);int*errors=alloc<int>(1);CK(cudaMemset(mx,0,4));init<<<(count+255)/256,256>>>(x,unused,count);amax_input<<<(count+255)/256,256>>>(x,count,mx);CK(cudaDeviceSynchronize());
auto b=[&](cudaStream_t st){quantize<false><<<(count+255)/256,256,0,st>>>(x,base,sbase,mx,di,layout,m,k);};
auto f=[&](cudaStream_t st){quantize<true><<<(count+255)/256,256,0,st>>>(x,fused,sfused,mx,di,layout,m,k);};
auto s=[&](cudaStream_t st){b(st);packed_gather<<<(groups+255)/256,256,0,st>>>(base,sbase,ref,sref,dq,layout,m,k);};
s(0);f(0);CK(cudaMemset(errors,0,4));compare<<<(groups+255)/256,256>>>(ref,sref,fused,sfused,layout,m,k,errors);int e;CK(cudaMemcpy(&e,errors,4,cudaMemcpyDeviceToHost));if(e)throw std::runtime_error("code/scale mismatch");CK(cudaMemset(errors,0,4));compare<<<(groups+255)/256,256>>>(ref,sref,base,sbase,layout,m,k,errors);CK(cudaMemcpy(&e,errors,4,cudaMemcpyDeviceToHost));if(!e)throw std::runtime_error("negativecontrol");printf("CORRECT m=%d k=%d codes+SM100scales bitwise negative=%d\n",m,k,e);
if(audit){
 std::vector<B> hx(count),hd(count);std::vector<uint64_t> hc(groups);std::vector<uint8_t> hs(sfcount);unsigned hm;
 CK(cudaMemcpy(hx.data(),x,count*2,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hc.data(),base,groups*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hs.data(),sbase,sfcount,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&hm,mx,4,cudaMemcpyDeviceToHost));float maximum;std::memcpy(&maximum,&hm,4);float global=maximum/(6.f*448.f);
 for(size_t t=0;t<count;++t){int r=t/k,g=(t%k)/16;unsigned c=(hc[t/16]>>((t%16)*4))&15;__nv_fp8_e4m3 scale;scale.__x=hs[layout(r,g*16,0)];hd[t]=__float2bfloat16(decode(c)*float(scale)*global);}
 auto dump=[&](const std::string& tag,const void*ptr,size_t bytes){std::ofstream f(std::string(audit)+"/"+tag+"_k"+std::to_string(k)+".bin",std::ios::binary);f.write((const char*)ptr,bytes);if(!f)throw std::runtime_error("audit write");};dump("input",hx.data(),count*2);dump("dequant",hd.data(),count*2);
}
for(int rep=0;rep<(audit?0:5);++rep){double bt,ft;if(rep%2){ft=mixfp4::benchmark_ms(f,10,100);bt=mixfp4::benchmark_ms(b,10,100);}else{bt=mixfp4::benchmark_ms(b,10,100);ft=mixfp4::benchmark_ms(f,10,100);}double st=mixfp4::benchmark_ms(s,10,100);printf("RESULT {\"m\":%d,\"k\":%d,\"rep\":%d,\"base_ms\":%.9f,\"fused_ms\":%.9f,\"separate_ms\":%.9f}\n",m,k,rep,bt,ft,st);}
for(void*v:{(void*)x,(void*)unused,(void*)base,(void*)ref,(void*)fused,(void*)sbase,(void*)sref,(void*)sfused,(void*)mx,(void*)errors})CK(cudaFree(v));}CK(cudaFree(dq));CK(cudaFree(di));}return 0;}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}

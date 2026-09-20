// Consumer-side permutation fusion; unchanged GEMM and BF16 operation order.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "gemm_benchmark.hpp"
using B=__nv_bfloat16;
#define CK(x) mixfp4::benchmark_cuda_check(x)
std::vector<int> read(const std::string& path,int n){std::ifstream f(path);std::vector<int> p;int v;while(f>>v)p.push_back(v);auto s=p;std::sort(s.begin(),s.end());if(int(s.size())!=n)throw std::runtime_error("length");for(int i=0;i<n;++i)if(s[i]!=i)throw std::runtime_error("permutation");return p;}
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int* device(const std::vector<int>& a){int*p=alloc<int>(a.size());CK(cudaMemcpy(p,a.data(),a.size()*4,cudaMemcpyHostToDevice));return p;}
__global__ void init(B*a,B*b,size_t count){size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(i<count){a[i]=__float2bfloat16(float(int((i*97+31)%1021)-510)/128.f);b[i]=__float2bfloat16(float(int((i*71+13)%509)-254)/96.f);}}
__global__ void gather(const B*a,B*b,const int*p,int n,size_t count){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<count)b[t]=a[(t/n)*n+p[t%n]];}
__device__ B silumul(B g,B u){float x=__bfloat162float(g);B silu=__float2bfloat16(x/(1.f+expf(-x)));return __float2bfloat16(__bfloat162float(silu)*__bfloat162float(u));}
template<bool Fused>__global__ void activation(const B*g,const B*u,B*out,const int*inv,const int*q,int n,size_t count){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<count){size_t row=(t/n)*n;int c=t%n;if constexpr(Fused)c=q[c];int cu=c;if constexpr(Fused)cu=inv[c];out[t]=silumul(g[row+c],u[row+cu]);}}
template<bool Fused>__global__ void residual(const B*r,const B*d,B*out,const int*inv,int n,size_t count){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<count){size_t j=t;if constexpr(Fused)j=(t/n)*n+inv[t%n];out[t]=__float2bfloat16(__bfloat162float(r[t])+__bfloat162float(d[j]));}}
__global__ void check(const B*a,const B*b,size_t count,int*errors){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<count&&reinterpret_cast<const unsigned short*>(a)[t]!=reinterpret_cast<const unsigned short*>(b)[t])atomicAdd(errors,1);}
int diffs(B*a,B*b,size_t count,int*e){CK(cudaMemset(e,0,4));check<<<(count+255)/256,256>>>(a,b,count,e);int h;CK(cudaMemcpy(&h,e,4,cudaMemcpyDeviceToHost));return h;}
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("permutation directory required");std::string dir=argv[1];
for(int mode=0;mode<2;++mode){int n=mode?5120:17408;auto p=read(dir+(mode?"/down_proj_row.txt":"/up_proj_row.txt"),n);std::vector<int> inv(n);for(int i=0;i<n;++i)inv[p[i]]=i;auto q=mode?std::vector<int>(n):read(dir+"/down_proj_col.txt",n);if(mode)for(int i=0;i<n;++i)q[i]=i;int*di=device(inv);int*dq=device(q);
for(int m:{1,128,512,2048}){size_t count=size_t(m)*n;int blocks=(count+255)/256;B*a=alloc<B>(count),*b=alloc<B>(count),*tmp=alloc<B>(count),*tmp2=alloc<B>(count),*ref=alloc<B>(count),*out=alloc<B>(count);int*e=alloc<int>(1);init<<<blocks,256>>>(a,b,count);CK(cudaDeviceSynchronize());
auto base=[&](cudaStream_t st){if(mode)residual<false><<<blocks,256,0,st>>>(a,b,out,di,n,count);else activation<false><<<blocks,256,0,st>>>(a,b,out,di,dq,n,count);};
auto separate=[&](cudaStream_t st){gather<<<blocks,256,0,st>>>(b,tmp,di,n,count);if(mode)residual<false><<<blocks,256,0,st>>>(a,tmp,ref,di,n,count);else{activation<false><<<blocks,256,0,st>>>(a,tmp,tmp2,di,dq,n,count);gather<<<blocks,256,0,st>>>(tmp2,ref,dq,n,count);}};
auto fused=[&](cudaStream_t st){if(mode)residual<true><<<blocks,256,0,st>>>(a,b,out,di,n,count);else activation<true><<<blocks,256,0,st>>>(a,b,out,di,dq,n,count);};
separate(0);fused(0);CK(cudaDeviceSynchronize());if(diffs(ref,out,count,e))throw std::runtime_error("fused mismatch");base(0);CK(cudaDeviceSynchronize());int neg=diffs(ref,out,count,e);if(!neg)throw std::runtime_error("negative control failed");printf("CORRECT %s m=%d bitwise; negative_differences=%d\n",mode?"residual":"silu_mul",m,neg);
for(int rep=0;rep<5;++rep){double bs,fs;if(rep%2){fs=mixfp4::benchmark_ms(fused,10,100);bs=mixfp4::benchmark_ms(base,10,100);}else{bs=mixfp4::benchmark_ms(base,10,100);fs=mixfp4::benchmark_ms(fused,10,100);}double ss=mixfp4::benchmark_ms(separate,10,100);printf("RESULT {\"op\":\"%s\",\"m\":%d,\"n\":%d,\"rep\":%d,\"base_ms\":%.9f,\"fused_ms\":%.9f,\"separate_ms\":%.9f}\n",mode?"residual":"silu_mul",m,n,rep,bs,fs,ss);}
for(B*v:{a,b,tmp,tmp2,ref,out})CK(cudaFree(v));CK(cudaFree(e));}CK(cudaFree(di));CK(cudaFree(dq));}return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}

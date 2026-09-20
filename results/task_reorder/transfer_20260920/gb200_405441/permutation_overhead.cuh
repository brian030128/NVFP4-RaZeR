// Exact 16-element FP4-code/scale gather and BF16 output restoration.
// Included by a generated copy of the existing SM100 benchmark driver.
#include <fstream>
#include <stdexcept>
namespace reorder_bench {
inline std::vector<int> read_perm(std::string path, int count) {
  std::ifstream f(path); std::vector<int> p; int x;
  while (f >> x) p.push_back(x);
  auto sorted=p; std::sort(sorted.begin(),sorted.end());
  if (int(p.size()) != count) throw std::runtime_error("permutation length: "+path);
  for(int i=0;i<count;++i) if(sorted[i]!=i) throw std::runtime_error("invalid permutation");
  return p;
}
__host__ __device__ inline uint64_t code_value(size_t i) {
  // Nonconstant finite FP4 nibbles; every group has a deterministic bit pattern.
  return (uint64_t(i)*0x9e3779b97f4a7c15ULL)^0x123456789abcdef0ULL;
}
template<class L> __global__ void init_original(uint64_t* a,uint8_t* sf,L layout,int m,int k) {
  size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
  if(t>=size_t(m)*(k/16)) return;
  int r=t/(k/16),g=t%(k/16);
  a[t]=code_value(t); sf[layout(r,g*16,0)]=uint8_t(0x38+(t%8));
}
template<class L> __global__ void gather(const uint64_t* src,const uint8_t* ss,
    uint64_t* dst,uint8_t* ds,L layout,const int* q,int m,int k) {
  size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
  if(t>=size_t(m)*(k/16)) return;
  int r=t/(k/16),g=t%(k/16),old=q[g];
  dst[t]=src[size_t(r)*(k/16)+old];
  ds[layout(r,g*16,0)]=ss[layout(r,old*16,0)];
}
template<class L> __global__ void check_gather(const uint64_t* a,const uint8_t* sf,
    L layout,const int* q,int m,int k,int* errors) {
  size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
  if(t>=size_t(m)*(k/16)) return;
  int r=t/(k/16),g=t%(k/16);size_t old=size_t(r)*(k/16)+q[g];
  if(a[t]!=code_value(old)||sf[layout(r,g*16,0)]!=uint8_t(0x38+(old%8))) atomicAdd(errors,1);
}
// Gather by inverse permutation gives coalesced writes to restored output.
__global__ void restore(const uint16_t* src,uint16_t* dst,const int* inv,int m,int n) {
  size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
  if(t<size_t(m)*n) dst[t]=src[(t/n)*n+inv[t%n]];
}
__global__ void check_restore(const uint16_t* src,const uint16_t* dst,const int* p,int m,int n,int* errors) {
  size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
  if(t<size_t(m)*n && src[t]!=dst[(t/n)*n+p[t%n]]) atomicAdd(errors,1);
}
template<class L> struct Pipeline {
  int m,n,k; L layout; uint64_t *original=nullptr,*a; uint8_t *original_sf=nullptr,*sf;
  uint16_t *d,*output=nullptr; int *q=nullptr,*p=nullptr,*inv=nullptr,*errors=nullptr;
  bool active=false;
  Pipeline(int mm,int nn,int kk,L l,void* aa,void* ss,size_t sfbytes,void* dd)
      :m(mm),n(nn),k(kk),layout(l),a((uint64_t*)aa),sf((uint8_t*)ss),d((uint16_t*)dd) {
    const char* prefix=std::getenv("MIXFP4_PERM_PREFIX"); if(!prefix)return;
    active=true;auto rows=read_perm(std::string(prefix)+"_row.txt",n);
    auto cols=read_perm(std::string(prefix)+"_col.txt",k);std::vector<int> groups(k/16),inverse(n);
    for(int i=0;i<n;++i)inverse[rows[i]]=i;
    for(int g=0;g<k/16;++g){groups[g]=cols[g*16]/16;for(int j=0;j<16;++j)
      if(cols[g*16+j]!=groups[g]*16+j)throw std::runtime_error("split scale group");}
    CUDA_CHECK(cudaMalloc(&original,size_t(m)*k/2));CUDA_CHECK(cudaMalloc(&original_sf,sfbytes));
    CUDA_CHECK(cudaMemset(original_sf,0,sfbytes));
    CUDA_CHECK(cudaMalloc(&output,size_t(m)*n*2));CUDA_CHECK(cudaMalloc(&q,groups.size()*4));
    CUDA_CHECK(cudaMalloc(&p,n*4));CUDA_CHECK(cudaMalloc(&inv,n*4));CUDA_CHECK(cudaMalloc(&errors,4));
    CUDA_CHECK(cudaMemcpy(q,groups.data(),groups.size()*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p,rows.data(),n*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(inv,inverse.data(),n*4,cudaMemcpyHostToDevice));
    init_original<<<(size_t(m)*(k/16)+255)/256,256>>>(original,original_sf,layout,m,k);
    before(0);CUDA_CHECK(cudaMemset(errors,0,4));
    check_gather<<<(size_t(m)*(k/16)+255)/256,256>>>(a,sf,layout,q,m,k,errors);
    check_errors();std::printf("Permutation gather: PASSED codes and scale bytes\n");
  }
  void check_errors(){int e=0;CUDA_CHECK(cudaMemcpy(&e,errors,4,cudaMemcpyDeviceToHost));
    if(e)throw std::runtime_error("permutation correctness failure: "+std::to_string(e));}
  void before(cudaStream_t stream){if(active)gather<<<(size_t(m)*(k/16)+255)/256,256,0,stream>>>(original,original_sf,a,sf,layout,q,m,k);}
  void after(cudaStream_t stream){if(active)restore<<<(size_t(m)*n+255)/256,256,0,stream>>>(d,output,inv,m,n);}
  void validate_output(){if(!active)return;after(0);CUDA_CHECK(cudaMemset(errors,0,4));
    check_restore<<<(size_t(m)*n+255)/256,256>>>(d,output,p,m,n,errors);check_errors();
    std::printf("Permutation output restoration: PASSED bitwise\n");}
  ~Pipeline(){cudaFree(original);cudaFree(original_sf);cudaFree(output);cudaFree(q);cudaFree(p);cudaFree(inv);cudaFree(errors);}
};
}

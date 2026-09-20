#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA_CHECK(x) do { auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(err));exit(1);}} while(0)
#include "permutation_overhead.cuh"
__global__ void distinct(uint16_t* d,size_t count){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<count)d[t]=uint16_t(t*4051+17);}
int main(int argc,char**argv){
 if(argc!=4)return 2;int m=std::atoi(argv[2]),n=std::atoi(argv[3]);
 // An odd multiplier makes each row's N<65536 bit patterns unique.
 if(m<1||n<1||n>=65536)return 2;
 auto rows=reorder_bench::read_perm(std::string(argv[1])+"_row.txt",n);
 std::vector<int> inverse(n),changed;
 for(int j=0;j<n;++j)inverse[rows[j]]=j;
 for(int j=0;j<n;++j)if(inverse[j]!=j)changed.push_back(j);
 int count=int(changed.size());if(!count)return 2;
 uint16_t *src,*dst,*tmp;int *p,*inv,*moved,*errors;
 CUDA_CHECK(cudaMalloc(&src,size_t(m)*n*2));CUDA_CHECK(cudaMalloc(&dst,size_t(m)*n*2));
 CUDA_CHECK(cudaMalloc(&tmp,size_t(m)*count*2));CUDA_CHECK(cudaMalloc(&p,n*4));CUDA_CHECK(cudaMalloc(&inv,n*4));
 CUDA_CHECK(cudaMalloc(&moved,count*4));CUDA_CHECK(cudaMalloc(&errors,4));
 CUDA_CHECK(cudaMemcpy(p,rows.data(),n*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(inv,inverse.data(),n*4,cudaMemcpyHostToDevice));
 CUDA_CHECK(cudaMemcpy(moved,changed.data(),count*4,cudaMemcpyHostToDevice));
 distinct<<<(size_t(m)*n+255)/256,256>>>(src,size_t(m)*n);
 for(int mode=0;mode<2;++mode){
  if(mode==0)reorder_bench::restore<<<(size_t(m)*n+255)/256,256>>>(src,dst,inv,m,n);
  else{
   CUDA_CHECK(cudaMemcpy(dst,src,size_t(m)*n*2,cudaMemcpyDeviceToDevice));
   reorder_bench::sparse_read<<<(size_t(m)*count+255)/256,256>>>(dst,tmp,moved,inv,m,n,count);
   reorder_bench::sparse_write<<<(size_t(m)*count+255)/256,256>>>(tmp,dst,moved,m,n,count);
  }
  CUDA_CHECK(cudaMemset(errors,0,4));reorder_bench::check_restore<<<(size_t(m)*n+255)/256,256>>>(src,dst,p,m,n,errors);
  int e;CUDA_CHECK(cudaMemcpy(&e,errors,4,cudaMemcpyDeviceToHost));if(e)return 1;
  // Negative control: unchanged output must fail for every moved channel.
  CUDA_CHECK(cudaMemset(errors,0,4));reorder_bench::check_restore<<<(size_t(m)*n+255)/256,256>>>(src,src,p,m,n,errors);
  CUDA_CHECK(cudaMemcpy(&e,errors,4,cudaMemcpyDeviceToHost));if(e!=m*count)return 1;
  printf("DISTINCT_OUTPUT PASSED m=%d n=%d sparse=%d moved=%d negative_errors=%d\n",m,n,mode,count,e);
 }
}

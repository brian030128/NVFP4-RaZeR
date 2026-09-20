"""Generate native unchanged-GEMM + fused-consumer pipeline benchmark."""
from pathlib import Path
import sys,json,hashlib
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
src=Path('/home/u4320956/mixfp4/src/mixed_nvfp4_gemm_sm100.cu');s=src.read_text()
s=s.replace('using namespace cute;','#define main consumer_microbenchmark_main\n#include "consumer_permutation_vector.cu"\n#undef main\n#include "permutation_overhead.cuh"\nusing namespace cute;',1)
s=s.replace('using namespace cute;', '__global__ void fill_b_words(uint64_t* dst,size_t count){size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(t<count)dst[t]=reorder_bench::code_value(t)^0xd349794c75323ab7ULL;}\nusing namespace cute;',1)
anchor='  std::vector<ElementD> checked_output;'
body=r'''
  if(std::getenv("MIXFP4_PERM_PREFIX")){
    if(check)throw std::runtime_error("Run separate small host-reference smoke");
    reorder_bench::Pipeline<LayoutSFA> perm(m,n,k,layout_sfa,block_a.device_data(),block_sfa.device_data(),block_sfa.capacity(),block_d.device_data());
    fill_b_words<<<(size_t(n)*k/16+255)/256,256>>>((uint64_t*)block_b.device_data(),size_t(n)*k/16);
    const bool down=(n==5120);if(!down&&n!=17408)throw std::runtime_error("unsupported projection");
    size_t count=size_t(m)*n;int blocks=(count/2+255)/256;
    B*other=alloc<B>(count),*unused=alloc<B>(count),*tmp=alloc<B>(count),*ref=alloc<B>(count),*out=alloc<B>(count);
    init<<<(count+255)/256,256>>>(other,unused,count);
    std::string prefix=std::getenv("MIXFP4_PERM_PREFIX");auto slash=prefix.find_last_of('/');
    std::vector<int> q(n);if(down){for(int i=0;i<n;++i)q[i]=i;}else q=read(prefix.substr(0,slash)+"/down_proj_col.txt",n);
    int*dq=device(q);B*gemmout=reinterpret_cast<B*>(block_d.device_data());
    auto consume_base=[&](cudaStream_t st){if(down)residual<false><<<blocks,256,0,st>>>(other,gemmout,out,perm.inv,n,count);else activation<false><<<blocks,256,0,st>>>(other,gemmout,out,perm.inv,dq,n,count);};
    auto consume_fused=[&](cudaStream_t st){if(down)residual<true><<<blocks,256,0,st>>>(other,gemmout,out,perm.inv,n,count);else activation<true><<<blocks,256,0,st>>>(other,gemmout,out,perm.inv,dq,n,count);};
    auto consume_separate=[&](cudaStream_t st){perm.after(st);if(down)residual<false><<<blocks,256,0,st>>>(other,reinterpret_cast<B*>(perm.output),ref,perm.inv,n,count);else{activation<false><<<blocks,256,0,st>>>(other,reinterpret_cast<B*>(perm.output),tmp,perm.inv,dq,n,count);gather<<<blocks,256,0,st>>>(tmp,ref,dq,n,count);}};
    launch();consume_separate(0);consume_fused(0);CUDA_CHECK(cudaDeviceSynchronize());
    if(diffs(ref,out,count,perm.errors))throw std::runtime_error("pipeline fused mismatch");
    consume_base(0);CUDA_CHECK(cudaDeviceSynchronize());int neg=diffs(ref,out,count,perm.errors);if(neg==0)throw std::runtime_error("pipeline negative control failed");
    printf("PIPELINE_CORRECT bitwise; negative_differences=%d\n",neg);
    for(int rep=0;rep<5;++rep){
      auto base=[&](cudaStream_t st){launch(st);consume_base(st);};
      auto fused=[&](cudaStream_t st){perm.before(st);launch(st);consume_fused(st);};
      auto separate=[&](cudaStream_t st){perm.before(st);launch(st);consume_separate(st);};
      double b,f;if(rep%2){f=mixfp4::benchmark_ms(fused,10,100);b=mixfp4::benchmark_ms(base,10,100);}else{b=mixfp4::benchmark_ms(base,10,100);f=mixfp4::benchmark_ms(fused,10,100);}
      double old=mixfp4::benchmark_ms(separate,10,100);
      printf("PIPELINE_RESULT {\"op\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"rep\":%d,\"base_ms\":%.9f,\"fused_ms\":%.9f,\"separate_ms\":%.9f}\n",down?"down_residual":"up_silu",m,n,k,rep,b,f,old);
    }
    for(B*v:{other,unused,tmp,ref,out})CUDA_CHECK(cudaFree(v));CUDA_CHECK(cudaFree(dq));return 0;
  }
'''
assert s.count(anchor)==1;s=s.replace(anchor,body+anchor)
(out/'consumer_pipeline.cu').write_text(s)
(out/'source.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,Path(__file__),Path('native/consumer_permutation_vector.cu'),Path('native/permutation_overhead.cuh'))},indent=2)+'\n')

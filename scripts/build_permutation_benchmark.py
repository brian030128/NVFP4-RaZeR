"""Generate a benchmark extension while preserving the sibling kernel source."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
p=root/'src/mixed_nvfp4_gemm_sm100.cu';s=p.read_text()
s=s.replace('using namespace cute;', '#include "permutation_overhead.cuh"\n\nusing namespace cute;',1)
needle='  std::vector<ElementD> checked_output;'
assert s.count(needle)==1
s=s.replace(needle,'''  reorder_bench::Pipeline<LayoutSFA> perm(m,n,k,layout_sfa,block_a.device_data(),
      block_sfa.device_data(),block_sfa.capacity(),block_d.device_data());
  if (perm.active && check) throw std::runtime_error("Use separate small GEMM reference check; permutation mode has its own byte-exact checks");
  launch(); CUDA_CHECK(cudaDeviceSynchronize()); perm.validate_output();
  if (perm.active) {
    for (int rep=0;rep<5;++rep) {
      double base=0,total=0;
      auto baseline=[&](){return mixfp4::benchmark_ms([&](cudaStream_t st){launch(st);},warmup_iters,bench_iters);};
      auto pipeline=[&](){return mixfp4::benchmark_ms([&](cudaStream_t st){perm.before(st);launch(st);perm.after(st);},warmup_iters,bench_iters);};
      if(rep%2){total=pipeline();base=baseline();}else{base=baseline();total=pipeline();}
      double input=mixfp4::benchmark_ms([&](cudaStream_t st){perm.before(st);},warmup_iters,bench_iters);
      double output=mixfp4::benchmark_ms([&](cudaStream_t st){perm.after(st);},warmup_iters,bench_iters);
      std::printf("PERM_RESULT {\\"rep\\":%d,\\"m\\":%d,\\"n\\":%d,\\"k\\":%d,\\"gemm_ms\\":%.9f,\\"pipeline_ms\\":%.9f,\\"input_ms\\":%.9f,\\"output_ms\\":%.9f}\\n",rep,m,n,k,base,total,input,output);
    }
    perm.validate_output(); return 0;
  }
'''+needle)
(out/'permutation_gemm.cu').write_text(s)
(out/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()},indent=2)+'\n')

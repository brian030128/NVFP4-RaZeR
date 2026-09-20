"""Generate a direct-store permutation epilogue alongside the original GEMM."""
from pathlib import Path
import sys,json,hashlib,re
src=Path('/home/u4320956/mixfp4');out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
aux=src/'3rdparty/cutlass/include/cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp'
s=aux.read_text();a=s.index('template <\n  class Element,\n  class EpilogueTile,   // Unused');b=s.index('\ntemplate <',a+20)
block=s[a:b];start=block.index('  using ElementAux = Element;')
block='template <class Element, class LayoutOrStrideMNL>\nstruct Sm100PermutationStore {\n  static constexpr auto RoundStyle = FloatRoundStyle::round_to_nearest;\n  static constexpr bool EnableNullptr = false;\n'+block[start:]
block=block.replace('Sm90AuxStore(', 'Sm100PermutationStore(')
block=block.replace('return frg_input;', 'return convert_input(frg_input);')
block=block.replace('StrideMNL dAux = {};','StrideMNL dAux = {};\n    int const* permutation = nullptr;')
a=block.index('      constexpr auto MCL');b=block.index('\n    }\n  };',a)
block=block[:a]+'''      auto coords = coalesce(tC_cAux(_,_,_,epi_m,epi_n));
      auto values = coalesce(tC_rAux);
      CUTLASS_PRAGMA_UNROLL
      for (int i=0;i<size(values);++i) {
        auto c=coords(i);
        if (elem_less(c,problem_shape_mnl)) {
          int n=int(get<1>(c));
          int destination=params_ptr->permutation ? params_ptr->permutation[n] : n;
          params_ptr->ptr_aux[int64_t(get<0>(c))*get<0>(params_ptr->dAux)+destination+
                              int64_t(get<2>(c))*get<2>(params_ptr->dAux)]=values(i);
        }
      }
'''+block[b:]
header=s[:s.index('#pragma once')] if '#pragma once' in s else s[:s.index('#include')]
(out/'permutation_store.cuh').write_text(header+'\n#pragma once\n#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"\nnamespace cutlass::epilogue::fusion {\n'+block+'\n}\n')
p=src/'src/mixed_nvfp4_gemm_sm100.cu';s=p.read_text()
s=s.replace('using namespace cute;', '#include "permutation_store.cuh"\n#include "permutation_overhead.cuh"\nusing namespace cute;',1)
anchor='using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;'
definitions='''
using PermStore = cutlass::epilogue::fusion::Sm100PermutationStore<ElementD,StrideD>;
using PermEVT = cutlass::epilogue::fusion::Sm90EVT<PermStore,cutlass::epilogue::fusion::Sm90AccFetch>;
using PermEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,OperatorClass,MmaTileShape,ClusterShape,
    Shape<_128,Int<(MIXFP4_TILE_N == 256 ? 64 : 32)>>,
    ElementAccumulator,ElementAccumulator,void,LayoutCTag,AlignmentC,
    void,LayoutDTag,AlignmentD,
    cute::conditional_t<MIXFP4_TILE_M == 256,cutlass::epilogue::TmaWarpSpecialized2Sm,
        cutlass::epilogue::TmaWarpSpecialized1Sm>,PermEVT>::CollectiveOp;
using PermKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,CollectiveMainloop,PermEpilogue,void>;
using PermGemm = cutlass::gemm::device::GemmUniversalAdapter<PermKernel>;
__global__ void nonconstant_words(uint64_t* data,size_t count){size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;if(i<count)data[i]=reorder_bench::code_value(i)^0xd349794c75323ab7ULL;}
'''
assert s.count(anchor)==1;s=s.replace(anchor,anchor+definitions)
anchor='  std::vector<ElementD> checked_output;'
body='''
  if (std::getenv("MIXFP4_PERM_PREFIX")) {
    if(check)throw std::runtime_error("Separate original GEMM reference check from paired full-shape fused checks");
    reorder_bench::Pipeline<LayoutSFA> perm(m,n,k,layout_sfa,block_a.device_data(),
        block_sfa.device_data(),block_sfa.capacity(),block_d.device_data());
    nonconstant_words<<<(size_t(n)*k/16+255)/256,256>>>((uint64_t*)block_b.device_data(),size_t(n)*k/16);
    cutlass::device_memory::allocation<ElementD> fused_output(size_t(m)*n);
    typename PermGemm::Arguments pargs{cutlass::gemm::GemmUniversalMode::kGemm,
        {m,n,k,1},mainloop_args,{{{}, {fused_output.get(),stride_d,perm.p}},nullptr,stride_c,block_d.device_data(),stride_d}};
    pargs.scheduler.raster_order=arguments.scheduler.raster_order;
    pargs.scheduler.max_swizzle_size=arguments.scheduler.max_swizzle_size;
    PermGemm fused;CUTLASS_CHECK(fused.can_implement(pargs));
    cutlass::device_memory::allocation<uint8_t> pworkspace(PermGemm::get_workspace_size(pargs));
    CUTLASS_CHECK(fused.initialize(pargs,pworkspace.get()));
    mixfp4::Sm100GemmLaunch<PermGemm> fused_launch(fused);
    launch();fused_launch();CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(perm.errors,0,4));
    reorder_bench::check_restore<<<(size_t(m)*n+255)/256,256>>>((uint16_t*)block_d.device_data(),(uint16_t*)fused_output.get(),perm.p,m,n,perm.errors);
    perm.check_errors();
    CUDA_CHECK(cudaMemset(perm.errors,0,4));
    reorder_bench::check_restore<<<(size_t(m)*n+255)/256,256>>>((uint16_t*)block_d.device_data(),(uint16_t*)block_d.device_data(),perm.p,m,n,perm.errors);
    int negative=0;CUDA_CHECK(cudaMemcpy(&negative,perm.errors,4,cudaMemcpyDeviceToHost));
    if(negative==0)throw std::runtime_error("Nonconstant fused-output negative control failed");
    printf("FUSED_CORRECT PASSED bitwise; unchanged-output differences=%d\\n",negative);
    for(int rep=0;rep<5;++rep){
      auto base=[&](){return mixfp4::benchmark_ms([&](cudaStream_t st){launch(st);},warmup_iters,bench_iters);};
      auto fused_only=[&](){return mixfp4::benchmark_ms([&](cudaStream_t st){fused_launch(st);},warmup_iters,bench_iters);};
      double b,f;if(rep%2){f=fused_only();b=base();}else{b=base();f=fused_only();}
      double total=mixfp4::benchmark_ms([&](cudaStream_t st){perm.before(st);fused_launch(st);},warmup_iters,bench_iters);
      double old=mixfp4::benchmark_ms([&](cudaStream_t st){perm.before(st);launch(st);perm.after(st);},warmup_iters,bench_iters);
      printf("FUSED_RESULT {\\"m\\":%d,\\"n\\":%d,\\"k\\":%d,\\"rep\\":%d,\\"base_ms\\":%.9f,\\"fused_ms\\":%.9f,\\"pipeline_ms\\":%.9f,\\"separate_ms\\":%.9f}\\n",m,n,k,rep,b,f,total,old);
    }
    return 0;
  }
'''
assert s.count(anchor)==1;s=s.replace(anchor,body+anchor)
# SM100's builder advertises void D, but this checkout's collective still uses
# void directly for shared-memory arithmetic. Supply the aux element type and
# suppress ordinary D writes only for this void-D fused instantiation.
epi=src/'3rdparty/cutlass/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp'
e=epi.read_text();assert e.count('  using ElementD = ElementD_;')==1
e=e.replace('  using ElementD = ElementD_;','  static constexpr bool DisableDestination = cute::is_void_v<ElementD_>;\n  using ElementD = cute::conditional_t<DisableDestination,epilogue::fusion::get_element_aux_t<FusionCallbacks>,ElementD_>;')
needle='copy(params.tma_store_d, bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));'
assert e.count(needle)==2;e=e.replace(needle,'if constexpr (!DisableDestination) { '+needle+' }')
needle='copy(tiled_r2s, tRS_rD, tRS_sD(_,_,_,store_pipe_producer_state.index()));'
assert e.count(needle)==2;e=e.replace(needle,'if constexpr (!DisableDestination) { '+needle+' }')
override=out/'override/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp';override.parent.mkdir(parents=True,exist_ok=True);override.write_text(e)
umbrella=epi.with_name('collective_epilogue.hpp');u=umbrella.read_text()
u=re.sub(r'#include \"([^\"]+)\"',lambda m: m.group(0) if '/' in m[1] or m[1]==epi.name else '#include \"'+str(epi.parent/m[1])+'\"',u)
override.with_name('collective_epilogue.hpp').write_text(u)
(out/'fused_permutation.cu').write_text(s)
(out/'source.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (p,aux,Path(__file__),Path('native/permutation_overhead.cuh'))},indent=2)+'\n')

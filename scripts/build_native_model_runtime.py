"""Build-local mixed mainloop override; never edits the sibling kernel checkout."""
from pathlib import Path
import sys,hashlib,json
out=Path(sys.argv[1]);(out/'collective').mkdir(parents=True,exist_ok=True)
src=Path('/home/u4320956/mixfp4/src')
h=(src/'collective/sm100_blockscaled_mma_mixed.hpp').read_text()
def replace(a,b,count):
 global h
 assert h.count(a)==count,(a,h.count(a))
 h=h.replace(a,b)
replace('uint8_t fmt_b{kFmtKeep};','uint8_t fmt_b{kFmtKeep};\n    uint8_t const* type_map{nullptr};\n    int type_stride{0};',1)
replace('uint8_t fmt_b;  // MIXFP4','uint8_t fmt_b;  // MIXFP4\n    uint8_t const* type_map;\n    int type_stride;',1)
replace(', fmt_b_(params.fmt_b) {    // MIXFP4',', fmt_b_(params.fmt_b) {    // MIXFP4\n    type_map_ = params.type_map; type_stride_ = params.type_stride;',1)
replace('args.fmt_b   // MIXFP4','args.fmt_b, args.type_map, args.type_stride // MIXFP4',1)
replace('uint8_t fmt_b_{kFmtKeep};  // MIXFP4','uint8_t fmt_b_{kFmtKeep};  // MIXFP4\n  uint8_t const* type_map_{nullptr};\n  int type_stride_{0};',1)
replace('tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;','tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;\n    int format_k_tile = 0;',1)
replace('int a_block = k_block;', '''int a_block = k_block;
          // Default full-K scheduler only. One descriptor covers N256 x K64.
          if (type_map_) {
            tiled_mma.idesc_.b_format_ = type_map_[int(get<1>(cta_tile_coord)) * type_stride_
                + format_k_tile * size<2>(tCrA) + k_block] ? 0 : 1;
          }''',2)
replace('mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);','mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);\n      ++format_k_tile;',2)
(out/'collective/sm100_blockscaled_mma_mixed.hpp').write_text(h)
s=(src/'mixed_nvfp4_gemm_sm100.cu').read_text();s=s[:s.index('template <typename T>\nauto make_iterator')]
(out/'model_types.cuh').write_text(s)
q=Path('native/quantize_permutation.cu').read_text();q=q[q.index('#include <cuda_fp8.h>'):q.index('template<class L>__global__ void packed_gather')]
q=q.replace('float global=__uint_as_float(*maximum)/(6.f*448.f);','float global=fmaxf(__uint_as_float(*maximum)/(6.f*448.f),1.17549435e-38f);')
(out/'model_quant.cuh').write_text(q)
paths=[Path(__file__),Path('native/model_runtime.cu'),Path('native/quantize_permutation.cu'),src/'collective/sm100_blockscaled_mma_mixed.hpp',src/'mixed_nvfp4_gemm_sm100.cu']
(out/'source.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')

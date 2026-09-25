#!/usr/bin/env python3
"""B. Complete-Linear benchmark: everything y = Linear(x) costs, not just the GEMM.

Per (shape, T) and policy it reports
    total_ms     one full forward (activation quantization + packing + scale placement, GEMM with the
                 fused per-token-scale / bias epilogue, output already in [T, out] layout), by CUDA
                 events over back-to-back calls -- i.e. device-bound time including launch gaps;
    graph_ms     the same call captured in a CUDA graph (launch overhead removed);
    host_us      host time per call (Python + ctypes + CUTLASS argument setup + Triton launch),
                 the floor on latency when the GPU is idle (decode);
    quant_us / gemm_us   CUPTI device time of the quantizer kernel and the GEMM kernel.
Policies: bf16 (nn.Linear, cuBLAS); native NVFP4 on the stock CUTLASS kernel family (same
quantizer, epilogue and per-GPU tile selection: the fair E2M1 baseline); native FourOverSix and
N16K64-map on the mixed family with tile selection; and N16K64-map on the single 128-wide build.

    python sm120/bench/linear.py --models llama8b,qwen4b --out sm120/results/bench/linear_rtx5090.json
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as B  # noqa: E402
from kernel import selector_masks  # noqa: E402

from mixfp4_sm120.artifact import pack_module  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import NativeLinear  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402


def graph_time(fn, x, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fn(x)
    t = B.time_fn(g.replay, iters=iters)
    return t['ms'], y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='llama8b,qwen4b')
    ap.add_argument('--tokens', default='1,16,128,512,2048,8192')
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-busy', action='store_true')
    args = ap.parse_args()
    if not args.allow_busy:
        B.require_idle()
    mixed, stock, wide = KernelSet('mixed'), KernelSet('stock'), Kernel.load('n16k64_wA')
    res = dict(gpu=B.gpu_info(), kernel_sets={'mixed': mixed.describe(), 'stock': stock.describe()}, rows=[])
    for model in args.models.split(','):
        sel = selector_masks(model, 16)
        for proj, (n, k) in B.MODEL_SHAPES[model].items():
            g = torch.Generator(device='cpu').manual_seed(n + k)
            lin = torch.nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device='cuda')
            with torch.no_grad():
                lin.weight.copy_(torch.randn(n, k, generator=g) * 0.02)
            mask = sel[proj][0] if proj in sel else None
            pols = {
                'bf16': lin,
                'native_nvfp4_stock': NativeLinear(pack_module(proj, lin.weight, None, 'nvfp4'), stock, 'nvfp4_rows', proj),
                'native_four_over_six': NativeLinear(pack_module(proj, lin.weight, None, 'four_over_six'), mixed,
                                                     'four_over_six_rows', proj),
                'native_n16k64_map': NativeLinear(pack_module(proj, lin.weight, None, 'map', mask, (16, 64)), mixed,
                                                  'four_over_six_rows', proj),
                'native_n16k64_map_w128': NativeLinear(pack_module(proj, lin.weight, None, 'map', mask, (16, 64)), wide,
                                                       'four_over_six_rows', proj),
            }
            for t in [int(x) for x in args.tokens.split(',')]:
                x = torch.randn(t, k, generator=g).cuda().bfloat16()
                for pname, mod in pols.items():
                    with torch.no_grad():
                        fn = lambda: mod(x)  # noqa: E731
                        tot = B.time_fn(fn, iters=20)
                        host = B.time_host(fn, iters=100)
                        gms, _ = graph_time(lambda xx: mod(xx), x)
                        kt = B.kernel_times(fn, iters=10)
                    quant = sum(v['us'] for kname, v in kt.items() if 'quant_rows' in kname)
                    gemm = sum(v['us'] for kname, v in kt.items() if 'quant_rows' not in kname)
                    res['rows'].append(dict(model=model, proj=proj, out=n, inp=k, tokens=t, policy=pname,
                                            total_ms=tot['ms'], graph_ms=gms, host_us=host, quant_us=quant,
                                            gemm_us=gemm, kernels=kt))
                print(f'{model} {proj} T={t} done', flush=True)
                B.write(args.out, res)
    B.write(args.out, res)


if __name__ == '__main__':
    main()

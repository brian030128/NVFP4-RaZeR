#!/usr/bin/env python3
"""C2-lite timing: M = N = K = 4096 GEMM on the RTX PRO 6000 (results/paper_extra/C2/PROTOCOL.md).

    PAPER_PYTHON experiments/paper_extra/C2_time.py --out JSON

CAVEAT (every C2 table carries it): ncu is unavailable on this machine (not installed; counters blocked,
ERR_NVGPUCTRPERM); clocks were not locked (no root), and the rotated 3-round method stands in for that; tensor-pipe
counts are static SASS census counts (C2_sass.py), not measured counters; there are no stall reasons and no
bank-conflict data; per-MMA-branch kernel numbers are historical (sm120/kernel/docs/mixed_nvfp4_report.md, RTX 5090,
ncu), cited, not re-measured.

Kernels, all at the 128-wide CTA tile (the builds without a width suffix), GEMM only:
  stock_wA        NVFP4 weights (numerics.quantize_weight 'nvfp4')
  e2m1            n16k64_wA with every tile E2M1 (tags clear; FourOverSix weights)
  e0m3            n16k64_wA with every tile E0M3
  real            n16k64_wA with the tags of model.layers.0.self_attn.o_proj (4096 x 4096) of the committed
                  Llama-3.1-8B TM-OPT+TC 16x64 map; its E0M3 share is recorded
  nodisp          n16k64_wA_nodisp: the same tile and arrangement, format dispatch compiled out (E2M1 only)
Activations: 4096 tokens, FourOverSix per-token (numerics.quantize_act). Weights: seeded normal x 0.02.
Timing: CUPTI kernel time, the median of 20 calls, in each of 3 rounds in a rotated order (round r starts at position
r * len / 3); the median of the per-round medians; TFLOP/s = 2 * 4096^3 / time.
"""
import argparse
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402
from kernel import place  # noqa: E402

from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402

CAVEAT = ('ncu is unavailable on this machine (not installed; counters blocked, ERR_NVGPUCTRPERM); clocks were not '
          'locked (no root), and the rotated 3-round method stands in for that; tensor-pipe counts are static SASS '
          'census counts (per-path OMMA count x loop trips), not measured counters; there are no stall reasons and no '
          'bank-conflict data; per-MMA-branch kernel numbers are historical (sm120/kernel/docs/mixed_nvfp4_report.md, '
          'RTX 5090, ncu), cited but not re-measured on this GPU.')
LAYER = 'model.layers.0.self_attn.o_proj'
MAP = Path('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_16x64.mixfp4map')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--size', type=int, default=4096)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    n = k = t = args.size
    g = torch.Generator('cpu').manual_seed(4096)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
    xp, xsf = N.pack_nibbles(xn), place(xsb, k)
    header, masks, _ = mapio.read_map(MAP)
    real = masks[LAYER]
    assert tuple(real.shape) == (n // 16, k // 64), real.shape
    grid = (n // 16, k // 64)
    tags = dict(e2m1=torch.zeros(grid, dtype=torch.bool), e0m3=torch.ones(grid, dtype=torch.bool), real=real)

    def weights(kind, mask=None):
        wn, wsb, gsw = N.quantize_weight(w, kind, mask, (16, 64) if mask is not None else None)
        return N.pack_nibbles(wn), place(wsb, k), float(gsw)
    kernels = dict(stock_wA=Kernel.load('stock_wA'), mixed=Kernel.load('n16k64_wA'), nodisp=Kernel.load('n16k64_wA_nodisp'))
    configs = [('stock_wA', kernels['stock_wA'], weights('nvfp4'))]
    configs += [(name, kernels['mixed'], weights('map', m)) for name, m in tags.items()]
    configs.append(('nodisp', kernels['nodisp'], weights('four_over_six')))

    def fn(kern, wt):
        wp, wsf, gsw = wt
        return lambda: kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gsw, scale_n=gsx, check=False)
    calls = [fn(kern, wt) for _, kern, wt in configs]
    times = {i: [] for i in range(len(configs))}
    for r in range(args.rounds):
        shift = (r * len(configs) // args.rounds) % len(configs)
        for i in list(range(shift, len(configs))) + list(range(shift)):
            kt = B.kernel_times(calls[i], iters=20, match=lambda s: 'cutlass' in s or 'Gemm' in s or 'device_kernel' in s)
            times[i].append(sum(v['us'] for v in kt.values()))
    flops = 2.0 * n * k * t
    res = dict(status='complete', caveat=CAVEAT, gpu=B.gpu_info(), size=args.size, rounds=args.rounds,
               kernels={c: kernels[c].sha256 for c in kernels}, layer=LAYER, map=str(MAP),
               layer_e0m3_share=float(real.float().mean()), rows=[])
    for i, (name, kern, _) in enumerate(configs):
        us = statistics.median(times[i])
        res['rows'].append(dict(kernel=name, build=kern.cfg.name, time_us=us, rounds_us=times[i], tflops=flops / us / 1e6))
        print(f'{name:9s} {kern.cfg.name:17s} {us:9.2f} us  {flops / us / 1e6:7.1f} TFLOP/s  rounds {[round(v, 2) for v in times[i]]}', flush=True)
    B.write(args.out, res)


if __name__ == '__main__':
    main()

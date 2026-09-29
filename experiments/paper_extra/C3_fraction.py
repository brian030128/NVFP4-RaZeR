#!/usr/bin/env python3
"""Experiment C3: GEMM latency against the share of E0M3 tiles, at fixed matrix size (results/paper_extra/C3/PROTOCOL.md).

    PAPER_PYTHON experiments/paper_extra/C3_fraction.py --out JSON [--shapes 4096x4096,...] [--tokens 128,...]

- Shapes (out x in): 4096x4096, 14336x4096, 4096x14336 (Llama-3.1-8B's). T in {128, 512, 2048, 8192}.
- Tags: E0M3 tile share f in {0, 1, 2, 5, 10, 25, 50, 75, 100} %, in two patterns: random (a seeded Bernoulli(f) per
  tile) and contiguous (the first round(f x tiles) tiles in row-major tile order).
- Weights: seeded random normal x 0.02, quantized with the tags (E0M3 alpha = 1 tiles, FourOverSix elsewhere); FourOverSix
  per-token activations (the operands of sm120/bench/kernel.py).
- Kernels, GEMM only (the activation quantizer is not timed):
    mixed@table   n16k64_wA at the RTX PRO 6000 tile table's width (the deployment path)
    mixed@128     n16k64_wA forced to width 128 (next to nodisp, for the dispatch cost)
    n8k64_wB      the 8x64 kernel (its single build)
  references (all-E2M1 weights):
    stock@table   stock_wA at the table's width      stock@128   stock_wA at width 128
    stock_wB      the 8x64 same-placement reference   nodisp@128  n16k64_wA_nodisp: same tile, format dispatch compiled out
- Timing: CUPTI kernel time, median of 20 calls, in each of 3 rounds with the configurations in a rotated order (round r
  starts at position r * len / 3; deviation 1 of results/paper/PROTOCOL.md); the median of the per-round medians.
Decomposition (C3_analyze.py): fixed cost = mixed at f = 0 against stock at the same width; dispatch cost = mixed@128 at
f = 0 against nodisp@128; the E0M3-share cost = mixed(f) against mixed(0).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402
from kernel import operands  # noqa: E402

from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
PATTERNS = ('random', 'contiguous')


def tags(grid, f, pattern, seed):
    n = grid[0] * grid[1]
    if pattern == 'random':
        g = torch.Generator('cpu').manual_seed(seed)
        return torch.rand(grid, generator=g) < f
    m = torch.zeros(n, dtype=torch.bool)
    m[:round(f * n)] = True
    return m.reshape(grid)


def gemm_us(fn):
    kt = B.kernel_times(fn, iters=20, match=lambda s: 'cutlass' in s or 'Gemm' in s or 'device_kernel' in s)
    return sum(v['us'] for v in kt.values()) if kt else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shapes', default='4096x4096,14336x4096,4096x14336')
    ap.add_argument('--tokens', default='128,512,2048,8192')
    ap.add_argument('--fractions', default=','.join(str(f) for f in FRACTIONS))
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    mixed, stock = KernelSet('mixed'), KernelSet('stock')
    k128, s128, nodisp = Kernel.load('n16k64_wA'), Kernel.load('stock_wA'), Kernel.load('n16k64_wA_nodisp')
    n8, swb = Kernel.load('n8k64_wB'), Kernel.load('stock_wB')
    fractions = [float(f) for f in args.fractions.split(',')]
    res = dict(status='running', gpu=B.gpu_info(), rounds=args.rounds, fractions=fractions, patterns=PATTERNS,
               kernels={k: (v.sha256 if isinstance(v, Kernel) else v.describe()) for k, v in dict(
                   mixed=mixed, stock=stock, n16k64_wA=k128, stock_wA=s128, nodisp=nodisp, n8k64_wB=n8, stock_wB=swb).items()},
               rows=[])
    for shape in args.shapes.split(','):
        n, k = (int(v) for v in shape.split('x'))
        for t in (int(v) for v in args.tokens.split(',')):
            seed = n + k + t
            wa = lambda op: (lambda kern: lambda: kern.gemm(op['wp'], op['wsf'], op['xp'], op['xsf'], n, t, k,  # noqa: E731
                                                           scale_m_default=op['gsw'], scale_n=op['gsx'], check=False))
            wb = lambda op: (lambda kern: lambda: kern.gemm(op['xp'], op['xsf'], op['wp'], op['wsf'], t, n, k,  # noqa: E731
                                                           scale_m=op['gsx'], scale_n_default=op['gsw'], check=False))
            base = operands(n, k, t, None, None, seed=seed)
            configs = [('stock@table', None, 0.0, wa(base)(stock.pick(n, k, t)), stock.width(n, k, t)),
                       ('stock@128', None, 0.0, wa(base)(s128), 128),
                       ('stock_wB', None, 0.0, wb(base)(swb), None),
                       ('nodisp@128', None, 0.0, wa(base)(nodisp), 128)]
            for pattern in PATTERNS:
                for f in fractions:
                    op16 = operands(n, k, t, tags((-(-n // 16), k // 64), f, pattern, seed), (16, 64), seed=seed)
                    op8 = operands(n, k, t, tags((-(-n // 8), k // 64), f, pattern, seed + 1), (8, 64), seed=seed)
                    configs += [('mixed@table', pattern, f, wa(op16)(mixed.pick(n, k, t)), mixed.width(n, k, t)),
                                ('mixed@128', pattern, f, wa(op16)(k128), 128),
                                ('n8k64_wB', pattern, f, wb(op8)(n8), None)]
            times = {i: [] for i in range(len(configs))}
            for r in range(args.rounds):
                shift = (r * len(configs) // args.rounds) % len(configs)
                for i in list(range(shift, len(configs))) + list(range(shift)):
                    times[i].append(gemm_us(configs[i][3]))
            for i, (kern, pattern, f, _, width) in enumerate(configs):
                res['rows'].append(dict(shape=shape, out=n, inp=k, tokens=t, kernel=kern, pattern=pattern, fraction=f,
                                        width=width, time_us=statistics.median(times[i]), rounds=times[i]))
            print(shape, t, ' '.join(f"{c[0]}:{c[1] or '-'}:{c[2]:g}={statistics.median(times[i]):.1f}"
                                     for i, c in enumerate(configs) if c[2] in (0.0, 0.5, 1.0)), flush=True)
            B.write(args.out, res)
            torch.cuda.empty_cache()
    res['status'] = 'complete'
    res['gpu_end'] = B.gpu_info()
    B.write(args.out, res)


if __name__ == '__main__':
    main()

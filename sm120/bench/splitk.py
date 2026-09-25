#!/usr/bin/env python3
"""Decomposition sweep: which K split makes small-token GEMMs fill the GPU (section 10).

With weights on A the output has ceil(out/128) x ceil(T/128) CTA tiles; for decode and small
batches that is 8-112 tiles on a 170-SM RTX 5090, and each CTA then streams its whole weight panel
alone. Times (CUPTI, summed over every kernel the call launches, i.e. including any separate
reduction) the data-parallel kernel against the Stream-K build under its heuristic, Stream-K and
split-K 2..8, per real shape and token count, for the mixed kernel (selector map tags) and the
stock NVFP4 kernel.

    python sm120/bench/splitk.py --out sm120/results/bench/splitk_rtx5090.json
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as B  # noqa: E402
from kernel import operands, selector_masks  # noqa: E402

from mixfp4_sm120.lib import Kernel  # noqa: E402

VARIANTS = [('dp', None, 1, 0), ('sk_heuristic', 'sk', 1, 0), ('stream_k', 'sk', 1, 3)] + \
           [(f'split{s}', 'sk', s, 2) for s in (2, 3, 4, 6, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='llama8b,qwen4b')
    ap.add_argument('--tokens', default='1,8,16,32,64,128,256,512,2048')
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-busy', action='store_true')
    args = ap.parse_args()
    if not args.allow_busy:
        B.require_idle()
    pairs = {'mixed': (Kernel.load('n16k64_wA'), Kernel.load('n16k64_wA_sk')),
             'stock': (Kernel.load('stock_wA'), Kernel.load('stock_wA_sk'))}
    res = dict(gpu=B.gpu_info(), kernels={k.cfg.name: k.sha256 for p in pairs.values() for k in p}, rows=[])
    for model in args.models.split(','):
        sel = selector_masks(model, 16)
        for proj, (n, k) in B.MODEL_SHAPES[model].items():
            for t in [int(v) for v in args.tokens.split(',')]:
                for fam, (dp, sk) in pairs.items():
                    mask = sel[proj][0] if (fam == 'mixed' and proj in sel) else None
                    op = operands(n, k, t, mask, (16, 64) if mask is not None else None, seed=n + k + t)
                    for vname, which, splits, mode in VARIANTS:
                        kern = dp if which is None else sk
                        fn = lambda: kern.gemm(op['wp'], op['wsf'], op['xp'], op['xsf'], n, t, k,  # noqa: E731
                                               scale_m_default=op['gsw'], scale_n=op['gsx'], check=False,
                                               splits=splits, decomposition=mode)
                        kt = B.kernel_times(fn, iters=20)
                        res['rows'].append(dict(model=model, proj=proj, out=n, inp=k, tokens=t, family=fam, variant=vname,
                                                us=sum(v['us'] for v in kt.values()), kernels=len(kt)))
                print(model, proj, t, 'done', flush=True)
                B.write(args.out, res)
    B.write(args.out, res)


if __name__ == '__main__':
    main()

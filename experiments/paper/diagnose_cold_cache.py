#!/usr/bin/env python3
"""DIAGNOSTIC (not a result; awaiting approval): why rotation and the 512 MiB flush disagree at Llama q_proj T = 2048.

Conditions, each an isolated CUPTI-timed GEMM as in check_l2_cold.py, 3 rotated rounds x 50:
  cold512        the registered method (flush, one weight copy)
  rotation       no flush, K distinct copies (K x size >= 4x L2)
  rot_flush      the flush AND the next distinct copy (both cold mechanisms)
  cold512_idle   the flush, then a 2 ms host idle gap before the activation copy / quantizer / GEMM
Readings: rot_flush ~ cold512 -> clock state (H2); rot_flush ~ rotation -> address diversity (H3).
cold512_idle tests H2 directly (an idle gap after the flush should push it toward rotation).
"""
import math
import statistics
import sys
import time
import dataclasses
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, '/home/dev/NVFP4-RaZeR/experiments/paper')
from bench_gemm_isolated import A, B, KernelSet, NativeLinear, classify, stats, tag_modules  # noqa: E402

ART = '/home/dev/n16k64_campaign/paper/artifacts'
CONDS = ('cold512', 'rotation', 'rot_flush', 'cold512_idle')


def main(out, reps=50, rounds=3, projs=('q_proj',), tokens=(2048,)):
    B.require_idle()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    ks = dict(stock=KernelSet('stock'), mixed=KernelSet('mixed'))
    flush = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
    res = []
    for kind, kset, variant in (('fo6', 'stock', 'first'), ('tc_16x64', 'mixed', 'typical')):
        meta, w = A.load(f'{ART}/llama8b_{kind}', device='cpu')
        tags = tag_modules(meta, w)
        for proj in projs:
            pw = w[tags[proj][variant][0]]
            pw = dataclasses.replace(pw, packed=pw.packed.cuda(), scales=pw.scales.cuda(), bias=None)
            lin = NativeLinear(pw, ks[kset], 'four_over_six_rows')
            n, k = lin.out_features, lin.in_features
            size = lin.packed.numel() + lin.sf.numel()
            copies = math.ceil(4 * l2 / size) + 1
            rot = [(lin.packed.clone(), lin.sf.clone()) for _ in range(copies)]
            for t in tokens:
                kern = lin.kernel_set.pick(n, k, t)
                g = torch.Generator('cpu').manual_seed(n + k + t)
                pool = [torch.randn(t, k, generator=g).to('cuda', torch.bfloat16) for _ in range(2)]
                x = torch.empty_like(pool[0])
                y = torch.empty((t, n), dtype=torch.bfloat16, device='cuda')
                stream = torch.cuda.current_stream().cuda_stream

                def rep(cond, i):
                    if cond in ('cold512', 'rot_flush', 'cold512_idle'):
                        flush.sum()
                        torch.cuda.synchronize()
                    if cond == 'cold512_idle':
                        time.sleep(0.002)
                    x.copy_(pool[i % 2])
                    torch.cuda.synchronize()
                    xp, xsf, gs = kern.quant_rows(x, 'four_over_six_rows')
                    torch.cuda.synchronize()
                    packed, sf = rot[i % copies] if cond in ('rotation', 'rot_flush') else (lin.packed, lin.sf)
                    kern.gemm_ptr(packed.data_ptr(), sf.data_ptr(), xp.data_ptr(), xsf.data_ptr(), n, t, k, None,
                                  lin.global_scale, gs.data_ptr(), 1.0, None, y, stream)
                    torch.cuda.synchronize()

                per = {c: [] for c in CONDS}
                for r in range(rounds):
                    shift = (r * len(CONDS) // rounds) % len(CONDS)
                    for cond in CONDS[shift:] + CONDS[:shift]:
                        for i in range(copies):
                            rep(cond, i)
                        with profile(activities=[ProfilerActivity.CUDA]) as prof:
                            for i in range(reps):
                                rep(cond, i)
                        d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and classify(e.name) == 'gemm']
                        assert len(d) == reps, (cond, len(d))
                        per[cond] += d
                med = {c: statistics.median(v) for c, v in per.items()}
                rec = dict(kind=kind, proj=proj, tokens=t, kernel=kern.cfg.name, stats={c: stats(v) for c, v in per.items()})
                res.append(rec)
                print(kind, proj, t, kern.cfg.name, ' '.join(f'{c} {med[c]:.1f}' for c in CONDS), flush=True)
    B.write(out, dict(diagnostic=True, cases=res, gpu=B.gpu_info()))


if __name__ == '__main__':
    main(sys.argv[1])

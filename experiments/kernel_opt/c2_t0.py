#!/usr/bin/env python3
"""Kernel-opt amendment 6 (t0), measurement C2‴: the 4096^3 breakdown, with and without the site-0 prmt tags.

    python experiments/kernel_opt/c2_t0.py --t0-root DIR --freq-root DIR --t0freq-root DIR --out JSON

Operands as C2' (c2_freq.py): weights N(0, 0.02) seed 4096, 4096 tokens N(0, 1) FourOverSix per token; the real map =
model.layers.0.self_attn.o_proj of the Llama-3.1-8B 16x64 TC map. Configurations (GEMM only, weights on A):
  stock_wA                              the reference (sm120/build)
  nodisp, nodisp_t0                     the no-dispatch ceiling: sm120/build / --t0-root
  {default,t0}_{e2m1,real,e0m3}         n16k64_wA from sm120/build / n16k64_wA_t0 from --t0-root
  {freq,freqt0}_{e2m1,real,e0m3}        n16k64_wA from --freq-root (#2) / n16k64_wA_t0 from --t0freq-root
  all-E2M1 / real / all-E0M3 tags.
Modes as C2': b2b (20 back-to-back calls), isolated (15 single calls after a 50 ms gap), sustained (2 s at the power
cap, NVML), each 3 rotated rounds. Checks before timing: every t0 configuration's output equals its tagged
counterpart's bitwise, and nodisp_t0's equals nodisp's.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
sys.path.insert(0, str(REPO / 'experiments' / 'kernel_opt'))
import common as B  # noqa: E402
from c2_freq import LAYER, MAPS, Sampler, gemm_times  # noqa: E402
from kernel import place  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402

PAIRS = [('nodisp', 'nodisp_t0')] + [(f'{a}_{tg}', f'{b}_{tg}') for a, b in (('default', 't0'), ('freq', 'freqt0'))
                                      for tg in ('e2m1', 'real', 'e0m3')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--t0-root', required=True)
    ap.add_argument('--freq-root', required=True)
    ap.add_argument('--t0freq-root', required=True)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    n = k = t = 4096
    g = torch.Generator('cpu').manual_seed(4096)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
    xp, xsf = N.pack_nibbles(xn), place(xsb, k)
    real = mapio.read_map(MAPS[16])[1][LAYER]

    def wt(kind, mask=None):
        wn, wsb, gsw = N.quantize_weight(w, kind, mask, (16, 64) if mask is not None else None)
        return N.pack_nibbles(wn), place(wsb, k), float(gsw)
    K = dict(stock_wA=Kernel.load('stock_wA'), nodisp=Kernel.load('n16k64_wA_nodisp'),
             nodisp_t0=Kernel.load('n16k64_wA_nodisp_t0', build_root=args.t0_root), default=Kernel.load('n16k64_wA'),
             t0=Kernel.load('n16k64_wA_t0', build_root=args.t0_root), freq=Kernel.load('n16k64_wA', build_root=args.freq_root),
             freqt0=Kernel.load('n16k64_wA_t0', build_root=args.t0freq_root))
    for kk in ('freq', 'freqt0'):
        assert K[kk].manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1
    grid = (n // 16, k // 64)
    tags = dict(e2m1=wt('map', torch.zeros(grid, dtype=torch.bool)), real=wt('map', real),
                e0m3=wt('map', torch.ones(grid, dtype=torch.bool)))
    cfgs = [('stock_wA', 'stock_wA', wt('nvfp4')), ('nodisp', 'nodisp', wt('four_over_six')),
            ('nodisp_t0', 'nodisp_t0', wt('four_over_six'))]
    for tg in ('e2m1', 'real', 'e0m3'):
        cfgs += [(f'{kk}_{tg}', kk, tags[tg]) for kk in ('default', 't0', 'freq', 'freqt0')]

    def call(kk, wts):
        wp, wsf, gsw = wts
        kern = K[kk]
        return lambda: kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gsw, scale_n=gsx, check=False)
    calls = {name: call(kk, wts) for name, kk, wts in cfgs}
    res = dict(protocol='results/kernel_opt/PROTOCOL.md, amendment 6 (C2‴)', gpu=B.gpu_info(), size=4096,
               kernels={kk: dict(sha256=v.sha256, root=str(v.path.parent.parent)) for kk, v in K.items()},
               checks={}, modes={})
    for a, b in PAIRS:
        ya, yb = calls[a](), calls[b]()
        torch.cuda.synchronize()
        ok = bool(torch.equal(ya.view(torch.int16), yb.view(torch.int16)))
        res['checks'][f'{b} == {a}'] = ok
        if not ok:
            B.write(args.out, res)
            raise SystemExit(f'{b} differs from {a}')
    names = [c[0] for c in cfgs]
    for mode in ('b2b', 'isolated', 'sustained'):
        per = {nm: [] for nm in names}
        tel = {nm: [] for nm in names}
        for r in range(args.rounds):
            shift = (r * len(names) // args.rounds) % len(names)
            for nm in names[shift:] + names[:shift]:
                fn = calls[nm]
                fn()
                torch.cuda.synchronize()
                if mode == 'b2b':
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(20):
                            fn()
                        torch.cuda.synchronize()
                    per[nm].append(statistics.median(gemm_times(prof)))
                elif mode == 'isolated':
                    iso = []
                    for _ in range(15):
                        time.sleep(0.05)
                        with profile(activities=[ProfilerActivity.CUDA]) as prof:
                            fn()
                            torch.cuda.synchronize()
                        iso += gemm_times(prof)
                    per[nm].append(statistics.median(iso))
                else:
                    evs = []
                    with Sampler() as smp:
                        t0_ = time.time()
                        half, end = t0_ + 1.0, t0_ + 2.0
                        while time.time() < end:
                            e0_, e1_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            e0_.record()
                            fn()
                            e1_.record()
                            evs.append((time.time(), e0_, e1_))
                            if len(evs) % 64 == 0:
                                torch.cuda.synchronize()
                        torch.cuda.synchronize()
                    per[nm].append(statistics.median(a.elapsed_time(b) * 1e3 for (tt, a, b) in evs if tt >= half))
                    tel[nm].append(smp.summary(half))
                print(f'{mode} r{r} {nm:16s} {per[nm][-1]:8.1f} us', flush=True)
        res['modes'][mode] = {nm: dict(us=statistics.median(per[nm]), rounds=per[nm], telemetry=tel[nm]) for nm in names}
        B.write(args.out, res)
    summ = {}
    for mode, d in res['modes'].items():
        u = {nm: v['us'] for nm, v in d.items()}
        pc = lambda a, b: 100 * (u[a] / u[b] - 1)  # noqa: E731
        summ[mode] = {'nodisp vs stock_wA': pc('nodisp', 'stock_wA'), 'nodisp_t0 vs stock_wA': pc('nodisp_t0', 'stock_wA'),
                      **{f'{b} vs {a}': pc(b, a) for a, b in PAIRS},
                      **{f'{kk}_{tg} vs nodisp_t0': pc(f'{kk}_{tg}', 'nodisp_t0') for kk in ('default', 't0', 'freq', 'freqt0')
                         for tg in ('e2m1', 'real', 'e0m3')},
                      **{f'{kk}_{tg} vs stock_wA': pc(f'{kk}_{tg}', 'stock_wA') for kk in ('default', 't0', 'freq', 'freqt0')
                         for tg in ('e2m1', 'real', 'e0m3')}}
    res['summary'] = summ
    res['status'] = 'complete'
    B.write(args.out, res)
    for mode, s in summ.items():
        print(mode, ' '.join(f'{k_}={v:+.2f}%' for k_, v in s.items() if ' vs ' in k_ and ('t0' in k_ or 'nodisp' in k_)))


if __name__ == '__main__':
    main()

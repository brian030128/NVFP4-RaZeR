#!/usr/bin/env python3
"""Kernel-opt amendment 3 (A'), measurement C2: the 4096^3 breakdown, today's n16k64_wA vs the 4-arm n16k64_wA_g32.

    python experiments/kernel_opt/c2_g32.py --a1-root DIR --out JSON

Operands as C2 (experiments/paper_extra/C2_time.py) and C2' (c2_freq.py): weights N(0, 0.02) seed 4096, 4096 tokens
N(0, 1) FourOverSix per token; the real map = model.layers.0.self_attn.o_proj of the Llama-3.1-8B TC 256x64 map (stored
as 16x64 granules). Configurations (GEMM only, the 128-wide builds):
  stock_wA, nodisp                  C2's references (sm120/build)
  default_{e2m1,real,e0m3}          n16k64_wA from sm120/build: today's path for 256x64 maps (16 arms)
  g32_{e2m1,real,e0m3}              n16k64_wA_g32 from --a1-root (4 arms), the same weights
Modes, one after the other (as c2_freq.py): b2b (C2's method: CUPTI over 20 back-to-back calls), isolated (15 single
calls after a 50 ms idle gap each, CUPTI), sustained (back-to-back for 2 s, CUDA events over the last half, NVML); 3
rotated rounds each, the median of the per-round medians.
Check before timing: every g32 configuration's output equals its default's bitwise.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments' / 'kernel_opt'))
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402
from c2_freq import Sampler, gemm_times  # noqa: E402
from kernel import place  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402

LAYER = 'model.layers.0.self_attn.o_proj'
MAP = Path('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_256x64.mixfp4map')
TAGS = ('e2m1', 'real', 'e0m3')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a1-root', required=True)
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
    header, masks, _ = mapio.read_map(MAP)
    assert tuple(header['type_block']) == (16, 64), header['type_block']
    grid = (n // 16, k // 64)
    real = masks[LAYER]
    assert tuple(real.shape) == grid
    assert torch.equal(real, real.reshape(n // 256, 16, k // 64).amax(1).repeat_interleave(16, 0)), 'not a 256x64 map'

    def wt(kind, mask=None):
        wn, wsb, gsw = N.quantize_weight(w, kind, mask, (16, 64) if mask is not None else None)
        return N.pack_nibbles(wn), place(wsb, k), float(gsw)
    K = dict(stock_wA=Kernel.load('stock_wA'), nodisp=Kernel.load('n16k64_wA_nodisp'), default=Kernel.load('n16k64_wA'),
             g32=Kernel.load('n16k64_wA_g32', build_root=args.a1_root))
    paper = REPO / 'sm120' / 'build'
    for kk in ('stock_wA', 'nodisp', 'default'):
        assert Path(K[kk].path).parent.parent == paper, K[kk].path
    assert K['g32'].cfg.map_tile_rows == 128 and tuple(K['g32'].type_block) == (32, 64)
    tags = dict(e2m1=wt('map', torch.zeros(grid, dtype=torch.bool)), real=wt('map', real),
                e0m3=wt('map', torch.ones(grid, dtype=torch.bool)))
    cfgs = [('stock_wA', 'stock_wA', wt('nvfp4')), ('nodisp', 'nodisp', wt('four_over_six'))]
    for tg in TAGS:
        cfgs += [(f'default_{tg}', 'default', tags[tg]), (f'g32_{tg}', 'g32', tags[tg])]

    def call(kk, wts):
        wp, wsf, gsw = wts
        kern = K[kk]
        return lambda: kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gsw, scale_n=gsx, check=False)
    calls = {name: call(kk, wts) for name, kk, wts in cfgs}
    res = dict(protocol="results/kernel_opt/PROTOCOL.md, amendment 3 (A', C2)", gpu=B.gpu_info(), size=4096,
               real_map=dict(file=str(MAP), layer=LAYER, e0m3_granules=int(real.sum()), granules=real.numel()),
               kernels={kk: dict(sha256=v.sha256, root=str(Path(v.path).parent.parent)) for kk, v in K.items()},
               checks={}, modes={})
    for tg in TAGS:
        ya, yb = calls[f'default_{tg}'](), calls[f'g32_{tg}']()
        torch.cuda.synchronize()
        ok = bool(torch.equal(ya.view(torch.int16), yb.view(torch.int16)))
        res['checks'][f'g32_{tg} == default_{tg}'] = ok
        if not ok:
            B.write(args.out, res)
            raise SystemExit(f'g32_{tg} differs from default_{tg}')
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
                        t0 = time.time()
                        half, end = t0 + 1.0, t0 + 2.0
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
                print(f'{mode} r{r} {nm:14s} {per[nm][-1]:8.1f} us', flush=True)
        res['modes'][mode] = {nm: dict(us=statistics.median(per[nm]), rounds=per[nm], telemetry=tel[nm]) for nm in names}
        B.write(args.out, res)
    summ = {}
    for mode, d in res['modes'].items():
        u = {nm: v['us'] for nm, v in d.items()}
        summ[mode] = {**{f'g32 vs default, {tg}': 100 * (u[f'g32_{tg}'] / u[f'default_{tg}'] - 1) for tg in TAGS},
                      **{f'{kk}_{tg} vs nodisp': 100 * (u[f'{kk}_{tg}'] / u['nodisp'] - 1) for kk in ('default', 'g32') for tg in TAGS},
                      **{f'{kk}_{tg} vs stock_wA': 100 * (u[f'{kk}_{tg}'] / u['stock_wA'] - 1) for kk in ('default', 'g32') for tg in TAGS},
                      'nodisp vs stock_wA': 100 * (u['nodisp'] / u['stock_wA'] - 1)}
    res['summary'] = summ
    res['status'] = 'complete'
    B.write(args.out, res)
    for mode, s in summ.items():
        print(mode, ' '.join(f'{k_}={v:+.2f}%' for k_, v in s.items() if 'g32 vs default' in k_))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Kernel-opt amendment 11 (the 8x64 plan's P2), measurement C2w‴: amendment 10's 4096^3 breakdown of the weights-on-B
kernel (c2_w8.py), with P2's t0 builds added.

    python experiments/kernel_opt/c2_w8p2.py --kopt-root DIR --freq-root DIR --t0-root DIR --t0freq-root DIR --w-root DIR \
        --b7 DIR --out JSON

Operands as C2' (c2_freq.py): weights N(0, 0.02) seed 4096, 4096 tokens N(0, 1) FourOverSix per token; the real map =
model.layers.0.self_attn.o_proj of the Llama-3.1-8B 8x64 TC map. Configurations (GEMM only):
  stock_wA                         the paper stock, weights on A (sm120/build)
  stock_ko                         the deployment's stock build at this shape and T (stock_ko from --b7, the adopted
                                   table's width and scheduler row), weights on A
  stock_wB                         stock, weights on B (sm120/build)
  nodisp_wB, nodisp_wB_t0          n8k64_wB's no-dispatch ceiling (1x8, E2M1 only), with / without the site-0 prmt
                                   tags (--w-root)
  wBdefault_{e2m1,real,e0m3}       n8k64_wB from --kopt-root (the 1b set's width-128 build), all-E2M1 / real / all-E0M3
  wBfreq_{e2m1,real,e0m3}          n8k64_wB from --freq-root (#2's dispatch)
  wBt0_{e2m1,real,e0m3}            n8k64_wB_t0 from --t0-root (P2: no site-0 prmt tags, default dispatch)
  wBt0freq_{e2m1,real,e0m3}        n8k64_wB_t0 from --t0freq-root (P2, #2's dispatch)
The stock configurations take NVFP4 weights and the ceilings FourOverSix weights (all E2M1), as C2'.
Modes as C2': b2b (20 back-to-back calls), isolated (15 single calls after a 50 ms gap), sustained (2 s, NVML), each 3
rotated rounds. Checks before timing: nodisp_wB_t0's output equals nodisp_wB's bitwise, and every wBfreq, wBt0 and
wBt0freq configuration's equals its wBdefault's.
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
from mixfp4_sm120 import select as S  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

TAGS = ('e2m1', 'real', 'e0m3')
BUILDS = ('wBdefault', 'wBfreq', 'wBt0', 'wBt0freq')
PAIRS = [('nodisp_wB', 'nodisp_wB_t0')] + [(f'wBdefault_{tg}', f'{kk}_{tg}') for tg in TAGS for kk in BUILDS[1:]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kopt-root', required=True)
    ap.add_argument('--freq-root', required=True)
    ap.add_argument('--t0-root', required=True)
    ap.add_argument('--t0freq-root', required=True)
    ap.add_argument('--w-root', required=True)
    ap.add_argument('--b7', required=True)
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
    real = mapio.read_map(MAPS[8])[1][LAYER]

    def wt(kind, mask=None):
        wn, wsb, gsw = N.quantize_weight(w, kind, mask, (8, 64) if mask is not None else None)
        return N.pack_nibbles(wn), place(wsb, k), float(gsw)
    sko = KernelSet('stock_ko', build_root=args.b7, table=S.TABLE_DIR / f'{S.gpu_slug()}.ko.json')
    sched = tuple(sko.schedule(n, k, t))
    K = dict(stock_wA=Kernel.load('stock_wA'), stock_ko=sko.pick(n, k, t), stock_wB=Kernel.load('stock_wB'),
             nodisp_wB=Kernel.load('n8k64_wB_nodisp', build_root=args.w_root),
             nodisp_wB_t0=Kernel.load('n8k64_wB_nodisp_t0', build_root=args.w_root),
             wBdefault=Kernel.load('n8k64_wB', build_root=args.kopt_root),
             wBfreq=Kernel.load('n8k64_wB', build_root=args.freq_root),
             wBt0=Kernel.load('n8k64_wB_t0', build_root=args.t0_root),
             wBt0freq=Kernel.load('n8k64_wB_t0', build_root=args.t0freq_root))
    for kk in ('wBfreq', 'wBt0freq'):
        assert K[kk].manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1, kk
    for kk in ('wBdefault', 'wBt0'):
        assert 'MIXFP4_DISPATCH_FREQ' not in (K[kk].manifest.get('extra_defines') or {}), kk
    for kk in ('wBt0', 'wBt0freq'):
        assert str(K[kk].manifest['blob_gen'].get('TAG0')) == '0', kk
    grid = (n // 8, k // 64)
    tags = dict(e2m1=wt('map', torch.zeros(grid, dtype=torch.bool)), real=wt('map', real),
                e0m3=wt('map', torch.ones(grid, dtype=torch.bool)))
    # (name, kernel, weights, weights on B, scheduler setting)
    cfgs = [('stock_wA', 'stock_wA', wt('nvfp4'), False, None), ('stock_ko', 'stock_ko', wt('nvfp4'), False, sched),
            ('stock_wB', 'stock_wB', wt('nvfp4'), True, None),
            ('nodisp_wB', 'nodisp_wB', wt('four_over_six'), True, None),
            ('nodisp_wB_t0', 'nodisp_wB_t0', wt('four_over_six'), True, None)]
    for tg in TAGS:
        cfgs += [(f'{kk}_{tg}', kk, tags[tg], True, None) for kk in BUILDS]

    def call(kk, wts, on_b, sch):
        wp, wsf, gsw = wts
        kern = K[kk]
        if not on_b:
            return lambda: kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gsw, scale_n=gsx, check=False, schedule=sch)
        return lambda: kern.gemm(xp, xsf, wp, wsf, t, n, k, scale_m=gsx, scale_n_default=gsw, check=False)
    calls = {name: call(kk, wts, on_b, sch) for name, kk, wts, on_b, sch in cfgs}
    res = dict(protocol='results/kernel_opt/PROTOCOL.md, amendment 11 (C2w‴)', gpu=B.gpu_info(), size=4096,
               kernels={kk: dict(config=v.cfg.name, sha256=v.sha256, root=str(v.path.parent.parent)) for kk, v in K.items()},
               stock_ko=dict(width=sko.width(n, k, t), schedule=list(sched)), checks={}, modes={})
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
                print(f'{mode} r{r} {nm:18s} {per[nm][-1]:8.1f} us', flush=True)
        res['modes'][mode] = {nm: dict(us=statistics.median(per[nm]), rounds=per[nm], telemetry=tel[nm]) for nm in names}
        B.write(args.out, res)
    summ = {}
    for mode, d in res['modes'].items():
        u = {nm: v['us'] for nm, v in d.items()}
        pc = lambda a, b: 100 * (u[a] / u[b] - 1)  # noqa: E731
        summ[mode] = {'stock_wB vs stock_wA': pc('stock_wB', 'stock_wA'), 'stock_ko vs stock_wA': pc('stock_ko', 'stock_wA'),
                      'stock_wB vs stock_ko': pc('stock_wB', 'stock_ko'),
                      'nodisp_wB vs stock_wB': pc('nodisp_wB', 'stock_wB'),
                      'nodisp_wB_t0 vs stock_wB': pc('nodisp_wB_t0', 'stock_wB'),
                      'nodisp_wB_t0 vs nodisp_wB': pc('nodisp_wB_t0', 'nodisp_wB'),
                      **{f'{kk}_{tg} vs stock_ko': pc(f'{kk}_{tg}', 'stock_ko') for kk in BUILDS for tg in TAGS},
                      **{f'{kk}_{tg} vs stock_wB': pc(f'{kk}_{tg}', 'stock_wB') for kk in BUILDS for tg in TAGS},
                      **{f'{kk}_{tg} vs nodisp_wB': pc(f'{kk}_{tg}', 'nodisp_wB') for kk in ('wBdefault', 'wBfreq') for tg in TAGS},
                      **{f'{kk}_{tg} vs nodisp_wB_t0': pc(f'{kk}_{tg}', 'nodisp_wB_t0') for kk in ('wBt0', 'wBt0freq') for tg in TAGS},
                      **{f't0 vs default, {tg}': pc(f'wBt0_{tg}', f'wBdefault_{tg}') for tg in TAGS},
                      **{f't0 #2 vs #2, {tg}': pc(f'wBt0freq_{tg}', f'wBfreq_{tg}') for tg in TAGS},
                      **{f"#2's vs default, {tg}": pc(f'wBfreq_{tg}', f'wBdefault_{tg}') for tg in TAGS},
                      **{f"t0: #2's vs default, {tg}": pc(f'wBt0freq_{tg}', f'wBt0_{tg}') for tg in TAGS}}
    res['summary'] = summ
    res['status'] = 'complete'
    B.write(args.out, res)
    for mode, s in summ.items():
        print(mode, ' '.join(f'{k_}={v:+.2f}%' for k_, v in s.items() if 'vs stock_ko' in k_ or ' vs ' in k_[:12]))


if __name__ == '__main__':
    main()

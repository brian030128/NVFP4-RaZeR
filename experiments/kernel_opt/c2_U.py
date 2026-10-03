#!/usr/bin/env python3
"""Kernel-opt amendment 17 (U: the uniform-branch dispatch, part A), measurement C2U: the 4096^3 breakdown (C2' / C2w‴'s
method) of both families' width-128 builds, today's against amendment 17's.

    python experiments/kernel_opt/c2_U.py --u-root DIR --b7 DIR --b7freq DIR --p3freq DIR --c3k DIR --p5 DIR --out JSON

Operands as C2' (c2_freq.py): weights N(0, 0.02) seed 4096, 4096 tokens N(0, 1) FourOverSix per token; the real maps =
model.layers.0.self_attn.o_proj of Llama-3.1-8B's TC 16x64 and 8x64 maps. Configurations (GEMM only):
  stock_ko                  the deployment's stock build at this shape and T (stock_ko from --b7: the adopted table's
                            width and scheduler row), weights on A, NVFP4 weights
  ceil16                    n16k64_wA_nodisp_e64_t0 (--c3k): the 16x64 width-128 tile with the dispatch compiled out
  ceil8                     n8k64_wB_nodisp_t0 (--p5): the 8x64 width-128 tile with the dispatch compiled out
  A16_{e2m1,real,e0m3}      n16k64_wA_e64_t0 from --b7freq: today's adopted 16x64 width-128 build (t0, #2's dispatch)
  U16_{e2m1,real,e0m3}      n16k64_wA_e64_t0 from --u-root: + MIXFP4_UNIFORM_DISPATCH=1
  A8_{e2m1,real,e0m3}       n8k64_wB_t0 from --p3freq: today's adopted 8x64 width-128 build (t0, #2's dispatch)
  U8_{e2m1,real,e0m3}       n8k64_wB_t0 from --u-root: + MIXFP4_PIPE_FLAGS=1
The ceilings take FourOverSix weights (all E2M1), as C2'. The mixed builds run with the default scheduler setting, as
in C2w‴ (their table rows at 4096 x 4096, T = 4096 have none). Modes as C2': b2b (20 back-to-back calls), isolated (15
single calls after a 50 ms gap), sustained (2 s, NVML), each 3 rotated rounds. Checks before timing: every U
configuration's output equals its A configuration's bitwise, and the defines are as registered.
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
PAIRS = [(f'A{u}_{tg}', f'U{u}_{tg}') for u in ('16', '8') for tg in TAGS]
DEFINES = dict(A16={'MIXFP4_DISPATCH_FREQ': 1}, U16={'MIXFP4_DISPATCH_FREQ': 1, 'MIXFP4_UNIFORM_DISPATCH': 1},
               A8={'MIXFP4_DISPATCH_FREQ': 1}, U8={'MIXFP4_DISPATCH_FREQ': 1, 'MIXFP4_PIPE_FLAGS': 1})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--u-root', required=True)
    ap.add_argument('--b7', required=True)
    ap.add_argument('--b7freq', required=True)
    ap.add_argument('--p3freq', required=True)
    ap.add_argument('--c3k', required=True)
    ap.add_argument('--p5', required=True)
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
    real = {u: mapio.read_map(MAPS[u])[1][LAYER] for u in (16, 8)}

    def wt(kind, mask=None, rows=None):
        wn, wsb, gsw = N.quantize_weight(w, kind, mask, (rows, 64) if mask is not None else None)
        return N.pack_nibbles(wn), place(wsb, k), float(gsw)
    sko = KernelSet('stock_ko', build_root=args.b7, table=S.TABLE_DIR / f'{S.gpu_slug()}.ko.json')
    sched = tuple(sko.schedule(n, k, t))
    K = dict(stock_ko=sko.pick(n, k, t),
             ceil16=Kernel.load('n16k64_wA_nodisp_e64_t0', build_root=args.c3k),
             ceil8=Kernel.load('n8k64_wB_nodisp_t0', build_root=args.p5),
             A16=Kernel.load('n16k64_wA_e64_t0', build_root=args.b7freq),
             U16=Kernel.load('n16k64_wA_e64_t0', build_root=args.u_root),
             A8=Kernel.load('n8k64_wB_t0', build_root=args.p3freq),
             U8=Kernel.load('n8k64_wB_t0', build_root=args.u_root))
    for kk, want in DEFINES.items():
        assert (K[kk].manifest.get('extra_defines') or {}) == want, (kk, K[kk].manifest.get('extra_defines'))
    tags = {}
    for u, rows in (('16', 16), ('8', 8)):
        grid = (n // rows, k // 64)
        tags[u] = dict(e2m1=wt('map', torch.zeros(grid, dtype=torch.bool), rows), real=wt('map', real[rows], rows),
                       e0m3=wt('map', torch.ones(grid, dtype=torch.bool), rows))
    # (name, kernel, weights, weights on B, scheduler setting)
    cfgs = [('stock_ko', 'stock_ko', wt('nvfp4'), False, sched), ('ceil16', 'ceil16', wt('four_over_six'), False, None),
            ('ceil8', 'ceil8', wt('four_over_six'), True, None)]
    for u in ('16', '8'):
        for tg in TAGS:
            cfgs += [(f'{v}{u}_{tg}', f'{v}{u}', tags[u][tg], u == '8', None) for v in ('A', 'U')]

    def call(kk, wts, on_b, sch):
        wp, wsf, gsw = wts
        kern = K[kk]
        if not on_b:
            return lambda: kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gsw, scale_n=gsx, check=False, schedule=sch)
        return lambda: kern.gemm(xp, xsf, wp, wsf, t, n, k, scale_m=gsx, scale_n_default=gsw, check=False)
    calls = {name: call(kk, wts, on_b, sch) for name, kk, wts, on_b, sch in cfgs}
    res = dict(protocol='results/kernel_opt/PROTOCOL.md, amendment 17 (C2U)', gpu=B.gpu_info(), size=4096,
               kernels={kk: dict(config=v.cfg.name, sha256=v.sha256, root=str(v.path.parent.parent),
                                 defines=v.manifest.get('extra_defines')) for kk, v in K.items()},
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
                print(f'{mode} r{r} {nm:12s} {per[nm][-1]:8.1f} us', flush=True)
        res['modes'][mode] = {nm: dict(us=statistics.median(per[nm]), rounds=per[nm], telemetry=tel[nm]) for nm in names}
        B.write(args.out, res)
    summ = {}
    for mode, d in res['modes'].items():
        u_ = {nm: v['us'] for nm, v in d.items()}
        pc = lambda a, b: 100 * (u_[a] / u_[b] - 1)  # noqa: E731
        s = {'ceil16 vs stock_ko': pc('ceil16', 'stock_ko'), 'ceil8 vs stock_ko': pc('ceil8', 'stock_ko')}
        for u in ('16', '8'):
            for tg in TAGS:
                s[f'U{u} vs A{u}, {tg}'] = pc(f'U{u}_{tg}', f'A{u}_{tg}')
                for v in ('A', 'U'):
                    s[f'{v}{u}_{tg} vs stock_ko'] = pc(f'{v}{u}_{tg}', 'stock_ko')
                    s[f'{v}{u}_{tg} vs ceil{u}'] = pc(f'{v}{u}_{tg}', f'ceil{u}')
        summ[mode] = s
    res['summary'] = summ
    res['status'] = 'complete'
    B.write(args.out, res)
    for mode, s in summ.items():
        print(mode, ' '.join(f'{k_}={v:+.2f}%' for k_, v in s.items() if ' vs A' in k_))


if __name__ == '__main__':
    main()

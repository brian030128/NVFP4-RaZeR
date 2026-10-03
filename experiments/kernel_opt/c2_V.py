#!/usr/bin/env python3
"""Kernel-opt amendment 18 (V), measurement C2V: the 4096^3 breakdown (C2' / C2U's method) of the 256x64 path's width-128
builds, today's against amendment 18's, with the no-dispatch ceiling.

    python experiments/kernel_opt/c2_V.py --v-root DIR --a1 DIR --b7 DIR --c3k DIR --out JSON

Operands as C2″ (c2_g32.py): weights N(0, 0.02) seed 4096, 4096 tokens N(0, 1) FourOverSix per token; the real map =
model.layers.0.self_attn.o_proj of the Llama-3.1-8B TC 256x64 map (stored as 16x64 granules). Configurations (GEMM only):
  stock_ko               the deployment's stock build at this shape and T (stock_ko from --b7: the adopted table's width
                         and scheduler row), NVFP4 weights
  ceil                   n16k64_wA_nodisp_e64_t0 (--c3k): the width-128 tile with the dispatch compiled out (FourOverSix)
  A_{e2m1,real,e0m3}     n16k64_wA_g32 from --a1: today's 256x64 path at width 128 (A')
  C_{e2m1,real,e0m3}     n16k64_wA_g32_e64_t0 from --v-root: amendment 18's (t0, #4's tile, the uniform-branch dispatch)
The mixed builds run with the default scheduler setting (0, 1). Modes as C2': b2b (20 back-to-back calls), isolated (15
single calls after a 50 ms gap), sustained (2 s, NVML), each 3 rotated rounds. Checks before timing: every C equals its
A bitwise, and the defines are as registered.
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
from c2_freq import Sampler, gemm_times  # noqa: E402
from kernel import place  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120 import select as S  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

LAYER = 'model.layers.0.self_attn.o_proj'
MAP = Path('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_256x64.mixfp4map')
TAGS = ('e2m1', 'real', 'e0m3')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v-root', required=True)
    ap.add_argument('--a1', required=True)
    ap.add_argument('--b7', required=True)
    ap.add_argument('--c3k', required=True)
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
    real = mapio.read_map(MAP)[1][LAYER]
    grid = (n // 16, k // 64)
    assert tuple(real.shape) == grid
    assert torch.equal(real, real.reshape(n // 256, 16, k // 64).amax(1).repeat_interleave(16, 0)), 'not a 256x64 map'

    def wt(kind, mask=None):
        wn, wsb, gsw = N.quantize_weight(w, kind, mask, (16, 64) if mask is not None else None)
        return N.pack_nibbles(wn), place(wsb, k), float(gsw)
    sko = KernelSet('stock_ko', build_root=args.b7, table=S.TABLE_DIR / f'{S.gpu_slug()}.ko.json')
    sched = tuple(sko.schedule(n, k, t))
    K = dict(stock_ko=sko.pick(n, k, t), ceil=Kernel.load('n16k64_wA_nodisp_e64_t0', build_root=args.c3k),
             A=Kernel.load('n16k64_wA_g32', build_root=args.a1),
             C=Kernel.load('n16k64_wA_g32_e64_t0', build_root=args.v_root))
    assert not K['A'].manifest.get('extra_defines'), K['A'].manifest.get('extra_defines')
    assert K['C'].manifest.get('extra_defines') == {'MIXFP4_UNIFORM_DISPATCH': 1}, K['C'].manifest.get('extra_defines')
    tags = dict(e2m1=wt('map', torch.zeros(grid, dtype=torch.bool)), real=wt('map', real),
                e0m3=wt('map', torch.ones(grid, dtype=torch.bool)))
    cfgs = [('stock_ko', 'stock_ko', wt('nvfp4'), sched), ('ceil', 'ceil', wt('four_over_six'), None)]
    for tg in TAGS:
        cfgs += [(f'{v}_{tg}', v, tags[tg], None) for v in ('A', 'C')]

    def call(kk, wts, sch):
        wp, wsf, gsw = wts
        kern = K[kk]
        return lambda: kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gsw, scale_n=gsx, check=False, schedule=sch)
    calls = {name: call(kk, wts, sch) for name, kk, wts, sch in cfgs}
    res = dict(protocol='results/kernel_opt/PROTOCOL.md, amendment 18 (C2V)', gpu=B.gpu_info(), size=4096,
               kernels={kk: dict(config=v.cfg.name, sha256=v.sha256, root=str(v.path.parent.parent),
                                 defines=v.manifest.get('extra_defines')) for kk, v in K.items()},
               real_map=dict(file=str(MAP), layer=LAYER, e0m3_granules=int(real.sum()), granules=real.numel()),
               stock_ko=dict(width=sko.width(n, k, t), schedule=list(sched)), checks={}, modes={})
    for tg in TAGS:
        ya, yc = calls[f'A_{tg}'](), calls[f'C_{tg}']()
        torch.cuda.synchronize()
        ok = bool(torch.equal(ya.view(torch.int16), yc.view(torch.int16)))
        res['checks'][f'C_{tg} == A_{tg}'] = ok
        if not ok:
            B.write(args.out, res)
            raise SystemExit(f'C_{tg} differs from A_{tg}')
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
                print(f'{mode} r{r} {nm:10s} {per[nm][-1]:8.1f} us', flush=True)
        res['modes'][mode] = {nm: dict(us=statistics.median(per[nm]), rounds=per[nm], telemetry=tel[nm]) for nm in names}
        B.write(args.out, res)
    summ = {}
    for mode, d in res['modes'].items():
        u_ = {nm: v['us'] for nm, v in d.items()}
        pc = lambda a, b: 100 * (u_[a] / u_[b] - 1)  # noqa: E731
        s = {'ceil vs stock_ko': pc('ceil', 'stock_ko')}
        for tg in TAGS:
            s[f'C vs A, {tg}'] = pc(f'C_{tg}', f'A_{tg}')
            for v in ('A', 'C'):
                s[f'{v}_{tg} vs stock_ko'] = pc(f'{v}_{tg}', 'stock_ko')
                s[f'{v}_{tg} vs ceil'] = pc(f'{v}_{tg}', 'ceil')
        summ[mode] = s
    res['summary'] = summ
    res['status'] = 'complete'
    B.write(args.out, res)
    for mode, s in summ.items():
        print(mode, ' '.join(f'{k_}={v:+.2f}%' for k_, v in s.items() if ' vs A' in k_))


if __name__ == '__main__':
    main()

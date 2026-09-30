#!/usr/bin/env python3
"""Kernel-opt amendment 2 (#2), measurement C2': the 4096^3 breakdown, default vs MIXFP4_DISPATCH_FREQ.

    python experiments/kernel_opt/c2_freq.py --freq-root DIR --out JSON

Operands as C2 (experiments/paper_extra/C2_time.py): weights N(0, 0.02) seed 4096, 4096 tokens N(0, 1) FourOverSix per
token; the real map = model.layers.0.self_attn.o_proj of the Llama-3.1-8B 16x64 TC map (8x64 map for the weights-on-B
rows). Configurations (GEMM only):
  stock_wA, nodisp                          C2's references (sm120/build)
  {default,freq}_{e2m1,real,e0m3}            n16k64_wA from sm120/build / from --freq-root, all-E2M1 / real / all-E0M3 tags
  stock_wB, wB{default,freq}_{e2m1,real,e0m3}  the weights-on-B pair: n8k64_wB from sm120/build / --freq-root, 8x64 tags
Modes, one after the other so that no mode follows another's load (test B/C's deviation 2):
  b2b        C2's method: CUPTI over 20 back-to-back calls, 3 rotated rounds, median of the per-round medians
  isolated   15 single calls after a 50 ms idle gap each, CUPTI, 3 rotated rounds
  sustained  back-to-back calls for 2 s, CUDA-event time over the last half, NVML (SM clock, power, power-cap share),
             3 rotated rounds
Check before timing: every freq configuration's output equals its default's bitwise.
"""
import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import pynvml
import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402
from kernel import place  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402

LAYER = 'model.layers.0.self_attn.o_proj'
MAPS = {16: Path('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_16x64.mixfp4map'),
        8: Path('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_8x64.mixfp4map')}


class Sampler:
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.rows, self.stop = [], threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            r = pynvml.nvmlDeviceGetCurrentClocksEventReasons(self.h)
            self.rows.append((time.time(), pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
                              pynvml.nvmlDeviceGetPowerUsage(self.h) / 1e3, bool(r & pynvml.nvmlClocksEventReasonSwPowerCap)))
            time.sleep(0.005)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()

    def summary(self, since):
        rows = [r for r in self.rows if r[0] >= since]
        return dict(sm_mhz=statistics.median(r[1] for r in rows), power_w=statistics.median(r[2] for r in rows),
                    power_cap_share=sum(r[3] for r in rows) / len(rows)) if rows else {}


def gemm_times(prof):
    return [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--freq-root', required=True)
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
    real = {rows: mapio.read_map(path)[1][LAYER] for rows, path in MAPS.items()}

    def wt(kind, mask=None, rows=16):
        wn, wsb, gsw = N.quantize_weight(w, kind, mask, (rows, 64) if mask is not None else None)
        return N.pack_nibbles(wn), place(wsb, k), float(gsw)
    K = dict(stock_wA=Kernel.load('stock_wA'), nodisp=Kernel.load('n16k64_wA_nodisp'), default=Kernel.load('n16k64_wA'),
             freq=Kernel.load('n16k64_wA', build_root=args.freq_root), stock_wB=Kernel.load('stock_wB'),
             wBdefault=Kernel.load('n8k64_wB'), wBfreq=Kernel.load('n8k64_wB', build_root=args.freq_root))
    for kk in ('freq', 'wBfreq'):
        assert K[kk].manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1
    tags = {}
    for rows in (16, 8):
        grid = (n // rows, k // 64)
        tags[rows] = dict(e2m1=wt('map', torch.zeros(grid, dtype=torch.bool), rows),
                          real=wt('map', real[rows], rows), e0m3=wt('map', torch.ones(grid, dtype=torch.bool), rows))
    cfgs = [('stock_wA', 'stock_wA', wt('nvfp4'), False), ('nodisp', 'nodisp', wt('four_over_six'), False)]
    for tg in ('e2m1', 'real', 'e0m3'):
        cfgs += [(f'default_{tg}', 'default', tags[16][tg], False), (f'freq_{tg}', 'freq', tags[16][tg], False)]
    cfgs.append(('stock_wB', 'stock_wB', wt('nvfp4'), True))
    for tg in ('e2m1', 'real', 'e0m3'):
        cfgs += [(f'wBdefault_{tg}', 'wBdefault', tags[8][tg], True), (f'wBfreq_{tg}', 'wBfreq', tags[8][tg], True)]

    def call(kk, wts, on_b):
        wp, wsf, gsw = wts
        kern = K[kk]
        if not on_b:
            return lambda: kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gsw, scale_n=gsx, check=False)
        return lambda: kern.gemm(xp, xsf, wp, wsf, t, n, k, scale_m=gsx, scale_n_default=gsw, check=False)
    calls = {name: call(kk, wts, on_b) for name, kk, wts, on_b in cfgs}
    res = dict(protocol='results/kernel_opt/PROTOCOL.md, amendment 2 (C2\')', gpu=B.gpu_info(), size=4096,
               kernels={kk: dict(sha256=v.sha256, root=str(v.path.parent.parent)) for kk, v in K.items()},
               checks={}, modes={})
    for a in ('e2m1', 'real', 'e0m3'):
        for d, f in ((f'default_{a}', f'freq_{a}'), (f'wBdefault_{a}', f'wBfreq_{a}')):
            ya, yb = calls[d](), calls[f]()
            torch.cuda.synchronize()
            ok = bool(torch.equal(ya.view(torch.int16), yb.view(torch.int16)))
            res['checks'][f'{f} == {d}'] = ok
            if not ok:
                B.write(args.out, res)
                raise SystemExit(f'{f} differs from {d}')
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
                print(f'{mode} r{r} {nm:18s} {per[nm][-1]:8.1f} us', flush=True)
        res['modes'][mode] = {nm: dict(us=statistics.median(per[nm]), rounds=per[nm], telemetry=tel[nm]) for nm in names}
        B.write(args.out, res)
    summ = {}
    for mode, d in res['modes'].items():
        u = {nm: v['us'] for nm, v in d.items()}
        summ[mode] = {**{f'freq vs default, {tg}': 100 * (u[f'freq_{tg}'] / u[f'default_{tg}'] - 1) for tg in ('e2m1', 'real', 'e0m3')},
                      **{f'wB freq vs default, {tg}': 100 * (u[f'wBfreq_{tg}'] / u[f'wBdefault_{tg}'] - 1) for tg in ('e2m1', 'real', 'e0m3')},
                      **{f'{kk}_{tg} vs nodisp': 100 * (u[f'{kk}_{tg}'] / u['nodisp'] - 1) for kk in ('default', 'freq') for tg in ('e2m1', 'real', 'e0m3')},
                      **{f'{kk}_{tg} vs stock_wA': 100 * (u[f'{kk}_{tg}'] / u['stock_wA'] - 1) for kk in ('default', 'freq') for tg in ('e2m1', 'real', 'e0m3')},
                      **{f'wB{kk}_{tg} vs stock_wB': 100 * (u[f'wB{kk}_{tg}'] / u['stock_wB'] - 1) for kk in ('default', 'freq') for tg in ('e2m1', 'real', 'e0m3')}}
    res['summary'] = summ
    res['status'] = 'complete'
    B.write(args.out, res)
    for mode, s in summ.items():
        print(mode, ' '.join(f'{k_}={v:+.2f}%' for k_, v in s.items() if 'freq vs default' in k_))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""E0M3 investigation, tests B and C: the real GEMM at M = N = K = 4096, arm position and power/data controls.

    python experiments/kernel_opt/e0m3/test_bc.py --diag-root DIR --out JSON

Protocol: results/kernel_opt/e0m3/PROTOCOL.md (tests B, C). Operands as C2 (experiments/paper_extra/C2_time.py): weights
N(0, 0.02) seed 4096, 4096 tokens N(0, 1) quantized FourOverSix per token. Configurations (GEMM only):
  stock_wA, nodisp, e2m1, e0m3, real       C2's five (n16k64_wA with all-E2M1 / all-E0M3 / the real o_proj tags)
  e0m3_samebits   n16k64_wA on e2m1's packed bytes with the E0M3 tag set on every scale byte: same bits, other format
  e2m1_const, e0m3_const   n16k64_wA on constant data (every nibble 0x2, every scale byte 1.0), tags clear / set
  xor_e2m1, xor_e0m3, xor_real   n16k64_wA_xor (diagnostic build, --diag-root): the dispatch tree's arms permuted so the
                  all-E0M3 arm has the all-fall-through path; the same weights as e2m1 / e0m3 / real
  wB_stock, wB_e2m1, wB_e0m3, wBxor_e2m1, wBxor_e0m3   the weights-on-B orientation: stock_wB and n8k64_wB with 8x64 tags,
                  and n8k64_wB_xor (--diag-root)
Three timing modes, every configuration in each of --rounds rounds (rotated order: round r starts at r * len / rounds):
  b2b        C2's method: CUPTI kernel time over 20 back-to-back calls (median)
  isolated   10 single calls after a 50 ms idle gap each, CUPTI (median)
  sustained  back-to-back calls for --sustain-s seconds; CUDA-event time per call over the last half (median), and NVML
             (SM clock, power, software power-cap reason) sampled every 5 ms over the same half
Before timing, the xor builds' outputs must equal the default builds' bitwise on the same operands (a failure stops).
"""
import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import pynvml
import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402
from kernel import place  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402

LAYER = 'model.layers.0.self_attn.o_proj'
MAP = Path('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_16x64.mixfp4map')


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
        if not rows:
            return {}
        return dict(samples=len(rows), sm_mhz_median=statistics.median(r[1] for r in rows), sm_mhz_min=min(r[1] for r in rows),
                    power_w_median=statistics.median(r[2] for r in rows), power_cap_share=sum(r[3] for r in rows) / len(rows))


def gemm_events(prof):
    return [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--diag-root', required=True, help='build directory of n16k64_wA_xor and n8k64_wB_xor')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--sustain-s', type=float, default=2.0)
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
    real = masks[LAYER]

    def wA(kind, mask=None):
        wn, wsb, gsw = N.quantize_weight(w, kind, mask, (16, 64) if mask is not None else None)
        return N.pack_nibbles(wn), wsb, float(gsw)

    def wB(mask=None):
        wn, wsb, gsw = N.quantize_weight(w, 'map' if mask is not None else 'nvfp4', mask, (8, 64) if mask is not None else None)
        return N.pack_nibbles(wn), wsb, float(gsw)

    g16 = (n // 16, k // 64)
    g8 = (n // 8, k // 64)
    e2 = wA('map', torch.zeros(g16, dtype=torch.bool))
    e0 = wA('map', torch.ones(g16, dtype=torch.bool))
    rl = wA('map', real)
    same = (e2[0], e2[1] | 0x80, e2[2])
    const_nib = torch.full_like(e2[0], 0x22)
    const_sb = torch.full_like(e2[1], 0x38)
    c2 = (const_nib, const_sb, 1.0)
    c0 = (const_nib, const_sb | 0x80, 1.0)
    k_ = dict(stock_wA=Kernel.load('stock_wA'), mixed=Kernel.load('n16k64_wA'), nodisp=Kernel.load('n16k64_wA_nodisp'),
              xor=Kernel.load('n16k64_wA_xor', build_root=args.diag_root), stock_wB=Kernel.load('stock_wB'),
              wB=Kernel.load('n8k64_wB'), wBxor=Kernel.load('n8k64_wB_xor', build_root=args.diag_root))
    cfgs = [('stock_wA', 'stock_wA', wA('nvfp4'), 0), ('nodisp', 'nodisp', wA('four_over_six'), 0),
            ('e2m1', 'mixed', e2, 0), ('e0m3', 'mixed', e0, 0), ('real', 'mixed', rl, 0),
            ('e0m3_samebits', 'mixed', same, 0), ('e2m1_const', 'mixed', c2, 0), ('e0m3_const', 'mixed', c0, 0),
            ('xor_e2m1', 'xor', e2, 0), ('xor_e0m3', 'xor', e0, 0), ('xor_real', 'xor', rl, 0),
            ('wB_stock', 'stock_wB', wB(), 1), ('wB_e2m1', 'wB', wB(torch.zeros(g8, dtype=torch.bool)), 1),
            ('wB_e0m3', 'wB', wB(torch.ones(g8, dtype=torch.bool)), 1),
            ('wBxor_e2m1', 'wBxor', wB(torch.zeros(g8, dtype=torch.bool)), 1),
            ('wBxor_e0m3', 'wBxor', wB(torch.ones(g8, dtype=torch.bool)), 1)]

    def call(kname, wt, on_b):
        kern = k_[kname]
        wp, wsb, gsw = wt
        wsf = place(wsb, k)
        if not on_b:
            return lambda: kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gsw, scale_n=gsx, check=False)
        return lambda: kern.gemm(xp, xsf, wp, wsf, t, n, k, scale_m=gsx, scale_n_default=gsw, check=False)
    calls = {name: call(kn, wt, on_b) for name, kn, wt, on_b in cfgs}
    res = dict(protocol='results/kernel_opt/e0m3/PROTOCOL.md (tests B, C)', gpu=B.gpu_info(), size=4096,
               kernels={kn: kk.sha256 for kn, kk in k_.items()}, layer=LAYER, real_e0m3_share=float(real.float().mean()),
               checks={}, rows=[])
    # registered check: the xor builds compute bitwise what the default builds compute
    for a, b in (('e2m1', 'xor_e2m1'), ('e0m3', 'xor_e0m3'), ('real', 'xor_real'), ('wB_e2m1', 'wBxor_e2m1'),
                 ('wB_e0m3', 'wBxor_e0m3')):
        ya, yb = calls[a](), calls[b]()
        torch.cuda.synchronize()
        ok = bool(torch.equal(ya.view(torch.int16), yb.view(torch.int16)))
        res['checks'][f'{b} == {a}'] = ok
        if not ok:
            B.write(args.out, res)
            raise SystemExit(f'{b} differs from {a}')
    names = [c[0] for c in cfgs]
    per = {nm: dict(b2b=[], isolated=[], sustained=[], telemetry=[]) for nm in names}
    for r in range(args.rounds):
        shift = (r * len(names) // args.rounds) % len(names)
        for nm in names[shift:] + names[:shift]:
            fn = calls[nm]
            fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
            per[nm]['b2b'].append(statistics.median(gemm_events(prof)))
            iso = []
            for _ in range(10):
                time.sleep(0.05)
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    fn()
                    torch.cuda.synchronize()
                iso += gemm_events(prof)
            per[nm]['isolated'].append(statistics.median(iso))
            evs = []
            with Sampler() as smp:
                t0 = time.time()
                half, end = t0 + args.sustain_s / 2, t0 + args.sustain_s
                while time.time() < end:
                    e0_, e1_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0_.record()
                    fn()
                    e1_.record()
                    evs.append((time.time(), e0_, e1_))
                    if len(evs) % 64 == 0:
                        torch.cuda.synchronize()        # keep the queue shallow so the host clock tracks the GPU
                torch.cuda.synchronize()
            per[nm]['sustained'].append(statistics.median(a.elapsed_time(b) * 1e3 for (tt, a, b) in evs if tt >= half))
            per[nm]['telemetry'].append(smp.summary(half))
            tl = per[nm]['telemetry'][-1]
            print(f"r{r} {nm:14s} b2b {per[nm]['b2b'][-1]:8.1f}  isolated {per[nm]['isolated'][-1]:8.1f}  sustained "
                  f"{per[nm]['sustained'][-1]:8.1f} us   SM {tl.get('sm_mhz_median')} MHz  {tl.get('power_w_median', 0):.0f} W  "
                  f"cap {100 * tl.get('power_cap_share', 0):.0f} %", flush=True)
            B.write(args.out, dict(res, partial=per))
    for nm in names:
        d = per[nm]
        res['rows'].append(dict(config=nm, **{m: statistics.median(d[m]) for m in ('b2b', 'isolated', 'sustained')},
                                rounds={m: d[m] for m in ('b2b', 'isolated', 'sustained')},
                                sm_mhz=statistics.median(tl.get('sm_mhz_median', 0) for tl in d['telemetry']),
                                power_w=statistics.median(tl.get('power_w_median', 0) for tl in d['telemetry']),
                                power_cap_share=statistics.median(tl.get('power_cap_share', 0) for tl in d['telemetry']),
                                telemetry=d['telemetry']))
    by = {r_['config']: r_ for r_ in res['rows']}

    def pct(a, b, m):
        return 100 * (by[a][m] / by[b][m] - 1)
    res['contrasts'] = {m: {f'{a} vs {b}': pct(a, b, m) for a, b in (
        ('e0m3', 'e2m1'), ('real', 'e2m1'), ('e2m1', 'nodisp'), ('nodisp', 'stock_wA'), ('e0m3_samebits', 'e2m1'),
        ('e0m3_const', 'e2m1_const'), ('e2m1_const', 'e2m1'), ('xor_e0m3', 'xor_e2m1'), ('xor_e2m1', 'e2m1'),
        ('xor_e0m3', 'e0m3'), ('xor_real', 'real'), ('wB_e0m3', 'wB_e2m1'), ('wB_e2m1', 'wB_stock'),
        ('wBxor_e0m3', 'wBxor_e2m1'), ('wBxor_e2m1', 'wB_e2m1'))} for m in ('b2b', 'isolated', 'sustained')}
    res['status'] = 'complete'
    B.write(args.out, res)
    for m, cs in res['contrasts'].items():
        print(m, ' '.join(f'{k}={v:+.1f}%' for k, v in cs.items()))


if __name__ == '__main__':
    main()

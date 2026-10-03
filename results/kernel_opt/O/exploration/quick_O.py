#!/usr/bin/env python3
"""Exploratory (disclosed, not registered), item 3: the outlier cells. At f = 0 (all-E2M1 tags), 4096x4096 T=2048 and
14336x4096 T=512 show a 16x64 dispatch cost of about 5 % (about 2 % elsewhere) and an 8x64 no-dispatch ceiling of
+10 … +14 % over stock_ko (stock_wB_ko about equal to stock_ko). Both run width 128 with the default scheduler setting.

    python results/kernel_opt/O/exploration/quick_O.py --part sweep|sched|modes [--out JSON]

Kernels, each a single build at its width (weights: FourOverSix for stock and the ceilings, all-E2M1 tags for the mixed
builds, i.e. f = 0):
  stock_ko     stock_wA_e64 (build_7)              stock_wB_ko   stock_wB_e64 (build_P3freq)
  ceil16       n16k64_wA_nodisp_e64_t0 (build_C3k) A16 / U16     n16k64_wA_e64_t0 (build_7freq / build_U)
  ceil8        n8k64_wB_nodisp_t0 (build_P5)       A8 / U8       n8k64_wB_t0 (build_P3freq / build_U)
  width alternatives already in the families: stock_64 (stock_wA_n64), ceil16_64 / A16_64 (n16k64_wA_{nodisp_,}n64_t0),
  ceil8_n64 / A8_n64 (n8k64_wB_n64_{nodisp_,}t0, '128x64'), ceil8_64 / A8_64 (n8k64_wB_m64_{nodisp_,}t0)
Parts:
  sweep  T around both cells (tile counts 336 … 640, 1.8 … 3.4 waves of 188 CTAs), default setting (0, 1), cold
         (M1's condition: weights rotated past the L2 after a 512 MiB flush, activations quantized after the flush),
         CUPTI device time, 3 rotated rounds x 30
  sched  the two cells and two controls: every scheduler setting (raster 0/1/2 x swizzle 1/2/4/8) of every kernel,
         cold, 3 rotated rounds x 20; every setting's output is checked bitwise against (0, 1)'s
  modes  the two cells and two controls, default setting: cold vs warm back-to-back (L2), and 2 s sustained with NVML
         SM clock and power
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/dev/NVFP4-RaZeR/sm120')
sys.path.insert(0, '/home/dev/NVFP4-RaZeR/sm120/bench')
sys.path.insert(0, '/home/dev/NVFP4-RaZeR/experiments/kernel_opt')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from c2_freq import Sampler  # noqa: E402
from kernel import operands  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from tune_tiles import cold_us  # noqa: E402

KO = '/home/dev/n16k64_campaign/kernel_opt'
K = {'stock_ko': ('stock_wA_e64', 'build_7'), 'stock_wB_ko': ('stock_wB_e64', 'build_P3freq'),
     'ceil16': ('n16k64_wA_nodisp_e64_t0', 'build_C3k'), 'A16': ('n16k64_wA_e64_t0', 'build_7freq'),
     'U16': ('n16k64_wA_e64_t0', 'build_U'),
     'ceil8': ('n8k64_wB_nodisp_t0', 'build_P5'), 'A8': ('n8k64_wB_t0', 'build_P3freq'), 'U8': ('n8k64_wB_t0', 'build_U'),
     'stock_64': ('stock_wA_n64', 'build_7'),
     'ceil16_64': ('n16k64_wA_nodisp_n64_t0', 'build_C3k'), 'A16_64': ('n16k64_wA_n64_t0', 'build_7freq'),
     'ceil8_n64': ('n8k64_wB_n64_nodisp_t0', 'build_P5'), 'A8_n64': ('n8k64_wB_n64_t0', 'build_P3freq'),
     'ceil8_64': ('n8k64_wB_m64_nodisp_t0', 'build_P5'), 'A8_64': ('n8k64_wB_m64_t0', 'build_P3freq')}
SWEEP = {(4096, 4096): (1536, 1792, 2048, 2304, 2560, 4096), (14336, 4096): (384, 448, 512, 576, 640, 1024)}
CELLS = [(4096, 4096, 2048), (14336, 4096, 512), (4096, 4096, 4096), (14336, 4096, 1024)]
SCHEDS = [(r, s) for r in (0, 1, 2) for s in (1, 2, 4, 8)]
SMS = 188
_KER = {}


def kern(name):
    if name not in _KER:
        cfg, root = K[name]
        _KER[name] = Kernel.load(cfg, build_root=f'{KO}/{root}')
    return _KER[name]


def ops(name, n, k, t):
    """Weights for a kernel at f = 0: FourOverSix for stock and the ceilings, all-E2M1 tags for the mixed builds."""
    kk = kern(name)
    if kk.type_block is None:
        return operands(n, k, t, None, None, seed=n + k + t)
    rows = kk.type_block[0]
    return operands(n, k, t, torch.zeros((n // rows, k // 64), dtype=torch.bool), kk.type_block, seed=n + k + t)


def launcher(name, op, n, k, t, sched):
    kk = kern(name)
    if kk.weight_operand == 0:
        return lambda wp, wsf, q: kk.gemm(wp, wsf, q[0], q[1], n, t, k, scale_m_default=op['gsw'], scale_n=q[2], check=False,
                                          schedule=sched)
    return lambda wp, wsf, q: kk.gemm(q[0], q[1], wp, wsf, t, n, k, scale_m=q[2], scale_n_default=op['gsw'], check=False,
                                      schedule=sched)


def act(name, op):
    return lambda: kern(name).quant_rows(op['x'], 'four_over_six_rows')


def tiles(name, n, k, t):
    d = kern(name).desc['tile_mnk']
    m_, n_ = (n, t) if kern(name).weight_operand == 0 else (t, n)
    return -(-m_ // d[0]) * -(-n_ // d[1])


def rotate(names, r, rounds):
    s = (r * len(names) // rounds) % len(names)
    return names[s:] + names[:s]


def part_sweep(out):
    names = ['stock_ko', 'stock_wB_ko', 'ceil16', 'A16', 'ceil8', 'A8']
    res = []
    for (n, k), ts in SWEEP.items():
        for t in ts:
            o = {nm: ops(nm, n, k, t) for nm in names}
            per = {nm: [] for nm in names}
            for r in range(3):
                for nm in rotate(names, r, 3):
                    per[nm].append(cold_us(launcher(nm, o[nm], n, k, t, (0, 1)), o[nm]['wp'], o[nm]['wsf'], iters=30,
                                           act=act(nm, o[nm])))
            med = {nm: statistics.median(v) for nm, v in per.items()}
            row = dict(n=n, k=k, t=t, tiles=tiles('stock_ko', n, k, t), waves=tiles('stock_ko', n, k, t) / SMS, us=med,
                       rounds=per)
            res.append(row)
            pc = lambda a, b: 100 * (med[a] / med[b] - 1)  # noqa: E731
            print(f"{n}x{k} T={t:5d} tiles {row['tiles']:4d} waves {row['waves']:.2f} | stock_ko {med['stock_ko']:7.2f} us | "
                  f"stock_wB_ko {pc('stock_wB_ko', 'stock_ko'):+6.2f} % | ceil16 {pc('ceil16', 'stock_ko'):+6.2f} % "
                  f"A16/ceil16 {pc('A16', 'ceil16'):+6.2f} % | ceil8 {pc('ceil8', 'stock_ko'):+6.2f} % "
                  f"A8/ceil8 {pc('A8', 'ceil8'):+6.2f} %", flush=True)
    out['sweep'] = res


def part_sched(out):
    names = ['stock_ko', 'stock_wB_ko', 'ceil16', 'A16', 'U16', 'ceil8', 'A8', 'U8', 'stock_64', 'ceil16_64', 'A16_64',
             'ceil8_n64', 'A8_n64', 'ceil8_64', 'A8_64']
    res = []
    for n, k, t in CELLS:
        o = {nm: ops(nm, n, k, t) for nm in names}
        # bitwise: every setting's output equals the default's (CTA order does not change the computation)
        checks = {}
        for nm in names:
            q = act(nm, o[nm])()
            ref = launcher(nm, o[nm], n, k, t, (0, 1))(o[nm]['wp'], o[nm]['wsf'], q).clone()
            ok = all(torch.equal(launcher(nm, o[nm], n, k, t, s)(o[nm]['wp'], o[nm]['wsf'], q).view(torch.int16),
                                 ref.view(torch.int16)) for s in SCHEDS)
            checks[nm] = ok
            assert ok, (n, k, t, nm)
        jobs = [(nm, s) for nm in names for s in SCHEDS]
        per = {j: [] for j in jobs}
        for r in range(3):
            for nm, s in rotate(jobs, r, 3):
                per[(nm, s)].append(cold_us(launcher(nm, o[nm], n, k, t, s), o[nm]['wp'], o[nm]['wsf'], iters=20,
                                            act=act(nm, o[nm])))
        med = {j: statistics.median(v) for j, v in per.items()}
        best = {nm: min(SCHEDS, key=lambda s: med[(nm, s)]) for nm in names}
        row = dict(n=n, k=k, t=t, checks=checks, us={f'{nm}@{s[0]},{s[1]}': v for (nm, s), v in med.items()},
                   rounds={f'{nm}@{s[0]},{s[1]}': v for (nm, s), v in per.items()},
                   best={nm: list(b) for nm, b in best.items()})
        res.append(row)
        ref = med[('stock_ko', (0, 1))]
        print(f'{n}x{k} T={t}: stock_ko (0,1) {ref:.2f} us; best setting per kernel, % vs stock_ko (0,1) [default -> best]',
              flush=True)
        for nm in names:
            b = best[nm]
            print(f'   {nm:12s} default {100 * (med[(nm, (0, 1))] / ref - 1):+6.2f} %  best {b} {100 * (med[(nm, b)] / ref - 1):+6.2f} %'
                  f'  tiles {tiles(nm, n, k, t)}', flush=True)
    out['sched'] = res


def part_modes(out):
    names = ['stock_ko', 'stock_wB_ko', 'ceil16', 'A16', 'ceil8', 'A8']
    res = []
    for n, k, t in CELLS:
        o = {nm: ops(nm, n, k, t) for nm in names}
        q = {nm: act(nm, o[nm])() for nm in names}
        row = dict(n=n, k=k, t=t, cold={}, warm={}, sustained={})
        for r in range(3):
            for nm in rotate(names, r, 3):
                row['cold'].setdefault(nm, []).append(cold_us(launcher(nm, o[nm], n, k, t, (0, 1)), o[nm]['wp'],
                                                              o[nm]['wsf'], iters=30, act=act(nm, o[nm])))
                fn = lambda nm=nm: launcher(nm, o[nm], n, k, t, (0, 1))(o[nm]['wp'], o[nm]['wsf'], q[nm])  # noqa: E731
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(30):
                        fn()
                    torch.cuda.synchronize()
                d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA'
                     and 'device_kernel' in e.name.lower()]
                row['warm'].setdefault(nm, []).append(statistics.median(d))
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
                row['sustained'].setdefault(nm, []).append(dict(
                    us=statistics.median(a.elapsed_time(b) * 1e3 for (tt, a, b) in evs if tt >= half),
                    telemetry=smp.summary(half)))
        res.append(row)
        sc = {m: {nm: statistics.median(v if m != 'sustained' else [x['us'] for x in v]) for nm, v in row[m].items()}
              for m in ('cold', 'warm', 'sustained')}
        print(f'{n}x{k} T={t}:', flush=True)
        for m in ('cold', 'warm', 'sustained'):
            ref = sc[m]['stock_ko']
            print(f'   {m:9s} stock_ko {ref:7.2f} us | ' + ' '.join(f'{nm} {100 * (sc[m][nm] / ref - 1):+6.2f} %'
                                                             for nm in names[1:]), flush=True)
        for nm in names:
            tel = [x['telemetry'] for x in row['sustained'][nm]]
            print(f'   sustained {nm:12s} ' + ', '.join(f'{key} {statistics.median(x.get(key, float("nan")) for x in tel):.2f}'
                                                     for key in ('sm_mhz', 'power_w', 'power_cap_share')), flush=True)
    out['modes'] = res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', required=True, choices=('sweep', 'sched', 'modes'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = dict(part=args.part, gpu=torch.cuda.get_device_name(0),
               kernels={nm: dict(config=c, root=r) for nm, (c, r) in K.items()})
    dict(sweep=part_sweep, sched=part_sched, modes=part_modes)[args.part](out)
    out['kernel_sha256'] = {nm: kk.sha256 for nm, kk in _KER.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Step 1 of deviation 2 (results/paper/PROTOCOL_GEMM_ISOLATED.md): verify the cold-cache method before the run.

    PAPER_PYTHON experiments/paper/check_l2_cold.py --fo6 ART --tc16 ART --out JSON

Llama-3.1-8B's gate_proj (14336x4096) and q_proj (4096x4096) at T = 128 and 2048; stock_wA (FourOverSix weights) and
n16k64_wA (the typical 16x64 module's tags), at the tile table's widths. Every GEMM launch is isolated and timed by
CUPTI, as in bench_gemm_isolated.py, under five conditions:
  warm      the fresh activation is quantized, the GEMM runs once untimed on the same weights (loading them into
            L2), then the timed GEMM
  cold512   the registered method: read a 512 MiB buffer, then the fresh activation and its quantization, then the GEMM
  cold1024  the same with a 1 GiB buffer
  rotation  no flush; each repetition uses the next of K distinct copies of the weights (packed codes and scales),
            K x size >= 4x the L2
  event512  cold512 timed by item #3's isolated-call method instead: an event pair around the launch, no profiler
3 rounds, the conditions in a rotated order, --reps repetitions each (default 50).
PASS (registered; a failure stops deviation 2's run) for every (shape, T, kernel), on the medians:
  cold512 >= 0.99 x warm;  |cold512 / cold1024 - 1| <= 2 %;  |cold512 / rotation - 1| <= 2 %.
Recorded, not rules: cold512 / warm - 1 (the size of the L2 effect) and event512 - cold512 (the host enqueue gap).
"""
import argparse
import dataclasses
import math
import statistics
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_gemm_isolated import A, B, Kernel, KernelSet, NativeLinear, Telemetry, classify, stats, tag_modules  # noqa: E402,F401

SHAPES = ('gate_proj', 'q_proj')
CONDITIONS = ('warm', 'cold512', 'cold1024', 'rotation', 'event512')
TOL = 0.02


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fo6', required=True, help='the FourOverSix artifact of Llama-3.1-8B')
    ap.add_argument('--tc16', required=True, help='the TM-OPT+TC 16x64 artifact of Llama-3.1-8B')
    ap.add_argument('--tokens', default='128,2048')
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    tel = Telemetry()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    res = dict(status='running', gpu=B.gpu_info(), power_limit_w=tel.power_limit_w(), l2_bytes=l2, reps=args.reps,
               rounds=args.rounds, tolerance=TOL, cases=[])
    kernels = dict(stock=KernelSet('stock'), mixed=KernelSet('mixed'))
    weights = {}
    for kind, path, variant in (('fo6', args.fo6, 'first'), ('tc_16x64', args.tc16, 'typical')):
        meta, w = A.load(path, device='cpu')
        tags = tag_modules(meta, w)
        for proj in SHAPES:
            name = tags[proj][variant][0]
            weights[(kind, proj)] = (name, tags[proj][variant][1], w[name])
        del w
    buffers = {512: torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda'),
               1024: torch.ones(1024 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')}
    stream = torch.cuda.current_stream()
    failed = []
    for proj in SHAPES:
        for kind, kset in (('fo6', 'stock'), ('tc_16x64', 'mixed')):
            name, tiles, w = weights[(kind, proj)]
            w = dataclasses.replace(w, packed=w.packed.cuda(), scales=w.scales.cuda(), bias=None)
            lin = NativeLinear(w, kernels[kset], 'four_over_six_rows', name=f'{proj}@{kind}')
            n, k = lin.out_features, lin.in_features
            size = lin.packed.numel() + lin.sf.numel()
            copies = math.ceil(4 * l2 / size) + 1
            rot = [(lin.packed.clone(), lin.sf.clone()) for _ in range(copies)]
            for t in (int(v) for v in args.tokens.split(',')):
                kern = lin.kernel_set.pick(n, k, t)
                g = torch.Generator('cpu').manual_seed(n + k + t)
                pool = [torch.randn(t, k, generator=g).to('cuda', torch.bfloat16) for _ in range(2)]
                x = torch.empty_like(pool[0])
                y = torch.empty((t, n), dtype=torch.bfloat16, device='cuda')

                def gemm(q, wp=None):
                    xp, xsf, gs = q
                    packed, sf = wp if wp is not None else (lin.packed, lin.sf)
                    kern.gemm_ptr(packed.data_ptr(), sf.data_ptr(), xp.data_ptr(), xsf.data_ptr(), n, t, k,
                                  None, lin.global_scale, gs.data_ptr(), 1.0, None, y, stream.cuda_stream)

                def rep(cond, i):
                    """One repetition; returns the event time (us) for event512, else None (CUPTI times the GEMM)."""
                    if cond in ('cold512', 'cold1024', 'event512'):
                        buffers[1024 if cond == 'cold1024' else 512].sum()
                        torch.cuda.synchronize()
                    x.copy_(pool[i % 2])
                    torch.cuda.synchronize()
                    q = kern.quant_rows(x, 'four_over_six_rows')
                    torch.cuda.synchronize()
                    wp = rot[i % copies] if cond == 'rotation' else None
                    if cond == 'warm':
                        gemm(q)
                        torch.cuda.synchronize()
                    if cond == 'event512':
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        gemm(q)
                        b.record()
                        torch.cuda.synchronize()
                        return a.elapsed_time(b) * 1e3
                    gemm(q, wp)
                    torch.cuda.synchronize()
                    return None

                per = {c: [] for c in CONDITIONS}
                for r in range(args.rounds):
                    shift = (r * len(CONDITIONS) // args.rounds) % len(CONDITIONS)
                    for cond in CONDITIONS[shift:] + CONDITIONS[:shift]:
                        for i in range(3 if cond != 'rotation' else copies):      # warm-up (rotation: every copy once)
                            rep(cond, i)
                        if cond == 'event512':
                            per[cond] += [rep(cond, i) for i in range(args.reps)]
                            continue
                        with profile(activities=[ProfilerActivity.CUDA]) as prof:
                            for i in range(args.reps):
                                rep(cond, i)
                        ev = sorted((e for e in prof.events() if e.device_type.name == 'CUDA' and classify(e.name) == 'gemm'),
                                    key=lambda e: e.time_range.start)
                        d = [e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total for e in ev]
                        if cond == 'warm':
                            d = d[1::2]                          # the untimed warm-up launch precedes each timed one
                        if len(d) != args.reps:
                            raise SystemExit(f'{proj} T={t} {kind} {cond}: {len(d)} GEMM launches profiled, expected {args.reps}')
                        per[cond] += d
                med = {c: statistics.median(v) for c, v in per.items()}
                rules = dict(cold_not_faster_than_warm=med['cold512'] >= 0.99 * med['warm'],
                             flush_1gib_agrees=abs(med['cold512'] / med['cold1024'] - 1) <= TOL,
                             rotation_agrees=abs(med['cold512'] / med['rotation'] - 1) <= TOL)
                case = dict(proj=proj, out=n, inp=k, tokens=t, kind=kind, module=name, e0m3_tiles=tiles, kernel=kern.cfg.name,
                            rotation_copies=copies, rotation_bytes=copies * size, stats={c: stats(v) for c, v in per.items()},
                            cold_vs_warm_pct=100 * (med['cold512'] / med['warm'] - 1),
                            cold512_vs_cold1024_pct=100 * (med['cold512'] / med['cold1024'] - 1),
                            cold512_vs_rotation_pct=100 * (med['cold512'] / med['rotation'] - 1),
                            event_minus_cupti_us=med['event512'] - med['cold512'], rules=rules, passed=all(rules.values()))
                res['cases'].append(case)
                if not case['passed']:
                    failed.append(f'{proj} T={t} {kind}: {rules}')
                print(f"{proj} T={t} {kind}: warm {med['warm']:.1f} cold512 {med['cold512']:.1f} cold1024 {med['cold1024']:.1f} "
                      f"rotation {med['rotation']:.1f} event512 {med['event512']:.1f} us; cold vs warm {case['cold_vs_warm_pct']:+.1f} %; "
                      f"{'PASS' if case['passed'] else 'FAIL'}", flush=True)
                B.write(args.out, res)
            del rot, lin
    res['gpu_end'] = B.gpu_info()
    res['passed'] = not failed
    res['status'] = 'complete'
    B.write(args.out, res)
    if failed:
        raise SystemExit('cold-cache verification FAILED: ' + '; '.join(failed))
    print('cold-cache verification PASSED')


if __name__ == '__main__':
    main()

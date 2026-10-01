#!/usr/bin/env python3
"""Kernel-opt E0M3-fraction sweep (results/kernel_opt/c3k/PROTOCOL.md): GEMM latency against the share of E0M3 tiles on the
adopted 16x64 path (both dispatch variants), its no-dispatch ceiling, and the paper kernel, each against its own stock.

    python experiments/kernel_opt/c3k_fraction.py --b7 DIR --b7freq DIR --bceil DIR --out JSON [--shapes ..] [--tokens ..]

- Shapes (out x in): Llama-3.1-8B's 4096x4096, 14336x4096, 4096x14336. T in {1, 16, 128, 512, 2048, 8192}.
- Tags: E0M3 16x64-tile share f in {0, 1, 2, 5, 10, 25, 50, 75, 100} %, random (a seeded Bernoulli(f) per tile) and
  contiguous (the first round(f x tiles) tiles in row-major tile order), as C3; f = 0 and 100 are one map each. Plus
  the real map: the typical module (median E0M3 count) of that projection in Llama-3.1-8B's TM-OPT+TC 16x64 map.
- Weights: seeded N(0, 0.02), quantized with the tags (E0M3 alpha 1 tiles, FourOverSix elsewhere); the references take
  the same weights quantized FourOverSix (all E2M1).
- Kernels, each a KernelSet choosing the width per T from its table and passing its scheduler rows:
    ko          'mixed_ko' from --b7 with the adopted table (t0, #4, 4b): the adopted path, default dispatch
    ko_freq     'mixed_ko' from --b7freq: the adopted path, #2's pattern-0-first dispatch
    paper       'mixed' from sm120/build with the paper table: the paper kernel (tagged, default dispatch)
  references (all-E2M1 weights; timed once per shape and T, their time does not depend on the tags):
    ceiling     'nodisp_ko' from --bceil with the adopted table: the adopted tiles with the dispatch compiled out
    stock_ko    'stock_ko' from --b7 with the adopted table: the adopted stock
    stock       'stock' from sm120/build with the paper table: the paper stock
- Timing, the deviation-2 method: cold weights (rotation through copies totalling > 4x the L2, and a 512 MiB
  read-flush before each launch), the activation quantizer run after the flush (not timed), CUPTI device time of the
  isolated GEMM launch; 3 rounds x 30 launches with the configurations of a (shape, T) in a rotated order (round r starts
  at position r * len / 3); the value is the median of all 90 launches. NVML/nvidia-smi telemetry is sampled.
- Checks: on the timed operands ko, ko_freq and paper return the same output bit for bit (every map, shape and T).
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
sys.path.insert(0, str(REPO / 'experiments' / 'paper'))
import bench_gemm_isolated as G  # noqa: E402
import common as B  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120 import select as S  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
PATTERNS = ('random', 'contiguous')
SHAPES = {(4096, 4096): 'o_proj', (14336, 4096): 'gate_proj', (4096, 14336): 'down_proj'}
MAP = Path('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_16x64.mixfp4map')
MIXED = ('ko', 'ko_freq', 'paper')
REFS = ('ceiling', 'stock_ko', 'stock')


def tags(grid, f, pattern, seed):
    if pattern == 'random':
        return torch.rand(grid, generator=torch.Generator('cpu').manual_seed(seed)) < f
    m = torch.zeros(grid[0] * grid[1], dtype=torch.bool)
    m[:round(f * grid[0] * grid[1])] = True
    return m.reshape(grid)


def maps_for(n, k, real):
    grid = (n // 16, k // 64)
    out = [('all', 0.0, tags(grid, 0.0, 'random', 0)), ('all', 1.0, tags(grid, 1.0, 'random', 0))]
    for pattern in PATTERNS:
        for i, f in enumerate(FRACTIONS[1:-1]):
            out.append((pattern, f, tags(grid, f, pattern, n + k + i)))
    out.append(('real', float(real.float().mean()), real))
    return out


def typical(header, masks, proj):
    mods = [m for m in header['modules'] if m['name'].rsplit('.', 1)[-1] == proj]
    order = sorted(range(len(mods)), key=lambda i: (mods[i]['selected'], i))
    m = mods[order[(len(mods) - 1) // 2]]
    return m['name'], masks[m['name']]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--b7', required=True)
    ap.add_argument('--b7freq', required=True)
    ap.add_argument('--bceil', required=True)
    ap.add_argument('--paper-root', default=str(REPO / 'sm120' / 'build'), help='the paper builds (mixed, stock)')
    ap.add_argument('--shapes', default=','.join(f'{n}x{k}' for n, k in SHAPES))
    ap.add_argument('--tokens', default='1,16,128,512,2048,8192')
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    ko_table = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    paper_table = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    assert ko_table.exists(), ko_table
    ks = dict(ko=KernelSet('mixed_ko', build_root=args.b7, table=ko_table),
              ko_freq=KernelSet('mixed_ko', build_root=args.b7freq, table=ko_table),
              paper=KernelSet('mixed', build_root=args.paper_root, table=paper_table),
              ceiling=KernelSet('nodisp_ko', build_root=args.bceil, table=ko_table),
              stock_ko=KernelSet('stock_ko', build_root=args.b7, table=ko_table),
              stock=KernelSet('stock', build_root=args.paper_root, table=paper_table))
    assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1 for k in ks['ko_freq'].kernels.values())
    assert ks['ko'].schedules and ks['ceiling'].schedules == ks['ko'].schedules and not ks['paper'].schedules
    header, masks, _ = mapio.read_map(MAP)
    flush = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    tel = G.Telemetry(Path(str(args.out) + '.telemetry.csv'))
    res = dict(protocol='results/kernel_opt/c3k/PROTOCOL.md', gpu=B.gpu_info(), power_limit_w=tel.power_limit_w(),
               l2_bytes=l2, kernels={c: v.describe() for c, v in ks.items()},
               method=dict(iters=args.iters, warmup=args.warmup, rounds=args.rounds, flush_mib=512,
                           value='median of all launches; CUPTI device time of the GEMM'),
               rows=[], checks=[], real_maps={})

    def copies_of(op):
        size = op['wp'].numel() + op['wsf'].numel()
        return [(op['wp'].clone(), op['wsf'].clone()) for _ in range(math.ceil(4 * l2 / size) + 1)]

    def cold(kset, op, cps, n, k, t, x):
        kern = kset.pick(n, k, t)
        sched = kset.schedule(n, k, t)
        times = []

        def one(i):
            flush.sum()
            torch.cuda.synchronize()
            xp, xsf, gs = kern.quant_rows(x, 'four_over_six_rows')
            torch.cuda.synchronize()
            wp, wsf = cps[i % len(cps)]
            kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False, schedule=sched)
            torch.cuda.synchronize()
        for i in range(args.warmup):
            one(i)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for i in range(args.iters):
                one(args.warmup + i)
        times = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
        assert len(times) == args.iters, len(times)
        return times, kern.cfg.name, list(sched), kset.width(n, k, t)

    try:
        for shape in args.shapes.split(','):
            n, k = (int(v) for v in shape.split('x'))
            proj = SHAPES[(n, k)]
            name, real = typical(header, masks, proj)
            res['real_maps'][shape] = dict(module=name, e0m3_share=float(real.float().mean()))
            g = torch.Generator('cpu').manual_seed(n * 7 + k)
            w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
            wn, wsb, gsw = N.quantize_weight(w, 'four_over_six', None, None)
            ref_op = dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw))
            ref_cps = copies_of(ref_op)
            ops = []
            for pattern, f, mask in maps_for(n, k, real):
                wn, wsb, gsw = N.quantize_weight(w, 'map', mask, (16, 64))
                op = dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw))
                ops.append((pattern, f, op, copies_of(op)))
            for t in (int(v) for v in args.tokens.split(',')):
                x = torch.randn(t, k, generator=torch.Generator('cpu').manual_seed(n + k + t)).cuda().bfloat16()
                # bitwise: ko, ko_freq and paper on every map's timed operands
                for pattern, f, op, _ in ops:
                    outs = {}
                    for c in MIXED:
                        kern = ks[c].pick(n, k, t)
                        xp, xsf, gs = kern.quant_rows(x, 'four_over_six_rows')
                        outs[c] = kern.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs,
                                            check=False, schedule=ks[c].schedule(n, k, t)).clone()
                    eq = all(torch.equal(outs[c].view(torch.int16), outs['paper'].view(torch.int16)) for c in MIXED)
                    res['checks'].append(dict(shape=shape, tokens=t, pattern=pattern, f=f, equal=eq))
                    if not eq:
                        B.write(args.out, res)
                        raise SystemExit(f'bitwise check failed: {shape} T={t} {pattern} f={f}')
                items = [(c, i) for c in MIXED for i in range(len(ops))] + [(c, None) for c in REFS]
                per = {it: [] for it in items}
                meta = {}
                for r in range(args.rounds):
                    shift = (r * len(items) // args.rounds) % len(items)
                    for it in items[shift:] + items[:shift]:
                        c, i = it
                        op, cps = (ref_op, ref_cps) if i is None else (ops[i][2], ops[i][3])
                        times, kname, sched, width = cold(ks[c], op, cps, n, k, t, x)
                        per[it].append(times)
                        meta[it] = (kname, sched, width)
                for (c, i), rounds in per.items():
                    pattern, f = ('reference', None) if i is None else ops[i][:2]
                    flat = [v for rr in rounds for v in rr]
                    res['rows'].append(dict(shape=shape, tokens=t, kernel=c, pattern=pattern, f=f, kernel_build=meta[(c, i)][0],
                                            schedule=meta[(c, i)][1], width=meta[(c, i)][2], us=statistics.median(flat),
                                            rounds=[statistics.median(rr) for rr in rounds]))
                print(shape, t, {c: round(statistics.median([v for rr in per[(c, None)] for v in rr]), 2) for c in REFS},
                      flush=True)
                B.write(args.out, res)
    finally:
        tel.close()
    res['sampler'] = G.Telemetry.summarize(Path(str(args.out) + '.telemetry.csv'))
    res['status'] = 'complete'
    B.write(args.out, res)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Kernel-opt E0M3-fraction sweep on the adopted 8x64 path (C3w; results/kernel_opt/c3w/PROTOCOL.md, amendment 16).
It is C3k (c3k_fraction.py, amendment 8) on the weights-on-B family. GEMM latency against the share of E0M3 8x64 tiles:
the adopted path (both dispatch variants), its no-dispatch ceiling and the paper kernel, against stock.

    python experiments/kernel_opt/c3w_fraction.py --bp3freq DIR --bp3 DIR --bceil DIR --b7 DIR --out JSON

- Shapes (out x in): Llama-3.1-8B's 4096x4096, 14336x4096, 4096x14336. T in {1, 16, 128, 512, 2048, 8192}.
- Tags: E0M3 8x64-tile share f in {0, 1, 2, 5, 10, 25, 50, 75, 100} %, random (a seeded Bernoulli(f) per tile) and
  contiguous (the first round(f x tiles) tiles in row-major tile order), as C3k; f = 0 and 100 are one map each. Plus
  the real map: the typical module (lower median of the E0M3 count) of that projection in Llama-3.1-8B's TM-OPT+TC
  8x64 map.
- Weights: seeded N(0, 0.02), quantized with the tags (E0M3 alpha 1 tiles, FourOverSix elsewhere); the references take
  the same weights quantized FourOverSix (all E2M1).
- Kernels, each a KernelSet choosing the width per T from its table and passing its scheduler rows:
    adopted_freq  'mixed_wB_ko' from --bp3freq with the adopted table: the adopted (deployed) 8x64 path, i.e. what
                  'auto' gives -- t0, #2's pattern-0-first dispatch, the adopted widths (amendments 11-12b)
    adopted       'mixed_wB_ko' from --bp3: the same builds with the default dispatch
    paper         n8k64_wB from sm120/build (the paper's 8x64 kernel: one 128-wide build, tagged, default dispatch)
  references (all-E2M1 weights; timed once per shape and T, their time does not depend on the tags):
    ceiling       'nodisp_wB_ko' from --bceil with the adopted table: the adopted tiles with the dispatch compiled out
                  (amendment 13)
    stock_ko      'stock_ko' from --b7 with the adopted table: the target (stock, weights on A)
    stock_wB_ko   'stock_wB_ko' from --bp3freq with the adopted table: stock with the weights on B, tuned alike
    stock         'stock' from sm120/build with the paper table: the paper stock (the paper kernel's own)
- Timing: C3k's deviation-2 method, unchanged.
  - Cold weights: rotation through copies totalling > 4x the L2, and a 512 MiB read-flush before each launch.
  - The activation quantizer runs after the flush and is not timed. The GEMM is timed by its CUPTI device time, as an
    isolated launch.
  - 3 rounds x 30 launches, with the configurations of a (shape, T) in a rotated order: round r starts at position
    r * len / 3.
  - The value is the median of all 90 launches. Telemetry is sampled.
- Checks: on the timed operands, adopted_freq, adopted and paper return the same output bit for bit, for every map,
  shape and T.
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
MAP = Path('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_8x64.mixfp4map')
TB = (8, 64)
MIXED = ('adopted_freq', 'adopted', 'paper')
REFS = ('ceiling', 'stock_ko', 'stock_wB_ko', 'stock')


def tags(grid, f, pattern, seed):
    if pattern == 'random':
        return torch.rand(grid, generator=torch.Generator('cpu').manual_seed(seed)) < f
    m = torch.zeros(grid[0] * grid[1], dtype=torch.bool)
    m[:round(f * grid[0] * grid[1])] = True
    return m.reshape(grid)


def maps_for(n, k, real):
    grid = (n // TB[0], k // TB[1])
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
    ap.add_argument('--bp3freq', required=True, help="the adopted 8x64 path's directory (#2's dispatch) and stock_wB_e64")
    ap.add_argument('--bp3', required=True, help='the same 8x64 builds with the default dispatch')
    ap.add_argument('--bceil', required=True, help="the ceiling's builds (nodisp_wB_ko)")
    ap.add_argument('--b7', required=True, help='stock_ko')
    ap.add_argument('--paper-root', default=str(REPO / 'sm120' / 'build'), help='the paper builds (n8k64_wB, stock)')
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
    ks = dict(adopted_freq=KernelSet('mixed_wB_ko', build_root=args.bp3freq, table=ko_table),
              adopted=KernelSet('mixed_wB_ko', build_root=args.bp3, table=ko_table),
              # the paper kernel: n8k64_wB alone (the 'mixed_wB' family restricted to its 128-wide build)
              paper=KernelSet('mixed_wB', widths=(128,), build_root=args.paper_root, table=paper_table),
              ceiling=KernelSet('nodisp_wB_ko', build_root=args.bceil, table=ko_table),
              stock_ko=KernelSet('stock_ko', build_root=args.b7, table=ko_table),
              stock_wB_ko=KernelSet('stock_wB_ko', build_root=args.bp3freq, table=ko_table),
              stock=KernelSet('stock', build_root=args.paper_root, table=paper_table))
    assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1 for k in ks['adopted_freq'].kernels.values())
    assert not any((k.manifest.get('extra_defines') or {}).get('MIXFP4_DISPATCH_FREQ') for k in ks['adopted'].kernels.values())
    assert all(str(k.manifest['blob_gen'].get('TAG0')) == '0'
               for c in ('adopted_freq', 'adopted') for k in ks[c].kernels.values())
    assert [k.cfg.name for k in ks['paper'].kernels.values()] == ['n8k64_wB']
    assert ks['adopted'].table == ks['adopted_freq'].table == ks['ceiling'].table and ks['adopted'].table
    assert ks['ceiling'].schedules == ks['adopted'].schedules == ks['adopted_freq'].schedules
    assert all(ks[c].weight_operand == 1 for c in ('adopted_freq', 'adopted', 'paper', 'ceiling', 'stock_wB_ko'))
    header, masks, _ = mapio.read_map(MAP)
    flush = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    tel = G.Telemetry(Path(str(args.out) + '.telemetry.csv'))
    res = dict(protocol='results/kernel_opt/c3w/PROTOCOL.md', gpu=B.gpu_info(), power_limit_w=tel.power_limit_w(),
               l2_bytes=l2, kernels={c: v.describe() for c, v in ks.items()},
               method=dict(iters=args.iters, warmup=args.warmup, rounds=args.rounds, flush_mib=512,
                           value='median of all launches; CUPTI device time of the GEMM'),
               rows=[], checks=[], real_maps={})

    def copies_of(op):
        size = op['wp'].numel() + op['wsf'].numel()
        return [(op['wp'].clone(), op['wsf'].clone()) for _ in range(math.ceil(4 * l2 / size) + 1)]

    def gemm(kern, wp, wsf, xp, xsf, gs, gsw, n, t, k, sched):
        if kern.weight_operand == 0:
            return kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gsw, scale_n=gs, check=False, schedule=sched)
        return kern.gemm(xp, xsf, wp, wsf, t, n, k, scale_m=gs, scale_n_default=gsw, check=False, schedule=sched)

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
            gemm(kern, wp, wsf, xp, xsf, gs, op['gsw'], n, t, k, sched)
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
                wn, wsb, gsw = N.quantize_weight(w, 'map', mask, TB)
                op = dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw))
                ops.append((pattern, f, op, copies_of(op)))
            for t in (int(v) for v in args.tokens.split(',')):
                x = torch.randn(t, k, generator=torch.Generator('cpu').manual_seed(n + k + t)).cuda().bfloat16()
                # bitwise: adopted_freq, adopted and paper on every map's timed operands
                for pattern, f, op, _ in ops:
                    outs = {}
                    for c in MIXED:
                        kern = ks[c].pick(n, k, t)
                        xp, xsf, gs = kern.quant_rows(x, 'four_over_six_rows')
                        outs[c] = gemm(kern, op['wp'], op['wsf'], xp, xsf, gs, op['gsw'], n, t, k,
                                       ks[c].schedule(n, k, t)).clone()
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

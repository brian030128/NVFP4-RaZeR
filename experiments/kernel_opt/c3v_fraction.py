#!/usr/bin/env python3
"""Kernel-opt E0M3-fraction sweep on the adopted paths of the three units (C3v; results/kernel_opt/c3v/PROTOCOL.md,
amendment 19). It is C3k (c3k_fraction.py, amendment 8, 16x64) and C3w (c3w_fraction.py, amendment 16, 8x64) on the
deployment directory adopted with amendment 18, and the same sweep for 256x64: GEMM latency against the share of E0M3
tiles for the adopted path of a unit, the path deployed before it, the paper kernel and the adopted path's no-dispatch
ceiling, against stock.

    python experiments/kernel_opt/c3v_fraction.py --unit 256x64|16x64|8x64 --bv DIR --bprev DIR --bceil DIR --b7 DIR --out JSON

- Shapes (out x in): Llama-3.1-8B's 4096x4096, 14336x4096, 4096x14336. T in {1, 16, 128, 512, 2048, 8192}.
- Tags: E0M3 share f in {0, 1, 2, 5, 10, 25, 50, 75, 100} % of the unit's tiles (256x64, 16x64 or 8x64), random (a
  seeded Bernoulli(f) per tile) and contiguous (the first round(f x tiles) tiles in row-major tile order), with C3k's and
  C3w's seeds, so that the 16x64 and 8x64 maps are theirs; f = 0 and 100 are one map each. Plus the real map: the
  typical module (lower median of the E0M3 count) of that projection in Llama-3.1-8B's TM-OPT+TC map of the unit. A
  256x64 map is expanded to the 16x64 granules its artifacts store (uniform over each 256x64 tile).
- Weights: seeded N(0, 0.02), quantized with the tags (E0M3 alpha 1 tiles, FourOverSix elsewhere); the references take
  the same weights quantized FourOverSix (all E2M1).
- Kernels, each a KernelSet choosing the width per T from its table and passing its scheduler rows:
    adopted   the adopted (deployed) path from --bv with the adopted table, i.e. what 'auto' gives: 256x64 'mixed256_ko'
              (amendment 18); 16x64 'mixed_ko' and 8x64 'mixed_wB_ko' (#2's dispatch, with amendment 17's uniform-branch
              dispatch / pipelined flag read on the wide tiles)
    previous  the path deployed before: 256x64 'mixed256' (A', amendment 3) with the paper table; 16x64 'mixed_ko' and
              8x64 'mixed_wB_ko' with #2's dispatch alone, on the adopted table (C3k's ko_freq, C3w's adopted_freq)
    paper     the paper kernel: n16k64_wA (one 128-wide build) for 256x64; the paper set 'mixed' (four widths, the paper
              table) for 16x64, as C3k; n8k64_wB (one 128-wide build) for 8x64, as C3w
  references (all-E2M1 weights; timed once per shape and T, their time does not depend on the tags):
    ceiling      the adopted tiles with the dispatch compiled out, on the adopted table and the adopted path's scheduler
                 rows: 'nodisp256_ko', 'nodisp_ko' or 'nodisp_wB_ko' from --bceil
    stock_ko     'stock_ko' from --b7 with the adopted table: the target (stock, weights on A)
    stock_wB_ko  (8x64 only, as C3w) 'stock_wB_ko' from --bv with the adopted table: stock with the weights on B
    stock        'stock' from the paper builds with the paper table: the paper stock (the paper kernel's own)
- Timing: C3k's deviation-2 method, unchanged.
  - Cold weights: rotation through copies totalling > 4x the L2, and a 512 MiB read-flush before each launch.
  - The activation quantizer runs after the flush and is not timed. The GEMM is timed by its CUPTI device time, as an
    isolated launch.
  - 3 rounds x 30 launches, with the configurations of a (shape, T) in a rotated order: round r starts at position
    r * len / 3.
  - The value is the median of all 90 launches. Telemetry is sampled.
- Checks: on the timed operands, adopted, previous and paper return the same output bit for bit, for every map, shape
  and T. Before timing, every set's builds carry their registered defines and the tables are the expected ones.
"""
import argparse
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
ART = Path('/home/dev/n16k64_campaign/paper/artifacts')
MIXED = ('adopted', 'previous', 'paper')
F, U, P = 'MIXFP4_DISPATCH_FREQ', 'MIXFP4_UNIFORM_DISPATCH', 'MIXFP4_PIPE_FLAGS'
# Per unit: the map's tile rows, the stored granule, the weight operand, the sets (family and table: 'ko' = the adopted
# table, 'paper' = the paper table; the paper kernel's family, widths and builds), the references, and the registered
# defines of the adopted builds by width (results/kernel_opt/V/build_V.sh) and of every build of the previous path.
UNITS = {
    '256x64': dict(tile=256, tb=(16, 64), operand=0, adopted='mixed256_ko', previous=('mixed256', 'paper'),
                   paper=('mixed', (128,), ['n16k64_wA']), ceiling='nodisp256_ko', refs=('ceiling', 'stock_ko', 'stock'),
                   adopted_defines={16: {U: 1}, 32: {U: 1}, 64: {U: 1}, 128: {U: 1}}, previous_defines={}),
    '16x64': dict(tile=16, tb=(16, 64), operand=0, adopted='mixed_ko', previous=('mixed_ko', 'ko'),
                  paper=('mixed', None, ['n16k64_wA_n16', 'n16k64_wA_n32', 'n16k64_wA_n64', 'n16k64_wA']),
                  ceiling='nodisp_ko', refs=('ceiling', 'stock_ko', 'stock'),
                  adopted_defines={16: {F: 1}, 32: {F: 1}, 64: {F: 1, U: 1}, 128: {F: 1, U: 1}}, previous_defines={F: 1}),
    '8x64': dict(tile=8, tb=(8, 64), operand=1, adopted='mixed_wB_ko', previous=('mixed_wB_ko', 'ko'),
                 paper=('mixed_wB', (128,), ['n8k64_wB']), ceiling='nodisp_wB_ko',
                 refs=('ceiling', 'stock_ko', 'stock_wB_ko', 'stock'),
                 adopted_defines={16: {F: 1}, 32: {F: 1}, 64: {F: 1, P: 1}, '128x64': {F: 1, P: 1}, 128: {F: 1, P: 1}},
                 previous_defines={F: 1}),
}


def tags(grid, f, pattern, seed):
    if pattern == 'random':
        return torch.rand(grid, generator=torch.Generator('cpu').manual_seed(seed)) < f
    m = torch.zeros(grid[0] * grid[1], dtype=torch.bool)
    m[:round(f * grid[0] * grid[1])] = True
    return m.reshape(grid)


def maps_for(n, k, real, tile, tb):
    """C3k's / C3w's maps on tiles of `tile` rows, expanded to the stored `tb` granules (the identity unless 256x64)."""
    grid = (-(-n // tile), k // tb[1])

    def expand(mask):
        return mask.repeat_interleave(tile // tb[0], 0)[:n // tb[0]]
    out = [('all', 0.0, expand(tags(grid, 0.0, 'random', 0))), ('all', 1.0, expand(tags(grid, 1.0, 'random', 0)))]
    for pattern in PATTERNS:
        for i, f in enumerate(FRACTIONS[1:-1]):
            out.append((pattern, f, expand(tags(grid, f, pattern, n + k + i))))
    out.append(('real', float(real.float().mean()), real))
    return out


def typical(header, masks, proj):
    mods = [m for m in header['modules'] if m['name'].rsplit('.', 1)[-1] == proj]
    order = sorted(range(len(mods)), key=lambda i: (mods[i]['selected'], i))
    m = mods[order[(len(mods) - 1) // 2]]
    return m['name'], masks[m['name']]


def defines(kset):
    return {w: (k.manifest.get('extra_defines') or {}) for w, k in kset.kernels.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--unit', choices=tuple(UNITS), default='256x64')
    ap.add_argument('--bv', required=True, help='the adopted deployment directory (build_V)')
    ap.add_argument('--bprev', required=True, help="the previous path's builds (build_A1 / build_7freq / build_P3freq)")
    ap.add_argument('--bceil', required=True, help="the ceiling's builds (build_C3k for 256x64 and 16x64, build_P5 for 8x64)")
    ap.add_argument('--b7', required=True, help='stock_ko')
    ap.add_argument('--paper-root', default=str(REPO / 'sm120' / 'build'), help='the paper builds (paper kernel, stock)')
    ap.add_argument('--table', type=Path, default=None, help="the adopted table (default: the tracked '<gpu>.ko.json')")
    ap.add_argument('--shapes', default=','.join(f'{n}x{k}' for n, k in SHAPES))
    ap.add_argument('--tokens', default='1,16,128,512,2048,8192')
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    cfg = UNITS[args.unit]
    tb = cfg['tb']
    ko_table = args.table or S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    paper_table = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    assert ko_table.exists(), ko_table
    tables = dict(ko=ko_table, paper=paper_table)
    pfam, pwidths, pnames = cfg['paper']
    ks = dict(adopted=KernelSet(cfg['adopted'], build_root=args.bv, table=ko_table),
              previous=KernelSet(cfg['previous'][0], build_root=args.bprev, table=tables[cfg['previous'][1]]),
              paper=KernelSet(pfam, widths=pwidths, build_root=args.paper_root, table=paper_table),
              ceiling=KernelSet(cfg['ceiling'], build_root=args.bceil, table=ko_table),
              stock_ko=KernelSet('stock_ko', build_root=args.b7, table=ko_table),
              stock=KernelSet('stock', build_root=args.paper_root, table=paper_table))
    if 'stock_wB_ko' in cfg['refs']:
        ks['stock_wB_ko'] = KernelSet('stock_wB_ko', build_root=args.bv, table=ko_table)
    assert defines(ks['adopted']) == cfg['adopted_defines'], defines(ks['adopted'])
    assert all(d == cfg['previous_defines'] for d in defines(ks['previous']).values()), defines(ks['previous'])
    assert all(str(k.manifest['blob_gen'].get('TAG0')) == '0' for k in ks['adopted'].kernels.values())
    if args.unit != '256x64':
        assert all(str(k.manifest['blob_gen'].get('TAG0')) == '0' for k in ks['previous'].kernels.values())
        # the previous path is the same family on the same table: only the builds differ
        assert ks['previous'].table == ks['adopted'].table and ks['previous'].schedules == ks['adopted'].schedules
    assert not any(d for c in ('paper', 'ceiling') for d in defines(ks[c]).values())
    assert [k.cfg.name for k in ks['paper'].kernels.values()] == pnames
    assert ks['adopted'].table and ks['adopted'].table == ks['ceiling'].table
    assert ks['ceiling'].schedules == ks['adopted'].schedules
    assert all(ks[c].weight_operand == cfg['operand'] for c in MIXED + ('ceiling',))
    assert 'stock_wB_ko' not in ks or ks['stock_wB_ko'].weight_operand == 1
    header, masks, _ = mapio.read_map(ART / f'llama8b_tc_{args.unit}.mixfp4map')
    flush = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    tel = G.Telemetry(Path(str(args.out) + '.telemetry.csv'))
    res = dict(protocol='results/kernel_opt/c3v/PROTOCOL.md', unit=args.unit, gpu=B.gpu_info(),
               power_limit_w=tel.power_limit_w(), l2_bytes=l2, kernels={c: v.describe() for c, v in ks.items()},
               defines={c: {str(w): d for w, d in defines(v).items()} for c, v in ks.items()},
               method=dict(iters=args.iters, warmup=args.warmup, rounds=args.rounds, flush_mib=512,
                           value='median of all launches; CUPTI device time of the GEMM'),
               rows=[], checks=[], real_maps={})
    refs = cfg['refs']

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
            for pattern, f, mask in maps_for(n, k, real, cfg['tile'], tb):
                wn, wsb, gsw = N.quantize_weight(w, 'map', mask, tb)
                op = dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw))
                ops.append((pattern, f, op, copies_of(op)))
            for t in (int(v) for v in args.tokens.split(',')):
                x = torch.randn(t, k, generator=torch.Generator('cpu').manual_seed(n + k + t)).cuda().bfloat16()
                # bitwise: adopted, previous and paper on every map's timed operands
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
                items = [(c, i) for c in MIXED for i in range(len(ops))] + [(c, None) for c in refs]
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
                print(args.unit, shape, t, {c: round(statistics.median([v for rr in per[(c, None)] for v in rr]), 2)
                                            for c in refs}, flush=True)
                B.write(args.out, res)
    finally:
        tel.close()
    res['sampler'] = G.Telemetry.summarize(Path(str(args.out) + '.telemetry.csv'))
    res['status'] = 'complete'
    B.write(args.out, res)


if __name__ == '__main__':
    main()

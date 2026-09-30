#!/usr/bin/env python3
"""Measure the fastest CTA-tile width per (out, in, token bucket) and write the per-GPU table.

    python sm120/bench/tune_tiles.py --models llama8b,mistral7b,phi4,qwen27b \
        --maps llama8b=ART/llama8b_tc_16x64.mixfp4map ...     # -> sm120/configs/<gpu>.json

For every Linear shape of the listed models and every bucket T in select.BUCKETS, every width of
each family (mixed map kernel, stock NVFP4) is timed (CUPTI kernel time, median of 20) with the
real selector map tags for the mixed family, and the fastest is recorded. The raw timings are kept
next to the table (`<gpu>.raw.json`) so the choice is auditable. Run on an idle GPU; re-run on
every GPU model (the RTX 5090 and RTX PRO 6000 are not assumed to share a configuration).

The mixed family's E0M3 tags: per projection, the mask of the module with the most E0M3 tiles, from the 16x64 map
given by --maps MODEL=PATH (a .mixfp4map, e.g. the TM-OPT+TC artifacts), else from the frozen campaign map in
sm120/maps, else all E2M1. A shape shared by several projections or models is timed once, with the first one's tags.
The tags used per shape are recorded in the table's meta ('tags').

kernel-opt additions:
- --families mixed_wB: the weights-on-B family (8x64 maps; the width is the CTA tile's M, i.e. the tokens), with the
  tags of --maps8 MODEL=PATH (8x64 .mixfp4map), same rule.
- --cold: time each width by isolated launches on cold weights (every call on the next of K weight copies, K x size
  >= 4x L2, after a 512 MiB read-flush; CUPTI device time, median of 20), the deviation-2 method of
  results/paper/PROTOCOL_GEMM_ISOLATED.md, instead of CUPTI over back-to-back calls on L2-warm weights.
- --update: merge the timed families into the existing table and raw files; every other family's entries (and the
  top-level meta) are kept byte for byte; the new families' meta goes under meta['families'].
- --rounds R / --iters I (kernel-opt re-tune, amendment 4): with --cold, every width is timed in each of R rounds, the
  width order rotated between rounds, with I isolated launches per width and round; the value is the median over the
  rounds of the per-round medians (R = 3, I = 30 is the deviation-2 method's repetition). The defaults (1, 20) are the
  earlier --cold measurement. The per-round values go to the raw file under 'us_rounds'.
"""
import argparse
import datetime
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as B  # noqa: E402
from kernel import operands, selector_masks  # noqa: E402

from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import select as S  # noqa: E402


def map_masks(path, rows=16):
    """kernel.selector_masks for a given rows x 64 map file: {proj: (mask, module, E0M3 tiles)}, the module with the most."""
    header, masks, _ = mapio.read_map(path)
    assert tuple(header['type_block']) == (rows, 64), (path, header['type_block'])
    best = {}
    for m in header['modules']:
        proj = m['name'].rsplit('.', 1)[-1]
        if m['selected'] > best.get(proj, (None, None, -1))[2]:
            best[proj] = (masks[m['name']], m['name'], m['selected'])
    return best


_FLUSH = []


def cold_us(launch, wp, wsf, iters=20, warmup=3):
    """Median CUPTI time of isolated launches on cold weights: each call runs on the next of K copies of the weight
    operand (K x size >= 4x L2) after a 512 MiB read-flush, synchronized before and after."""
    if not _FLUSH:
        _FLUSH.append(torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda'))
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    copies = [(wp.clone(), wsf.clone()) for _ in range(math.ceil(4 * l2 / (wp.numel() + wsf.numel())) + 1)]

    def one(i):
        _FLUSH[0].sum()
        torch.cuda.synchronize()
        launch(*copies[i % len(copies)])
        torch.cuda.synchronize()
    for i in range(warmup):
        one(i)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            one(warmup + i)
    # the GEMM launches only (experiments/paper/bench_gemm_isolated.py's classification), not the flush's reduction
    d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'quant_rows_kernel' not in e.name.lower()
         and any(s in e.name.lower() for s in ('cutlass', 'device_kernel', 'gemm'))]
    assert len(d) == iters, (len(d), iters)
    return statistics.median(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='llama8b,mistral7b,phi4,qwen27b')
    ap.add_argument('--maps', action='append', default=[], metavar='MODEL=PATH',
                    help='16x64 .mixfp4map whose E0M3 tags the mixed family uses for that model')
    ap.add_argument('--maps8', action='append', default=[], metavar='MODEL=PATH',
                    help='8x64 .mixfp4map whose E0M3 tags the mixed_wB family uses for that model')
    ap.add_argument('--families', default='mixed,stock')
    ap.add_argument('--cold', action='store_true', help='isolated launches on cold weights (see the docstring)')
    ap.add_argument('--update', action='store_true', help='merge into the existing table (see the docstring)')
    ap.add_argument('--rounds', type=int, default=1, help='with --cold: rotated rounds per width (see the docstring)')
    ap.add_argument('--iters', type=int, default=20, help='with --cold: isolated launches per width and round')
    ap.add_argument('--out-dir', default=str(S.TABLE_DIR))
    ap.add_argument('--allow-busy', action='store_true')
    args = ap.parse_args()
    if not args.allow_busy:
        B.require_idle()
    fams = {f: S.KernelSet(f, table={}) for f in args.families.split(',')}
    shapes, tags, shapes8, tags8 = {}, {}, {}, {}
    maps = dict(spec.split('=', 1) for spec in args.maps)
    maps8 = dict(spec.split('=', 1) for spec in args.maps8)
    for model in args.models.split(','):
        if model in maps:
            sel, source = map_masks(maps[model]), maps[model]
        else:
            sel = selector_masks(model, 16)
            source = 'sm120/maps (frozen campaign map)' if sel else 'none (all E2M1)'
        sel8, source8 = (map_masks(maps8[model], 8), maps8[model]) if model in maps8 else ({}, 'none (all E2M1)')
        for proj, (n, k) in B.MODEL_SHAPES[model].items():
            if (n, k) not in shapes:
                entry = sel.get(proj)
                shapes[(n, k)] = entry[0] if entry else None
                tags[f'{n}x{k}'] = dict(model=model, proj=proj, source=source, module=entry[1] if entry else None,
                                        e0m3_tiles=entry[2] if entry else 0,
                                        tiles=int(entry[0].numel()) if entry else -(-n // 16) * (k // 64))
            if (n, k) not in shapes8:
                entry = sel8.get(proj)
                shapes8[(n, k)] = entry[0] if entry else None
                tags8[f'{n}x{k}'] = dict(model=model, proj=proj, source=source8, module=entry[1] if entry else None,
                                         e0m3_tiles=entry[2] if entry else 0,
                                         tiles=int(entry[0].numel()) if entry else -(-n // 8) * (k // 64))
    if args.rounds > 1 and not args.cold:
        raise SystemExit('--rounds needs --cold')
    table = {f: {} for f in fams}
    raw = {f: {} for f in fams}
    raw_rounds = {f: {} for f in fams}
    for (n, k), mask in shapes.items():
        for t in S.BUCKETS:
            for f, ks in fams.items():
                m, tb = {'mixed': (mask, (16, 64)), 'mixed_wB': (shapes8.get((n, k)), (8, 64))}.get(f, (None, None))
                op = operands(n, k, t, m, tb if m is not None else None, seed=n + k + t)
                times, launches = {}, {}
                for w, kern in ks.kernels.items():
                    if kern.weight_operand == 0:
                        launch = lambda wp, wsf, kern=kern: kern.gemm(wp, wsf, op['xp'], op['xsf'], n, t, k,  # noqa: E731
                                                                     scale_m_default=op['gsw'], scale_n=op['gsx'], check=False)
                    else:
                        launch = lambda wp, wsf, kern=kern: kern.gemm(op['xp'], op['xsf'], wp, wsf, t, n, k,  # noqa: E731
                                                                     scale_m=op['gsx'], scale_n_default=op['gsw'], check=False)
                    launches[w] = launch
                if args.cold:
                    ws, per = list(launches), {}
                    for r in range(args.rounds):
                        shift = (r * len(ws) // args.rounds) % len(ws)
                        for w in ws[shift:] + ws[:shift]:
                            per.setdefault(w, []).append(cold_us(launches[w], op['wp'], op['wsf'], iters=args.iters))
                    times = {w: statistics.median(per[w]) for w in ws}
                    if args.rounds > 1:
                        raw_rounds[f].setdefault(f'{n}x{k}', {})[str(t)] = {str(w): [round(v, 2) for v in per[w]] for w in ws}
                else:
                    for w, launch in launches.items():
                        fn = lambda launch=launch: launch(op['wp'], op['wsf'])  # noqa: E731
                        times[w] = sum(v['us'] for v in B.kernel_times(fn, iters=20).values())
                best = min(times, key=times.get)
                table[f].setdefault(f'{n}x{k}', {})[str(t)] = best
                raw[f].setdefault(f'{n}x{k}', {})[str(t)] = {str(w): round(v, 2) for w, v in times.items()}
            print(n, k, t, {f: table[f][f'{n}x{k}'][str(t)] for f in fams}, flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = S.gpu_slug()
    meta = dict(gpu=B.gpu_info(), created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                kernels={f: ks.sha256 for f, ks in fams.items()}, models=args.models, buckets=list(S.BUCKETS),
                note='value = fastest CTA tile width (tokens) for (out x in) at token counts <= bucket')
    if 'mixed' in fams or 'mixed_wB' not in fams:
        meta['tags'] = tags
    if 'mixed_wB' in fams:
        meta['tags_mixed_wB'] = tags8
    meta['method'] = 'cold isolated launches (--cold)' if args.cold else 'CUPTI over back-to-back calls'
    if args.cold and (args.rounds, args.iters) != (1, 20):
        meta['method'] += (f', {args.rounds} rotated rounds x {args.iters} launches per width, the median of the '
                           f'per-round medians')
    if args.update:
        old = json.loads((out / f'{slug}.json').read_text())
        old_raw = json.loads((out / f'{slug}.raw.json').read_text())
        for f in fams:
            old[f], old_raw['us'][f] = table[f], raw[f]
            old['meta'].setdefault('families', {})[f] = meta
            old_raw['meta'].setdefault('families', {})[f] = meta
            if args.rounds > 1:
                old_raw.setdefault('us_rounds', {})[f] = raw_rounds[f]
        (out / f'{slug}.json').write_text(json.dumps(old, indent=1, sort_keys=True) + '\n')
        (out / f'{slug}.raw.json').write_text(json.dumps(old_raw, indent=1, sort_keys=True) + '\n')
        print('updated', out / f'{slug}.json', 'families', list(fams))
        return
    (out / f'{slug}.json').write_text(json.dumps(dict(meta=meta, **table), indent=1, sort_keys=True) + '\n')
    raw_file = dict(meta=meta, us=raw, **(dict(us_rounds=raw_rounds) if args.rounds > 1 else {}))
    (out / f'{slug}.raw.json').write_text(json.dumps(raw_file, indent=1, sort_keys=True) + '\n')
    print('wrote', out / f'{slug}.json')


if __name__ == '__main__':
    main()

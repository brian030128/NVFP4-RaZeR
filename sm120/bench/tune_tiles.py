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
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as B  # noqa: E402
from kernel import operands, selector_masks  # noqa: E402

from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import select as S  # noqa: E402


def map_masks(path):
    """kernel.selector_masks for a given 16x64 map file: {proj: (mask, module, E0M3 tiles)}, the module with the most."""
    header, masks, _ = mapio.read_map(path)
    assert tuple(header['type_block']) == (16, 64), (path, header['type_block'])
    best = {}
    for m in header['modules']:
        proj = m['name'].rsplit('.', 1)[-1]
        if m['selected'] > best.get(proj, (None, None, -1))[2]:
            best[proj] = (masks[m['name']], m['name'], m['selected'])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='llama8b,mistral7b,phi4,qwen27b')
    ap.add_argument('--maps', action='append', default=[], metavar='MODEL=PATH',
                    help='16x64 .mixfp4map whose E0M3 tags the mixed family uses for that model')
    ap.add_argument('--families', default='mixed,stock')
    ap.add_argument('--out-dir', default=str(S.TABLE_DIR))
    ap.add_argument('--allow-busy', action='store_true')
    args = ap.parse_args()
    if not args.allow_busy:
        B.require_idle()
    fams = {f: S.KernelSet(f, table={}) for f in args.families.split(',')}
    shapes, tags = {}, {}
    maps = dict(spec.split('=', 1) for spec in args.maps)
    for model in args.models.split(','):
        if model in maps:
            sel, source = map_masks(maps[model]), maps[model]
        else:
            sel = selector_masks(model, 16)
            source = 'sm120/maps (frozen campaign map)' if sel else 'none (all E2M1)'
        for proj, (n, k) in B.MODEL_SHAPES[model].items():
            if (n, k) in shapes:
                continue
            entry = sel.get(proj)
            shapes[(n, k)] = entry[0] if entry else None
            tags[f'{n}x{k}'] = dict(model=model, proj=proj, source=source, module=entry[1] if entry else None,
                                    e0m3_tiles=entry[2] if entry else 0,
                                    tiles=int(entry[0].numel()) if entry else -(-n // 16) * (k // 64))
    table = {f: {} for f in fams}
    raw = {f: {} for f in fams}
    for (n, k), mask in shapes.items():
        for t in S.BUCKETS:
            for f, ks in fams.items():
                m = mask if f == 'mixed' else None
                op = operands(n, k, t, m, (16, 64) if m is not None else None, seed=n + k + t)
                times = {}
                for w, kern in ks.kernels.items():
                    fn = lambda: kern.gemm(op['wp'], op['wsf'], op['xp'], op['xsf'], n, t, k,  # noqa: E731
                                           scale_m_default=op['gsw'], scale_n=op['gsx'], check=False)
                    times[w] = sum(v['us'] for v in B.kernel_times(fn, iters=20).values())
                best = min(times, key=times.get)
                table[f].setdefault(f'{n}x{k}', {})[str(t)] = best
                raw[f].setdefault(f'{n}x{k}', {})[str(t)] = {str(w): round(v, 2) for w, v in times.items()}
            print(n, k, t, {f: table[f][f'{n}x{k}'][str(t)] for f in fams}, flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = S.gpu_slug()
    meta = dict(gpu=B.gpu_info(), created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                kernels={f: ks.sha256 for f, ks in fams.items()}, models=args.models, buckets=list(S.BUCKETS), tags=tags,
                note='value = fastest CTA tile width (tokens) for (out x in) at token counts <= bucket')
    (out / f'{slug}.json').write_text(json.dumps(dict(meta=meta, **table), indent=1, sort_keys=True) + '\n')
    (out / f'{slug}.raw.json').write_text(json.dumps(dict(meta=meta, us=raw), indent=1, sort_keys=True) + '\n')
    print('wrote', out / f'{slug}.json')


if __name__ == '__main__':
    main()

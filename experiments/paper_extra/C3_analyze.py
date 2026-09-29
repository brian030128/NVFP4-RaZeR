#!/usr/bin/env python3
"""Experiment C3 tables: results/paper_extra/C3/{C3.csv, C3.md} from C3_fraction.py's JSON. CPU, seconds.

    PAPER_PYTHON experiments/paper_extra/C3_analyze.py --in C3.json [--dest results/paper_extra/C3]

CSV: shape, T, kernel, pattern, fraction, time_us, overhead_vs_stock. The stock reference is stock@table for
mixed@table, stock@128 for mixed@128 and nodisp@128, and stock_wB for n8k64_wB.
Decomposition per (shape, T):
- fixed cost: the mixed kernel with no E0M3 tile against stock, at the same width;
- dispatch cost: mixed@128 with no E0M3 tile against nodisp@128 (same tile and arrangement, dispatch compiled out);
- arrangement: nodisp@128 against stock@128;
- the E0M3-share cost: mixed(f) against mixed(0), per pattern.
The real TM-OPT+TC maps' tile shares (the 16x64 and 8x64 maps of each paper model, from their .mixfp4map headers) are
marked on the curve.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper'))
import paper_common as P  # noqa: E402

REF = {'mixed@table': 'stock@table', 'mixed@128': 'stock@128', 'nodisp@128': 'stock@128', 'n8k64_wB': 'stock_wB',
       'stock@table': 'stock@table', 'stock@128': 'stock@128', 'stock_wB': 'stock_wB'}


def real_shares():
    """E0M3 tile share of the committed 8x64 and 16x64 maps, per paper model (their .mixfp4map headers)."""
    sys.path.insert(0, str(P.REPO / 'sm120'))
    from mixfp4_sm120 import mapio
    out = {}
    for m in P.MODELS:
        for u in ('8x64', '16x64'):
            f = P.artifact(P.OUT_DEFAULT, m, f'tc_{u}').with_name(f'{m}_tc_{u}.mixfp4map')
            if f.exists():
                h, _, _ = mapio.read_map(f)
                out[f'{m} {u}'] = h['totals']['selected_tiles'] / h['totals']['total_tiles']
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', type=Path, required=True)
    ap.add_argument('--dest', type=Path, default=P.REPO / 'results' / 'paper_extra' / 'C3')
    args = ap.parse_args()
    r = json.loads(args.inp.read_text())
    args.dest.mkdir(parents=True, exist_ok=True)
    by = {(x['shape'], x['tokens'], x['kernel'], x['pattern'], x['fraction']): x for x in r['rows']}
    t_of = lambda s, t, k, pat=None, f=0.0: by[(s, t, k, pat, f)]['time_us']  # noqa: E731
    csv = ['shape,T,kernel,pattern,fraction,time_us,overhead_vs_stock']
    for x in r['rows']:
        ref = t_of(x['shape'], x['tokens'], REF[x['kernel']])
        csv.append(f"{x['shape']},{x['tokens']},{x['kernel']},{x['pattern'] or ''},{x['fraction']:g},{x['time_us']:.3f},"
                   f"{x['time_us'] / ref - 1:.5f}")
    (args.dest / 'C3.csv').write_text('\n'.join(csv) + '\n')
    shapes = sorted({x['shape'] for x in r['rows']}, key=lambda s: [int(v) for v in s.split('x')])
    tokens = sorted({x['tokens'] for x in r['rows']})
    fr = r['fractions']
    md = ['# Experiment C3: GEMM latency against the E0M3 tile share\n',
          'CUPTI kernel time, median of 3 rounds in a rotated order (each the median of 20 calls). Overheads in %.\n',
          '## Decomposition at no E0M3 tile\n',
          '| shape | T | table width | fixed: mixed@table vs stock@table | fixed: mixed@128 vs stock@128 | dispatch: mixed@128 vs nodisp@128 | arrangement: nodisp@128 vs stock@128 | 8x64 fixed: n8k64_wB vs stock_wB |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    dec = {}
    for s in shapes:
        for t in tokens:
            st, s128, nd, wb = (t_of(s, t, k) for k in ('stock@table', 'stock@128', 'nodisp@128', 'stock_wB'))
            m0 = {k: min(t_of(s, t, k, p) for p in ('random', 'contiguous')) for k in ('mixed@table', 'mixed@128', 'n8k64_wB')}
            d = dec[f'{s}@{t}'] = dict(fixed_table=m0['mixed@table'] / st - 1, fixed_128=m0['mixed@128'] / s128 - 1,
                                       dispatch=m0['mixed@128'] / nd - 1, arrangement=nd / s128 - 1,
                                       fixed_8x64=m0['n8k64_wB'] / wb - 1)
            w = by[(s, t, 'mixed@table', 'random', 0.0)]['width']
            md.append(f"| {s} | {t} | {w} | " + ' | '.join(f'{100 * d[k]:+.1f}' for k in
                                                           ('fixed_table', 'fixed_128', 'dispatch', 'arrangement', 'fixed_8x64')) + ' |')
    md += ['\n## The E0M3-share cost: kernel(f) / kernel(0) − 1, in %\n']
    curves = {}
    for kern in ('mixed@table', 'n8k64_wB'):
        md += [f'### {kern}\n', '| shape | T | pattern | ' + ' | '.join(f'{100 * f:g} %' for f in fr) + ' |',
               '|---|---:|---|' + '---:|' * len(fr)]
        for s in shapes:
            for t in tokens:
                for pat in ('random', 'contiguous'):
                    z = t_of(s, t, kern, pat, 0.0)
                    c = [t_of(s, t, kern, pat, f) / z - 1 for f in fr]
                    curves[f'{kern}|{s}|{t}|{pat}'] = c
                    md.append(f'| {s} | {t} | {pat} | ' + ' | '.join(f'{100 * v:+.1f}' for v in c) + ' |')
        md.append('')
    shares = real_shares()
    md += ['## The real maps\n', 'E0M3 tile share of the committed FlipQuant (ours) maps (TM-OPT+TC): ' +
           ', '.join(f'{k} {100 * v:.2f} %' for k, v in shares.items()) + '.\n']
    (args.dest / 'C3.md').write_text('\n'.join(md) + '\n')
    (args.dest / 'C3_summary.json').write_text(json.dumps(dict(decomposition=dec, curves=curves, fractions=fr,
                                                               real_map_tile_shares=shares), indent=1) + '\n')
    print('wrote', args.dest / 'C3.csv', args.dest / 'C3.md')


if __name__ == '__main__':
    main()

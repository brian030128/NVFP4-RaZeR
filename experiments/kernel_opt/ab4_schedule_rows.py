#!/usr/bin/env python3
"""Kernel-opt amendment 5 (#4), post-hoc listing (not a registered script): the tuned scheduler rows that differ from
the default (0, 1), per family, with the tuner's per-round times.

    python experiments/kernel_opt/ab4_schedule_rows.py [--table-dir results/kernel_opt/4/table] [--out MD]

The percentages are recomputed from the raw file's times, which are rounded to 0.01 us, so a row the tuner's rule
(median >= 0.5 % below the default's, on unrounded times) accepted can print as -0.49 %.
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SLUG = 'nvidia_rtx_pro_6000_blackwell_workstation_edition'
RASTER = {0: 'heuristic', 1: 'along M', 2: 'along N'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--table-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / '4' / 'table')
    ap.add_argument('--out', type=Path, default=REPO / 'results' / 'kernel_opt' / '4' / 'schedule_rows.md')
    args = ap.parse_args()
    tab = json.loads((args.table_dir / f'{SLUG}.json').read_text())
    raw = json.loads((args.table_dir / f'{SLUG}.schedule_raw.json').read_text())['us_rounds']
    md = ['## #4: tuned scheduler rows that differ from the default (0, 1)\n',
          'Times are the tuner\'s per-round medians (µs, 3 rounds × 30, cold weights, warm activations), rounded to '
          '0.01 µs; the change is recomputed from them.\n']
    for fam, rows in tab['schedule'].items():
        total = sum(len(r) for r in rows.values())
        picks = [(shape, int(b), tuple(rs)) for shape, r in rows.items() for b, rs in r.items() if tuple(rs) != (0, 1)]
        count = collections.Counter(p[2] for p in picks)
        md.append(f'### {fam}: {len(picks)} of {total} (shape, bucket) rows\n')
        md.append('Settings: ' + ', '.join(f'{RASTER[r]} / swizzle {s}: {n}' for (r, s), n in count.most_common()) + '\n')
        md.append('| shape (N x K) | bucket (T ≤) | setting | change | default rounds | setting rounds |')
        md.append('|---|---:|---|---:|---|---|')
        for shape, b, (r, s) in sorted(picks, key=lambda p: (p[1], p[0])):
            per = raw[fam][shape][str(b)]
            dflt, best = per['0,1'], per[f'{r},{s}']
            g = 100 * (statistics.median(best) / statistics.median(dflt) - 1)
            md.append(f'| {shape} | {b} | {RASTER[r]} / {s} | {g:+.2f} % | {dflt} | {best} |')
        md.append('')
    args.out.write_text('\n'.join(md) + '\n')
    print('\n'.join(md[:2]), f'... wrote {args.out}')


if __name__ == '__main__':
    main()

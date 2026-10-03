#!/usr/bin/env python3
"""Tables and figures of the 8x64 E0M3-fraction sweep (C3w; c3w_fraction.py; results/kernel_opt/c3w/PROTOCOL.md).
It is c3k_analyze.py on the weights-on-B kernels, with the same tables and figure layout.

    python experiments/kernel_opt/c3w_analyze.py --src JSON --out-dir results/kernel_opt/c3w [--c3k results/kernel_opt/c3k/C3k_raw.json]

- C3w.md:
  - the decomposition at no E0M3 tile: tiles (ceiling vs stock_ko), dispatch (adopted vs ceiling), and placement
    (stock_wB_ko vs stock_ko);
  - per kernel, the overhead vs its own stock and kernel(f) / kernel(0) - 1, per shape x T x pattern;
  - for the adopted path, the overhead vs stock_wB_ko (the same placement);
  - the slope (least squares on the random pattern, f in 0 .. 1), and which dispatch variant of the adopted path is
    faster at each f.
- C3w.csv: every timed row. C3w_summary.json: the derived numbers.
- C3w_overhead.png / .pdf: the overhead vs own stock against f. Rows are shapes, columns are T, one line per kernel
  (random solid, contiguous dashed, the real map as a star).
- C3w_overhead_mean.png / .pdf: the same, averaged over the shapes (C3k's layout).
- C3w_vs_C3k_mean.png / .pdf (with --c3k): the adopted paths of both families, mean of the shapes, overhead vs stock_ko.
  This compares two separate runs: C3k ran 2026-10-01.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

KERNELS = ('adopted', 'adopted_freq', 'paper')
OWN_STOCK = dict(adopted='stock_ko', adopted_freq='stock_ko', paper='stock', ceiling='stock_ko')
LABEL = dict(adopted='adopted 8x64, default dispatch', adopted_freq="adopted 8x64, #2's dispatch (deployed)",
             paper='paper 8x64 kernel (n8k64_wB)', ceiling='no-dispatch ceiling (adopted 8x64 tiles)')
COLOR = dict(adopted='tab:blue', adopted_freq='tab:orange', paper='tab:gray', ceiling='tab:green')
FS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)


def pct(a, b):
    return 100 * (a / b - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--c3k', type=Path, default=None, help="C3k's raw JSON, for the cross-family comparison figure")
    args = ap.parse_args()
    rec = json.loads(args.src.read_text())
    rows = rec['rows']
    shapes = list(dict.fromkeys(r['shape'] for r in rows))
    tokens = sorted({r['tokens'] for r in rows})
    ref = {(r['shape'], r['tokens'], r['kernel']): r for r in rows if r['pattern'] == 'reference'}

    def val(shape, t, kern, pattern, f):
        for r in rows:
            if (r['shape'], r['tokens'], r['kernel']) == (shape, t, kern) and \
                    (r['pattern'] == pattern or (r['pattern'] == 'all' and f in (0.0, 1.0))) and \
                    (f is None or abs(r['f'] - f) < 1e-9):
                return r
        return None

    md = ['# Kernel-opt E0M3-fraction sweep on the adopted 8x64 path (C3w)\n',
          'The deviation-2 method, as C3k: cold weights, the activation quantizer after the flush, isolated CUPTI GEMM',
          'launches, 3 rotated rounds × 30; the median of all launches. Overheads in %. Each kernel is against its own',
          'stock: the adopted 8x64 kernels and their ceiling against `stock_ko` (the target, weights on A), the paper',
          '8x64 kernel against the paper `stock`. The adopted path is also given against `stock_wB_ko` (the same',
          'placement, weights on B).\n',
          f"Bitwise checks: {sum(c['equal'] for c in rec['checks'])} of {len(rec['checks'])} equal "
          '(adopted, #2\'s dispatch and the paper kernel, every map, shape and T).\n',
          '## Decomposition at no E0M3 tile\n',
          'tiles = the ceiling vs stock_ko (the 1x8 / 1x4 tiles, dispatch compiled out); dispatch = the adopted kernel vs '
          'the ceiling; placement = stock_wB_ko vs stock_ko.\n',
          '| shape | T | width (adopted / paper) | adopted (#2) vs stock_ko | adopted (default) vs stock_ko | '
          'tiles: ceiling vs stock_ko | dispatch: adopted (#2) vs ceiling | dispatch: adopted (default) vs ceiling | '
          'placement: stock_wB_ko vs stock_ko | paper vs paper stock | stock_ko vs paper stock |',
          '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    summ = dict(decomposition={}, slope={}, crossover={}, real={})
    for shape in shapes:
        for t in tokens:
            sk, sp, ce = ref[(shape, t, 'stock_ko')]['us'], ref[(shape, t, 'stock')]['us'], ref[(shape, t, 'ceiling')]['us']
            swb = ref[(shape, t, 'stock_wB_ko')]['us']
            ad0, af0, pa0 = (val(shape, t, c, 'all', 0.0) for c in KERNELS)
            d = dict(adopted_freq=pct(af0['us'], sk), adopted=pct(ad0['us'], sk), ceiling=pct(ce, sk),
                     dispatch_freq=pct(af0['us'], ce), dispatch=pct(ad0['us'], ce), placement=pct(swb, sk),
                     paper=pct(pa0['us'], sp), stock_ko_vs_stock=pct(sk, sp))
            summ['decomposition'][f'{shape}@{t}'] = d
            md.append(f"| {shape} | {t} | {af0['width']} / {pa0['width']} | {d['adopted_freq']:+.1f} | {d['adopted']:+.1f} | "
                      f"{d['ceiling']:+.1f} | {d['dispatch_freq']:+.1f} | {d['dispatch']:+.1f} | {d['placement']:+.1f} | "
                      f"{d['paper']:+.1f} | {d['stock_ko_vs_stock']:+.1f} |")
    md.append('')
    for kind, title in (('own', 'overhead vs own stock'), ('rel', 'kernel(f) / kernel(0) − 1'),
                        ('wB', 'overhead vs stock_wB_ko (the same placement)')):
        md.append(f'## {title}, in %\n')
        for c in (KERNELS if kind != 'wB' else ('adopted_freq', 'adopted')):
            md.append(f'### {LABEL[c]}\n')
            md.append('| shape | T | pattern | ' + ' | '.join(f'{round(100 * f)} %' for f in FS) + ' | real map (f) |')
            md.append('|---|---:|---|' + '---:|' * (len(FS) + 1))
            for shape in shapes:
                for t in tokens:
                    base = (ref[(shape, t, OWN_STOCK[c])]['us'] if kind == 'own' else ref[(shape, t, 'stock_wB_ko')]['us']
                            if kind == 'wB' else val(shape, t, c, 'all', 0.0)['us'])
                    real = val(shape, t, c, 'real', None)
                    for pattern in ('random', 'contiguous'):
                        cells = []
                        for f in FS:
                            r = val(shape, t, c, 'all' if f in (0.0, 1.0) else pattern, f)
                            cells.append(f"{pct(r['us'], base):+.1f}")
                        rcell = f"{pct(real['us'], base):+.1f} ({100 * real['f']:.0f} %)" if pattern == 'random' else ''
                        md.append(f'| {shape} | {t} | {pattern} | ' + ' | '.join(cells) + f' | {rcell} |')
            md.append('')
    # slope: least squares of the overhead vs own stock on f (random pattern), per kernel, shape, T
    md.append('## Slope and crossover\n')
    md.append('Slope: least-squares percentage points of overhead per unit E0M3 share (random pattern, f = 0 … 1). '
              'Crossover: the adopted default dispatch vs #2\'s, adopted_freq / adopted − 1 in % (negative: #2\'s is faster).\n')
    md.append('| shape | T | slope adopted | slope adopted (#2) | slope paper | ' + ' | '.join(f'#2 vs default @ {round(100 * f)} %' for f in FS) + ' |')
    md.append('|---|---:|---:|---:|---:|' + '---:|' * len(FS))
    for shape in shapes:
        for t in tokens:
            slopes = {}
            for c in KERNELS:
                base = ref[(shape, t, OWN_STOCK[c])]['us']
                ys = [pct(val(shape, t, c, 'all' if f in (0.0, 1.0) else 'random', f)['us'], base) for f in FS]
                mf, my = statistics.mean(FS), statistics.mean(ys)
                slopes[c] = sum((f - mf) * (y - my) for f, y in zip(FS, ys)) / sum((f - mf) ** 2 for f in FS)
            cross = [pct(val(shape, t, 'adopted_freq', 'all' if f in (0.0, 1.0) else 'random', f)['us'],
                         val(shape, t, 'adopted', 'all' if f in (0.0, 1.0) else 'random', f)['us']) for f in FS]
            summ['slope'][f'{shape}@{t}'] = slopes
            summ['crossover'][f'{shape}@{t}'] = dict(zip([str(f) for f in FS], cross))
            md.append(f"| {shape} | {t} | {slopes['adopted']:+.2f} | {slopes['adopted_freq']:+.2f} | {slopes['paper']:+.2f} | "
                      + ' | '.join(f'{v:+.1f}' for v in cross) + ' |')
    md.append('')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'C3w.md').write_text('\n'.join(md) + '\n')
    with open(args.out_dir / 'C3w.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['shape', 'tokens', 'kernel', 'pattern', 'f', 'kernel_build', 'schedule', 'us', 'round_medians'])
        for r in rows:
            wr.writerow([r['shape'], r['tokens'], r['kernel'], r['pattern'], r['f'], r['kernel_build'], r['schedule'],
                         f"{r['us']:.3f}", ' '.join(f'{v:.3f}' for v in r['rounds'])])
    (args.out_dir / 'C3w_summary.json').write_text(json.dumps(summ, indent=1) + '\n')

    def panel(ax, shape_list, t):
        seen = {}                                     # y-values per kernel, to scale the panel to the adopted path
        for c in KERNELS + ('ceiling',):
            for pattern, ls in (('random', '-'), ('contiguous', '--')):
                if c == 'ceiling' and pattern == 'contiguous':
                    continue
                ys = []
                for f in FS:
                    vals = []
                    for shape in shape_list:
                        base = ref[(shape, t, OWN_STOCK[c])]['us']
                        if c == 'ceiling':
                            vals.append(pct(ref[(shape, t, 'ceiling')]['us'], base))
                        else:
                            vals.append(pct(val(shape, t, c, 'all' if f in (0.0, 1.0) else pattern, f)['us'], base))
                    ys.append(statistics.mean(vals))
                seen.setdefault(c, []).extend(ys)
                ax.plot([100 * f for f in FS], ys, ls if c != 'ceiling' else ':', color=COLOR[c], marker='o' if ls == '-' else None,
                        ms=3, label=f"{LABEL[c]}{'' if c == 'ceiling' else ', ' + pattern}")
            if c != 'ceiling':
                rf = statistics.mean(val(shape, t, c, 'real', None)['f'] for shape in shape_list)
                ry = statistics.mean(pct(val(shape, t, c, 'real', None)['us'], ref[(shape, t, OWN_STOCK[c])]['us']) for shape in shape_list)
                seen[c].append(ry)
                ax.plot([100 * rf], [ry], marker='*', ms=10, color=COLOR[c], linestyle='none')
        # The paper kernel is one 128-wide build: at small T it is far above its stock, which runs narrow tiles. Scale
        # the panel to the adopted path and the ceiling, and print the paper kernel's range when it does not fit.
        own = [v for c in ('adopted', 'adopted_freq', 'ceiling') for v in seen[c]]
        lo, hi = min(own + [0.0]), max(own + [0.0])
        top = hi + 0.25 * (hi - lo) + 1
        ax.set_ylim(lo - 0.1 * (hi - lo) - 1, top)
        if max(seen['paper']) > top:
            ax.text(0.98, 0.03, f"paper kernel: {min(seen['paper']):+.0f} … {max(seen['paper']):+.0f} % (off scale)",
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color=COLOR['paper'])
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xscale('symlog', linthresh=1)          # spreads the small shares: linear below 1 %, logarithmic above
        ax.set_xticks([0, 1, 2, 5, 10, 25, 50, 100], ['0', '1', '2', '5', '10', '25', '50', '100'])
        ax.set_title(f'T = {t}', fontsize=9)
        ax.set_xlabel('E0M3 tile share (%)', fontsize=8)
        ax.tick_params(labelsize=7)

    fig, axes = plt.subplots(len(shapes), len(tokens), figsize=(3.2 * len(tokens), 2.8 * len(shapes)), squeeze=False)
    for i, shape in enumerate(shapes):
        for j, t in enumerate(tokens):
            panel(axes[i][j], [shape], t)
        axes[i][0].set_ylabel(f'{shape}\noverhead vs own stock (%)', fontsize=8)
    axes[0][0].legend(fontsize=6, loc='upper left')
    fig.suptitle('8x64: GEMM overhead vs E0M3 tile share (stars: real TM-OPT+TC 8x64 map, typical module)', fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(args.out_dir / f'C3w_overhead.{ext}', dpi=150)
    fig, axes = plt.subplots(1, len(tokens), figsize=(3.2 * len(tokens), 3.2), squeeze=False)
    for j, t in enumerate(tokens):
        panel(axes[0][j], shapes, t)
    axes[0][0].set_ylabel('overhead vs own stock (%), mean of 3 shapes', fontsize=8)
    axes[0][0].legend(fontsize=6, loc='upper left')
    fig.suptitle('8x64 (weights on B): GEMM overhead vs E0M3 tile share, mean of 3 shapes', fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(args.out_dir / f'C3w_overhead_mean.{ext}', dpi=150)
    if args.c3k and args.c3k.exists():
        k3 = json.loads(args.c3k.read_text())['rows']
        kref = {(r['shape'], r['tokens'], r['kernel']): r for r in k3 if r['pattern'] == 'reference'}

        def kval(shape, t, kern, pattern, f):
            for r in k3:
                if (r['shape'], r['tokens'], r['kernel']) == (shape, t, kern) and \
                        (r['pattern'] == pattern or (r['pattern'] == 'all' and f in (0.0, 1.0))) and abs(r['f'] - f) < 1e-9:
                    return r
        fig, axes = plt.subplots(1, len(tokens), figsize=(3.2 * len(tokens), 3.2), squeeze=False)
        for j, t in enumerate(tokens):
            ax = axes[0][j]
            for fam, getv, gref, kern, ceil, col in (('16x64', kval, kref, 'ko_freq', 'ceiling', 'tab:purple'),
                                                    ('8x64', val, ref, 'adopted_freq', 'ceiling', 'tab:orange')):
                ys = [statistics.mean(pct(getv(sh, t, kern, 'all' if f in (0.0, 1.0) else 'random', f)['us'],
                                          gref[(sh, t, 'stock_ko')]['us']) for sh in shapes) for f in FS]
                ax.plot([100 * f for f in FS], ys, '-', color=col, marker='o', ms=3, label=f"{fam} adopted, #2's dispatch")
                yc = statistics.mean(pct(gref[(sh, t, ceil)]['us'], gref[(sh, t, 'stock_ko')]['us']) for sh in shapes)
                ax.plot([100 * f for f in FS], [yc] * len(FS), ':', color=col, label=f'{fam} no-dispatch ceiling')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xticks([0, 1, 2, 5, 10, 25, 50, 100], ['0', '1', '2', '5', '10', '25', '50', '100'])
            ax.set_title(f'T = {t}', fontsize=9)
            ax.set_xlabel('E0M3 tile share (%)', fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0][0].set_ylabel('overhead vs stock_ko (%), mean of 3 shapes', fontsize=8)
        axes[0][0].legend(fontsize=6, loc='upper left')
        fig.suptitle('16x64 (C3k, 2026-10-01) vs 8x64 (C3w): the deployed paths, random placement', fontsize=10)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(args.out_dir / f'C3w_vs_C3k_mean.{ext}', dpi=150)
    print('\n'.join(md[:40]))


if __name__ == '__main__':
    main()

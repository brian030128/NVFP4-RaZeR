#!/usr/bin/env python3
"""Tables and figures of the E0M3-fraction sweep on the adopted paths of the three units (C3v; c3v_fraction.py;
results/kernel_opt/c3v/PROTOCOL.md, amendment 19). Per unit it is c3k_analyze.py / c3w_analyze.py, with the same tables
and figure layout.

    python experiments/kernel_opt/c3v_analyze.py --src DIR --out-dir results/kernel_opt/c3v \
        [--c3k results/kernel_opt/c3k/C3k_raw.json --c3w results/kernel_opt/c3w/C3w_raw.json]

--src holds c3v_256x64.json, c3v_16x64.json and c3v_8x64.json; a unit without its file is skipped.
- C3v.md:
  - a summary per unit and T (means over the 3 shapes): the adopted path against stock_ko at no, the real and every
    E0M3 tile; the tiles (the ceiling vs stock_ko) and the dispatch (the adopted path vs the ceiling) at no E0M3 tile;
    the adopted path against the previous one;
  - per unit: the decomposition at no E0M3 tile per shape and T; per kernel, the overhead vs its own stock,
    kernel(f) / kernel(0) - 1 and the overhead vs stock_ko, per shape x T x pattern; the slope (least squares on the
    random pattern, f in 0 .. 1) and the adopted path against the previous one at each f;
  - with --c3k / --c3w, the cross-session check: this run's previous path, paper kernel, ceiling and stocks against the
    same builds in C3k (16x64, 2026-10-01) and C3w (8x64, 2026-10-03).
- C3v.csv: every timed row. C3v_summary.json: the derived numbers.
- C3v_<unit>_overhead.png / .pdf: the overhead vs own stock against f. Rows are shapes, columns are T, one line per
  kernel (random solid, contiguous dashed, the real map as a star).
- C3v_<unit>_overhead_mean.png / .pdf: the same, averaged over the shapes (C3k's layout).
- C3_units_mean.png / .pdf: the adopted paths of the three units from this run, mean of the shapes, overhead vs stock_ko,
  random placement, with their no-dispatch ceilings and the real maps (stars).
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

UNITS = ('256x64', '16x64', '8x64')
KERNELS = ('adopted', 'previous', 'paper')
# Each kernel against its own stock: the adopted paths, their ceilings and the previous 16x64 / 8x64 paths (on the adopted
# table) against stock_ko; A' and the paper kernels (on the paper table) against the paper stock.
OWN_STOCK = {u: dict(adopted='stock_ko', previous='stock_ko', paper='stock', ceiling='stock_ko') for u in UNITS}
OWN_STOCK['256x64']['previous'] = 'stock'
LABEL = {'256x64': dict(adopted='adopted 256x64 (mixed256_ko)', previous="previous: A' (mixed256, paper table)",
                        paper='paper kernel (n16k64_wA, 128-wide)'),
         '16x64': dict(adopted='adopted 16x64 (mixed_ko)', previous="previous: #2's dispatch alone (build_7freq)",
                       paper="paper set ('mixed', paper table)"),
         '8x64': dict(adopted='adopted 8x64 (mixed_wB_ko)', previous="previous: #2's dispatch alone (build_P3freq)",
                      paper='paper kernel (n8k64_wB, 128-wide)')}
for _u in UNITS:
    LABEL[_u]['ceiling'] = 'no-dispatch ceiling (adopted tiles)'
PREV = {'256x64': 'A′', '16x64': 'the path before amendment 17', '8x64': 'the path before amendment 17'}
COLOR = dict(adopted='tab:red', previous='tab:blue', paper='tab:gray', ceiling='tab:green')
UNIT_COLOR = {'16x64': 'tab:purple', '8x64': 'tab:orange', '256x64': 'tab:red'}
# the cross-session check: this run's kernel -> the same builds' kernel name in C3k / C3w
OLD = {'16x64': ('C3k', dict(previous='ko_freq', paper='paper', ceiling='ceiling', stock_ko='stock_ko', stock='stock')),
       '8x64': ('C3w', dict(previous='adopted_freq', paper='paper', ceiling='ceiling', stock_ko='stock_ko',
                            stock_wB_ko='stock_wB_ko', stock='stock'))}
FS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)


def pct(a, b):
    return 100 * (a / b - 1)


def finder(rows):
    def val(shape, t, kern, pattern, f):
        for r in rows:
            if (r['shape'], r['tokens'], r['kernel']) == (shape, t, kern) and \
                    (r['pattern'] == pattern or (r['pattern'] == 'all' and f in (0.0, 1.0))) and \
                    (f is None or abs(r['f'] - f) < 1e-9):
                return r
        return None
    return val


def at(val, shape, t, c, f, pattern='random'):
    return val(shape, t, c, 'all' if f in (0.0, 1.0) else pattern, f)


class Unit:
    def __init__(self, unit, rec):
        self.unit, self.rec, self.rows = unit, rec, rec['rows']
        self.shapes = list(dict.fromkeys(r['shape'] for r in self.rows))
        self.tokens = sorted({r['tokens'] for r in self.rows})
        self.ref = {(r['shape'], r['tokens'], r['kernel']): r for r in self.rows if r['pattern'] == 'reference'}
        self.val = finder(self.rows)
        self.own = OWN_STOCK[unit]

    def base(self, shape, t, c):
        return self.ref[(shape, t, self.own[c])]['us']

    def us(self, shape, t, c, f, pattern='random'):
        return self.ref[(shape, t, c)]['us'] if c == 'ceiling' else at(self.val, shape, t, c, f, pattern)['us']

    def real(self, shape, t, c):
        return self.val(shape, t, c, 'real', None)


def summary_rows(u):
    out = {}
    for t in u.tokens:
        m = lambda fn: statistics.mean(fn(sh) for sh in u.shapes)  # noqa: E731
        sk = lambda sh: u.ref[(sh, t, 'stock_ko')]['us']  # noqa: E731
        out[t] = dict(
            adopted_f0=m(lambda sh: pct(u.us(sh, t, 'adopted', 0.0), sk(sh))),
            adopted_real=m(lambda sh: pct(u.real(sh, t, 'adopted')['us'], sk(sh))),
            adopted_f100=m(lambda sh: pct(u.us(sh, t, 'adopted', 1.0), sk(sh))),
            tiles=m(lambda sh: pct(u.ref[(sh, t, 'ceiling')]['us'], sk(sh))),
            dispatch=m(lambda sh: pct(u.us(sh, t, 'adopted', 0.0), u.ref[(sh, t, 'ceiling')]['us'])),
            vs_previous_f0=m(lambda sh: pct(u.us(sh, t, 'adopted', 0.0), u.us(sh, t, 'previous', 0.0))),
            vs_previous_real=m(lambda sh: pct(u.real(sh, t, 'adopted')['us'], u.real(sh, t, 'previous')['us'])),
            vs_previous_f100=m(lambda sh: pct(u.us(sh, t, 'adopted', 1.0), u.us(sh, t, 'previous', 1.0))),
            real_share=m(lambda sh: 100 * u.real(sh, t, 'adopted')['f']))
    return out


def unit_tables(u, md, summ):
    unit, val, ref, L = u.unit, u.val, u.ref, LABEL[u.unit]
    wb = unit == '8x64'
    md += [f'## {unit}\n',
           f"Bitwise checks: {sum(c['equal'] for c in u.rec['checks'])} of {len(u.rec['checks'])} equal (adopted, "
           f"previous and paper, every map, shape and T). Kernels: {L['adopted']}; {L['previous']}; {L['paper']}.\n",
           '### Decomposition at no E0M3 tile\n',
           'tiles = the ceiling vs stock_ko (the adopted tiles, dispatch compiled out); dispatch = the adopted path vs the '
           'ceiling' + ('; placement = stock_wB_ko vs stock_ko' if wb else '') + '. "previous" is against its own stock '
           f"(`{u.own['previous']}`).\n",
           '| shape | T | width (adopted / previous / paper) | adopted vs stock_ko | tiles: ceiling vs stock_ko | '
           'dispatch: adopted vs ceiling | previous vs own stock | paper vs paper stock | adopted vs previous | '
           + ('placement: stock_wB_ko vs stock_ko | ' if wb else '') + 'stock_ko vs paper stock |',
           '|---|---:|---|' + '---:|' * (7 + wb)]
    dec = summ.setdefault('decomposition', {})
    for shape in u.shapes:
        for t in u.tokens:
            sk, sp, ce = ref[(shape, t, 'stock_ko')]['us'], ref[(shape, t, 'stock')]['us'], ref[(shape, t, 'ceiling')]['us']
            ad0, pr0, pa0 = (val(shape, t, c, 'all', 0.0) for c in KERNELS)
            d = dict(adopted=pct(ad0['us'], sk), ceiling=pct(ce, sk), dispatch=pct(ad0['us'], ce),
                     previous=pct(pr0['us'], u.base(shape, t, 'previous')), paper=pct(pa0['us'], sp),
                     adopted_vs_previous=pct(ad0['us'], pr0['us']), stock_ko_vs_stock=pct(sk, sp))
            if wb:
                d['placement'] = pct(ref[(shape, t, 'stock_wB_ko')]['us'], sk)
            dec[f'{shape}@{t}'] = d
            md.append(f"| {shape} | {t} | {ad0['width']} / {pr0['width']} / {pa0['width']} | {d['adopted']:+.1f} | "
                      f"{d['ceiling']:+.1f} | {d['dispatch']:+.1f} | {d['previous']:+.1f} | {d['paper']:+.1f} | "
                      f"{d['adopted_vs_previous']:+.1f} | " + (f"{d['placement']:+.1f} | " if wb else '')
                      + f"{d['stock_ko_vs_stock']:+.1f} |")
    md.append('')
    for kind, title in (('own', 'overhead vs own stock'), ('rel', 'kernel(f) / kernel(0) − 1'),
                        ('ko', 'overhead vs stock_ko (the target)')):
        md.append(f'### {unit}: {title}, in %\n')
        for c in KERNELS:
            md.append(f'#### {L[c]}\n')
            md.append('| shape | T | pattern | ' + ' | '.join(f'{round(100 * f)} %' for f in FS) + ' | real map (f) |')
            md.append('|---|---:|---|' + '---:|' * (len(FS) + 1))
            for shape in u.shapes:
                for t in u.tokens:
                    base = (u.base(shape, t, c) if kind == 'own' else ref[(shape, t, 'stock_ko')]['us']
                            if kind == 'ko' else val(shape, t, c, 'all', 0.0)['us'])
                    real = u.real(shape, t, c)
                    for pattern in ('random', 'contiguous'):
                        cells = [f"{pct(at(val, shape, t, c, f, pattern)['us'], base):+.1f}" for f in FS]
                        rcell = f"{pct(real['us'], base):+.1f} ({100 * real['f']:.1f} %)" if pattern == 'random' else ''
                        md.append(f'| {shape} | {t} | {pattern} | ' + ' | '.join(cells) + f' | {rcell} |')
            md.append('')
    md.append(f'### {unit}: slope, and the adopted path against {PREV[unit]}\n')
    md.append('Slope: least-squares percentage points of overhead vs own stock per unit E0M3 share (random pattern, '
              'f = 0 … 1). Then adopted / previous − 1 in % at each f (random pattern; negative: the adopted path is '
              'faster).\n')
    md.append('| shape | T | slope adopted | slope previous | slope paper | '
              + ' | '.join(f'adopted vs previous @ {round(100 * f)} %' for f in FS) + ' |')
    md.append('|---|---:|---:|---:|---:|' + '---:|' * len(FS))
    for shape in u.shapes:
        for t in u.tokens:
            slopes = {}
            for c in KERNELS:
                ys = [pct(at(val, shape, t, c, f)['us'], u.base(shape, t, c)) for f in FS]
                mf, my = statistics.mean(FS), statistics.mean(ys)
                slopes[c] = sum((f - mf) * (y - my) for f, y in zip(FS, ys)) / sum((f - mf) ** 2 for f in FS)
            rel = [pct(at(val, shape, t, 'adopted', f)['us'], at(val, shape, t, 'previous', f)['us']) for f in FS]
            summ.setdefault('slope', {})[f'{shape}@{t}'] = slopes
            summ.setdefault('vs_previous', {})[f'{shape}@{t}'] = dict(zip([str(f) for f in FS], rel))
            real = {c: u.real(shape, t, c) for c in KERNELS}
            summ.setdefault('real', {})[f'{shape}@{t}'] = dict(
                f=real['adopted']['f'], **{c: pct(real[c]['us'], u.base(shape, t, c)) for c in KERNELS},
                adopted_vs_stock_ko=pct(real['adopted']['us'], ref[(shape, t, 'stock_ko')]['us']),
                adopted_vs_previous=pct(real['adopted']['us'], real['previous']['us']))
            md.append(f"| {shape} | {t} | {slopes['adopted']:+.2f} | {slopes['previous']:+.2f} | {slopes['paper']:+.2f} | "
                      + ' | '.join(f'{v:+.1f}' for v in rel) + ' |')
    md.append('')


def cross_session(u, old_rows, md):
    """This run against C3k / C3w on the same builds: new / old - 1 per matched cell, and whether the width and the
    scheduler row were the same."""
    name, pairs = OLD[u.unit]
    oval = finder(old_rows)
    oref = {(r['shape'], r['tokens'], r['kernel']): r for r in old_rows if r['pattern'] == 'reference'}
    out = {}
    md.append(f'### {u.unit}: this run against {name} (the same builds)\n')
    md.append(f'Per matched cell, this run / {name} − 1 in %; for the tagged kernels every map (random, contiguous, '
              'all, real). Widths and scheduler rows as recorded per row.\n')
    md.append(f'| this run | {name} | cells | median | min | max | same width and scheduler row |')
    md.append('|---|---|---:|---:|---:|---:|---:|')
    for new, oldk in pairs.items():
        d, same, n = [], 0, 0
        for r in u.rows:
            if r['kernel'] != new:
                continue
            o = (oref.get((r['shape'], r['tokens'], oldk)) if r['pattern'] == 'reference' else
                 oval(r['shape'], r['tokens'], oldk, r['pattern'], r['f']))
            if o is None:
                continue
            n += 1
            d.append(pct(r['us'], o['us']))
            same += (r.get('width'), r.get('schedule')) == (o.get('width'), o.get('schedule'))
        if d:
            out[new] = dict(old=oldk, cells=n, median=statistics.median(d), min=min(d), max=max(d), same_width_schedule=same)
            md.append(f'| {new} | {oldk} | {n} | {statistics.median(d):+.2f} | {min(d):+.2f} | {max(d):+.2f} | {same} of {n} |')
    md.append('')
    return out


def panel(ax, u, shape_list, t):
    seen = {}
    for c in KERNELS + ('ceiling',):
        for pattern, ls in (('random', '-'), ('contiguous', '--')):
            if c == 'ceiling' and pattern == 'contiguous':
                continue
            ys = [statistics.mean(pct(u.us(sh, t, c, f, pattern), u.base(sh, t, c)) for sh in shape_list) for f in FS]
            seen.setdefault(c, []).extend(ys)
            ax.plot([100 * f for f in FS], ys, ls if c != 'ceiling' else ':', color=COLOR[c], marker='o' if ls == '-' else None,
                    ms=3, label=f"{LABEL[u.unit][c]}{'' if c == 'ceiling' else ', ' + pattern}")
        if c != 'ceiling':
            rf = statistics.mean(u.real(sh, t, c)['f'] for sh in shape_list)
            ry = statistics.mean(pct(u.real(sh, t, c)['us'], u.base(sh, t, c)) for sh in shape_list)
            seen[c].append(ry)
            ax.plot([100 * rf], [ry], marker='*', ms=10, color=COLOR[c], linestyle='none')
    # a one-width paper kernel is far above its stock at small T, where the stock runs narrow tiles. Scale the panel to
    # the adopted path, the previous one and the ceiling, and print the paper kernel's range when it does not fit.
    own = [v for c in ('adopted', 'previous', 'ceiling') for v in seen[c]]
    lo, hi = min(own + [0.0]), max(own + [0.0])
    top = hi + 0.25 * (hi - lo) + 1
    bottom = lo - 0.1 * (hi - lo) - 1
    ax.set_ylim(bottom, top)
    if max(seen['paper']) > top or min(seen['paper']) < bottom:
        ax.text(0.98, 0.03, f"paper kernel: {min(seen['paper']):+.0f} … {max(seen['paper']):+.0f} % (off scale)",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color=COLOR['paper'])
    axis(ax, t)


def axis(ax, t):
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xticks([0, 1, 2, 5, 10, 25, 50, 100], ['0', '1', '2', '5', '10', '25', '50', '100'])
    ax.set_title(f'T = {t}', fontsize=9)
    ax.set_xlabel('E0M3 tile share (%)', fontsize=8)
    ax.tick_params(labelsize=7)


def unit_figures(u, out_dir):
    tok = u.tokens
    fig, axes = plt.subplots(len(u.shapes), len(tok), figsize=(3.2 * len(tok), 2.8 * len(u.shapes)), squeeze=False)
    for i, shape in enumerate(u.shapes):
        for j, t in enumerate(tok):
            panel(axes[i][j], u, [shape], t)
        axes[i][0].set_ylabel(f'{shape}\noverhead vs own stock (%)', fontsize=8)
    axes[0][0].legend(fontsize=6, loc='upper left')
    fig.suptitle(f'{u.unit}: GEMM overhead vs E0M3 tile share (stars: real TM-OPT+TC {u.unit} map, typical module)',
                 fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'C3v_{u.unit}_overhead.{ext}', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(tok), figsize=(3.2 * len(tok), 3.2), squeeze=False)
    for j, t in enumerate(tok):
        panel(axes[0][j], u, u.shapes, t)
    axes[0][0].set_ylabel('overhead vs own stock (%), mean of 3 shapes', fontsize=8)
    axes[0][0].legend(fontsize=6, loc='upper left')
    fig.suptitle(f'{u.unit}: GEMM overhead vs E0M3 tile share, mean of 3 shapes', fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'C3v_{u.unit}_overhead_mean.{ext}', dpi=150)
    plt.close(fig)


def units_figure(units, out_dir):
    tok = sorted(set.intersection(*(set(u.tokens) for u in units)))
    shapes = [sh for sh in units[0].shapes if all(sh in u.shapes for u in units)]
    fig, axes = plt.subplots(1, len(tok), figsize=(3.2 * len(tok), 3.4), squeeze=False)
    for j, t in enumerate(tok):
        ax = axes[0][j]
        for u in units:
            col = UNIT_COLOR[u.unit]
            sk = lambda sh: u.ref[(sh, t, 'stock_ko')]['us']  # noqa: E731
            ys = [statistics.mean(pct(u.us(sh, t, 'adopted', f), sk(sh)) for sh in shapes) for f in FS]
            ax.plot([100 * f for f in FS], ys, '-', color=col, marker='o', ms=3, label=f'{u.unit}, adopted path')
            yc = statistics.mean(pct(u.ref[(sh, t, 'ceiling')]['us'], sk(sh)) for sh in shapes)
            ax.plot([100 * f for f in FS], [yc] * len(FS), ':', color=col, label=f'{u.unit}, no-dispatch ceiling')
            rf = statistics.mean(u.real(sh, t, 'adopted')['f'] for sh in shapes)
            ry = statistics.mean(pct(u.real(sh, t, 'adopted')['us'], sk(sh)) for sh in shapes)
            ax.plot([100 * rf], [ry], marker='*', ms=10, color=col, linestyle='none')
        axis(ax, t)
    axes[0][0].set_ylabel('overhead vs stock_ko (%), mean of 3 shapes', fontsize=8)
    axes[0][0].legend(fontsize=6, loc='upper left')
    fig.suptitle('The three units: adopted (deployed) paths against stock_ko, random placement, one session (stars: the '
                 'real TM-OPT+TC maps, typical module)', fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'C3_units_mean.{ext}', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True, help='the directory with c3v_<unit>.json')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--c3k', type=Path, default=None, help="C3k's raw JSON (16x64), for the cross-session check")
    ap.add_argument('--c3w', type=Path, default=None, help="C3w's raw JSON (8x64), for the cross-session check")
    args = ap.parse_args()
    units = []
    for unit in UNITS:
        p = args.src / f'c3v_{unit}.json'
        if p.exists():
            rec = json.loads(p.read_text())
            assert rec.get('unit', '256x64') == unit, (p, rec.get('unit'))
            units.append(Unit(unit, rec))
    assert units, f'no c3v_<unit>.json in {args.src}'
    md = ['# Kernel-opt E0M3-fraction sweep on the adopted paths of the three units (C3v)\n',
          'The deviation-2 method, as C3k / C3w: cold weights, the activation quantizer after the flush, isolated CUPTI '
          'GEMM launches, 3 rotated rounds × 30; the median of all launches. Overheads in %. Shares are of the unit\'s '
          'tiles. Each kernel is against its own stock: the adopted paths, their ceilings and the previous 16x64 / 8x64 '
          'paths against `stock_ko` (the target); A′ and the paper kernels (on the paper table) against the paper '
          '`stock`.\n',
          'Previous paths: 256x64 A′ (`mixed256`, the paper table); 16x64 and 8x64 the adopted families with #2\'s '
          'dispatch alone (`build_7freq`, `build_P3freq`: C3k\'s and C3w\'s deployed paths), on the adopted table.\n',
          '## Summary: means over the 3 shapes\n',
          'adopted vs stock_ko at no E0M3 tile, at the real map (its share) and at every tile E0M3; tiles = the ceiling '
          'vs stock_ko; dispatch = the adopted path vs the ceiling at no E0M3 tile; then the adopted path against the '
          'previous one (random pattern).\n',
          '| unit | T | adopted @ 0 % | adopted @ real (share) | adopted @ 100 % | tiles | dispatch @ 0 % | '
          'vs previous @ 0 % | vs previous @ real | vs previous @ 100 % |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    summ, csv_rows = {}, []
    for u in units:
        s = summary_rows(u)
        summ[u.unit] = dict(summary={str(t): v for t, v in s.items()})
        for t, v in s.items():
            md.append(f"| {u.unit} | {t} | {v['adopted_f0']:+.1f} | {v['adopted_real']:+.1f} ({v['real_share']:.1f} %) | "
                      f"{v['adopted_f100']:+.1f} | {v['tiles']:+.1f} | {v['dispatch']:+.1f} | {v['vs_previous_f0']:+.1f} | "
                      f"{v['vs_previous_real']:+.1f} | {v['vs_previous_f100']:+.1f} |")
    md.append('')
    olds = {'16x64': args.c3k, '8x64': args.c3w}
    for u in units:
        unit_tables(u, md, summ[u.unit])
        if u.unit in olds and olds[u.unit] and olds[u.unit].exists():
            summ[u.unit]['cross_session'] = cross_session(u, json.loads(olds[u.unit].read_text())['rows'], md)
        csv_rows += [dict(unit=u.unit, **r) for r in u.rows]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'C3v.md').write_text('\n'.join(md) + '\n')
    with open(args.out_dir / 'C3v.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['unit', 'shape', 'tokens', 'kernel', 'pattern', 'f', 'kernel_build', 'width', 'schedule', 'us',
                     'round_medians'])
        for r in csv_rows:
            wr.writerow([r['unit'], r['shape'], r['tokens'], r['kernel'], r['pattern'], r['f'], r['kernel_build'],
                         r.get('width'), r['schedule'], f"{r['us']:.3f}", ' '.join(f'{v:.3f}' for v in r['rounds'])])
    (args.out_dir / 'C3v_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
    for u in units:
        unit_figures(u, args.out_dir)
    units_figure(sorted(units, key=lambda u: ('16x64', '8x64', '256x64').index(u.unit)), args.out_dir)
    print('\n'.join(md[:12 + 6 * len(units)]))


if __name__ == '__main__':
    main()

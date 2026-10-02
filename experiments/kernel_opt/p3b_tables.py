#!/usr/bin/env python3
"""Kernel-opt amendment 12b (the 8x64 plan's P3b, option C): the reduced 8x64 table and its fallback (CPU only).

    python experiments/kernel_opt/p3b_tables.py select --m1 DIR --p3 P3.json --paper PAPER.json --out CELLS.json
    python experiments/kernel_opt/p3b_tables.py compose --ko KO.json --paper PAPER.json --p3 P3.json --cells CELLS.json \
        --out-dir DIR

select: from amendment 12's M1 records (--m1, the gemm/<model>.json files), the P3 width changes kept for the reduced
table. A change at (shape, bucket), i.e. a P3 'mixed_wB' width that differs from 1b's paper row, is kept only if it
was faster in every round on both tag sets for every (model, projection) of that shape at T = bucket:
- typical tags: wBw_typical (P3's widths, no scheduler rows) against wBp2_typical (1b's rows);
- worst tags: wBko_worst (the P3 table: the width, plus P3's scheduler row where it has one) against wBp2_worst.
  M1 has no widths-only worst configuration.
'Faster in every round' means every round's per-GEMM median of the change below every round's of 1b's row. A bucket M1
did not measure (2, 8) gives no evidence, so its change is not kept. The output lists every change with its evidence,
and the kept cells.

compose writes two tables, each the tracked adopted table (--ko) with 'mixed_wB' rows and stock_wB_ko's P3 scheduler
rows added, every other key unchanged:
- table_p3b/<gpu>.json, the candidate: 1b's 'mixed_wB' rows (the paper table's) with the kept cells set to P3's widths;
  no scheduler rows for mixed_wB_ko.
- table_a/<gpu>.json, the fallback (option A): 1b's rows unchanged; no scheduler rows for mixed_wB_ko. This is P2's
  adopted path.
"""
import argparse
import copy
import json
import statistics
from pathlib import Path

TOKENS = (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')


def faster(before, after):
    rb, ra = [r['gemm_us'] for r in before['rounds']], [r['gemm_us'] for r in after['rounds']]
    return max(ra) < min(rb), 100 * (statistics.median(ra) / statistics.median(rb) - 1)


def select(args):
    p3 = json.loads(Path(args.p3).read_text())
    new, sched = p3['mixed_wB'], p3['schedule']['mixed_wB_ko']
    old = json.loads(Path(args.paper).read_text())['mixed_wB']
    rows, uses = {}, {}
    for m in MODELS:
        rec = json.loads((Path(args.m1) / f'{m}.json').read_text())
        assert rec['status'] == 'complete', m
        for r in rec['rows']:
            rows[(m, r['proj'], r['tokens'], r['config'])] = r
        for p, v in rec['projections'].items():
            uses.setdefault('%dx%d' % tuple(v['shape']), []).append((m, p))
    changes, kept = [], []
    for shape in sorted(new):
        for b in sorted(new[shape], key=int):
            if str(new[shape][b]) == str(old[shape][b]):
                continue
            t = int(b)
            ev, ok = [], t in TOKENS
            if ok:
                for m, p in uses[shape]:
                    ft, dt = faster(rows[(m, p, t, 'wBp2_typical')], rows[(m, p, t, 'wBw_typical')])
                    fw, dw = faster(rows[(m, p, t, 'wBp2_worst')], rows[(m, p, t, 'wBko_worst')])
                    ev.append(dict(model=m, proj=p, typical_pct=dt, typical_every_round=ft, worst_pct=dw,
                                   worst_every_round=fw))
                    ok &= ft and fw
            changes.append(dict(shape=shape, bucket=t, width_1b=old[shape][b], width_p3=new[shape][b],
                                p3_scheduler_row=sched[shape][b], measured=t in TOKENS, evidence=ev, kept=ok))
            if ok:
                kept.append(dict(shape=shape, bucket=t, width=new[shape][b]))
    out = dict(criterion=__doc__.split('select: ')[1].split('\n\ncompose')[0], source_m1=str(args.m1), p3_table=str(args.p3),
               paper_table=str(args.paper), changes=changes, kept=kept)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    print(f'{len(changes)} P3 width changes, {len(kept)} kept:')
    for c in kept:
        print(' ', c)


def compose(args):
    ko = json.loads(Path(args.ko).read_text())
    paper = json.loads(Path(args.paper).read_text())
    p3 = json.loads(Path(args.p3).read_text())
    cells = json.loads(Path(args.cells).read_text())['kept']
    assert 'mixed_wB' not in ko and 'mixed_wB_ko' not in ko.get('schedule', {}) and 'stock_wB_ko' not in ko.get('schedule', {})
    slug = Path(args.ko).name.split('.')[0]
    for name, rows in (('table_a', copy.deepcopy(paper['mixed_wB'])), ('table_p3b', copy.deepcopy(paper['mixed_wB']))):
        if name == 'table_p3b':
            for c in cells:
                assert str(p3['mixed_wB'][c['shape']][str(c['bucket'])]) == str(c['width'])
                rows[c['shape']][str(c['bucket'])] = c['width']
        out = copy.deepcopy(ko)
        out['mixed_wB'] = rows
        out['schedule']['stock_wB_ko'] = p3['schedule']['stock_wB_ko']
        out['meta'] = dict(ko['meta'])
        out['meta']['families'] = dict(out['meta'].get('families', {}), mixed_wB=dict(
            source=("1b's rows (the paper table)" + (f" with {len(cells)} P3 cells ({args.cells})" if name == 'table_p3b' else '')),
            note='kernel-opt amendment 12b; no scheduler rows for mixed_wB_ko'))
        out['meta']['schedule'] = dict(out['meta'].get('schedule', {}), stock_wB_ko=p3['meta']['schedule']['stock_wB_ko'])
        path = Path(args.out_dir) / name / f'{slug}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
        print('wrote', path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select')
    s.add_argument('--m1', required=True)
    s.add_argument('--p3', required=True)
    s.add_argument('--paper', required=True)
    s.add_argument('--out', required=True)
    c = sub.add_parser('compose')
    c.add_argument('--ko', required=True)
    c.add_argument('--paper', required=True)
    c.add_argument('--p3', required=True)
    c.add_argument('--cells', required=True)
    c.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    (select if args.cmd == 'select' else compose)(args)


if __name__ == '__main__':
    main()

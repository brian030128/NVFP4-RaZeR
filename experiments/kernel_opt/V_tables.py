#!/usr/bin/env python3
"""Kernel-opt amendment 18 (V): the 256x64 path's table rows (CPU only).

    python experiments/kernel_opt/V_tables.py today --out-dir DIR
        -> DIR/table_v0.json: the adopted table with 'mixed256' rows = the paper table's 'mixed' rows (today's mixed256
           widths) and no 'mixed256_ko' scheduler rows: the new builds at today's widths (M1's build effect)
    python experiments/kernel_opt/V_tables.py compose --widths DIR --out-dir DIR
        -> DIR/table_v0w.json: the adopted table with 'mixed256' rows = the width tuner's 'mixed256_ko' rows (the fastest
           median per (shape, bucket), 4b's method), for the scheduler tuning
    python experiments/kernel_opt/V_tables.py finalize --schedule DIR --out-dir DIR
        -> DIR/table_v.json: the scheduler tuner's output on table_v0w ('mixed256_ko' rows by amendment 7's decisive
           rule); checks that it differs from the adopted table only in 'mixed256', schedule['mixed256_ko'] and meta
    python experiments/kernel_opt/V_tables.py sensitivity --widths DIR --out-dir DIR
        -> DIR/sensitivity.json: the decisive-margin rule on the widths, reported only (P3's definition): a width other
           than the default replaces it only if its median is >= 0.5 % below the default's and all its rounds are below
           all of the default's; defaults 'dm128' (the 128-wide build) and 'dmfb' (select.fallback_width)
The adopted table is the tracked sm120/configs/<gpu>.ko.json; the paper table is sm120/configs/<gpu>.json.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
from mixfp4_sm120 import select as S  # noqa: E402

FAMILY, ROWS = 'mixed256_ko', 'mixed256'
KO = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
PAPER = S.TABLE_DIR / f'{S.gpu_slug()}.json'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')
    print('wrote', path)


def only_rows(table, ko, schedule_ok):
    """table differs from the adopted table only in ROWS, meta and (if schedule_ok) schedule[FAMILY]."""
    for key in set(table) | set(ko):
        if key in (ROWS, 'meta'):
            continue
        if key == 'schedule':
            for f in set(table.get('schedule', {})) | set(ko.get('schedule', {})):
                if f == FAMILY and schedule_ok:
                    continue
                assert table['schedule'].get(f) == ko['schedule'].get(f), f'schedule {f}'
            continue
        assert table.get(key) == ko.get(key), key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=('today', 'compose', 'finalize', 'sensitivity'))
    ap.add_argument('--widths', type=Path, help="tune_tiles.py --families mixed256_ko output directory")
    ap.add_argument('--schedule', type=Path, help="tune_tiles.py --schedule --families mixed256_ko output directory")
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    ko = json.loads(KO.read_text())
    assert ROWS not in ko and FAMILY not in ko.get('schedule', {}), f'{KO} already has the 256x64 rows'
    slug = S.gpu_slug()
    if args.what == 'today':
        paper = json.loads(PAPER.read_text())
        out = dict(ko)
        out[ROWS] = paper['mixed']
        out['meta'] = dict(ko['meta'], families=dict(ko['meta'].get('families', {}), **{ROWS: dict(
            note="kernel-opt amendment 18: today's mixed256 widths (the paper table's 'mixed' rows), for M1's build effect",
            source=str(PAPER))}))
        only_rows(out, ko, False)
        write(args.out_dir / 'table_v0.json', out)
    elif args.what == 'compose':
        w = json.loads((args.widths / f'{slug}.json').read_text())
        out = dict(ko)
        out[ROWS] = w[FAMILY]
        out['meta'] = dict(ko['meta'], families=dict(ko['meta'].get('families', {}), **{ROWS: dict(
            w['meta'], tuned_as=FAMILY, source=str(args.widths / f'{slug}.json'))}))
        only_rows(out, ko, False)
        write(args.out_dir / 'table_v0w.json', out)
    elif args.what == 'finalize':
        t = json.loads((args.schedule / f'{slug}.json').read_text())
        b0 = json.loads((args.out_dir / 'table_v0w.json').read_text())
        assert t[ROWS] == b0[ROWS], 'the scheduler tuner changed the width rows'
        assert FAMILY in t['schedule'], 'no scheduler rows'
        only_rows(t, ko, True)
        write(args.out_dir / 'table_v.json', t)
        rows = t['schedule'][FAMILY]
        n = sum(len(r) for r in rows.values())
        moved = sum(1 for r in rows.values() for v in r.values() if tuple(v) != (0, 1))
        print(f'{FAMILY}: {n} scheduler cells, {moved} not (0, 1)')
    else:
        raw = json.loads((args.widths / f'{slug}.raw.json').read_text())['us_rounds'][FAMILY]
        chosen = json.loads((args.widths / f'{slug}.json').read_text())[FAMILY]
        tables, changed = {'dm128': {}, 'dmfb': {}}, {'dm128': [], 'dmfb': []}

        def decisive(rounds, best, default):
            if best == default:
                return False
            mb, md = statistics.median(rounds[best]), statistics.median(rounds[default])
            return mb <= 0.995 * md and max(rounds[best]) < min(rounds[default])
        for shape, by_bucket in raw.items():
            for b, rounds in by_bucket.items():
                med = {w: statistics.median(v) for w, v in rounds.items()}
                fastest = str(chosen[shape][b])
                for name, default in (('dm128', '128'), ('dmfb', str(S.fallback_width(int(b))))):
                    pick = fastest if decisive(rounds, fastest, default) else default
                    tables[name].setdefault(shape, {})[b] = S.parse_key(pick)
                    if pick != fastest:
                        changed[name].append(dict(shape=shape, bucket=int(b), fastest=fastest, rule=pick,
                                                  rule_vs_fastest_pct=100 * (med[pick] / med[fastest] - 1),
                                                  fastest_rounds=rounds[fastest], rule_rounds=rounds[pick]))
        res = dict(rule="a non-default width replaces the default only if its median is >= 0.5 % below the default's and "
                        "all its rounds are below all of the default's (reported only, as for 8x64)",
                   cells=sum(len(v) for v in raw.values()), tables={k: {ROWS: v} for k, v in tables.items()},
                   changed=changed,
                   summary={k: dict(cells_changed=len(v),
                                    median_pct=statistics.median([c['rule_vs_fastest_pct'] for c in v]) if v else None,
                                    max_pct=max([c['rule_vs_fastest_pct'] for c in v]) if v else None)
                            for k, v in changed.items()})
        write(args.out_dir / 'sensitivity.json', res)
        print(res['summary'])


if __name__ == '__main__':
    main()

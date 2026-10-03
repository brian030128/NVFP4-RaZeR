#!/usr/bin/env python3
"""Kernel-opt amendment 17, part B: compose the re-tuned 16x64 width cells into a copy of the adopted table.

    python experiments/kernel_opt/U_tables.py widths --tuned DIR --out-dir DIR
        -> DIR/table_b0.json: the adopted table (sm120/configs/<gpu>.ko.json) with the tuned cells' 'mixed' widths;
           DIR/cells.json: per cell the adopted and the tuned width, and whether it changed; prints the changed cells
           as tune_tiles.py --cells (empty if none)
    python experiments/kernel_opt/U_tables.py finalize --out-dir DIR [--scheduled DIR]
        -> DIR/table_b.json: table_b0 with the changed cells' 'mixed_ko' scheduler rows from --scheduled (tune_tiles.py
           --schedule --cells on table_b0); without --scheduled (no cell changed), table_b0 itself. Checks that it
           differs from the adopted table only in those cells (and in meta).

The cells (registered): the 16x64 cells where the adopted table's 'mixed' and 'stock' rows choose different widths and
one of the two is 64 or 128, the widths whose builds part A changes (the seventh disagreeing cell, 17408x5120 at T = 32,
is 32 against 16, two builds part A leaves as they are).
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(REPO / 'sm120'))
from mixfp4_sm120 import select as S  # noqa: E402

KO = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
CELLS = [('4096x4096', '128'), ('4096x14336', '128'), ('17408x5120', '128'), ('1024x4096', '512'), ('1024x5120', '512'),
         ('12288x5120', '512')]


def disagreeing(table):
    """The (shape, bucket) cells where 'mixed' and 'stock' choose different widths, one of them 64 or 128."""
    return sorted((s, b) for s, row in table['mixed'].items() for b, w in row.items()
                  if str(table['stock'][s][b]) != str(w) and {str(table['stock'][s][b]), str(w)} & {'64', '128'})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=('widths', 'finalize'))
    ap.add_argument('--tuned', type=Path, help="tune_tiles.py --families mixed_ko --cells ... output directory")
    ap.add_argument('--scheduled', type=Path, help="tune_tiles.py --schedule --cells ... output directory")
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    ko = json.loads(KO.read_text())
    assert disagreeing(ko) == sorted(CELLS), (disagreeing(ko), CELLS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = S.gpu_slug()
    if args.what == 'widths':
        tuned = json.loads((args.tuned / f'{slug}.json').read_text())
        rows = tuned['mixed_ko']
        got = sorted((s, b) for s, row in rows.items() for b in row)
        assert got == sorted(CELLS), got
        new = json.loads(KO.read_text())
        cells = []
        for s, b in CELLS:
            old_w, new_w = new['mixed'][s][b], rows[s][b]
            new['mixed'][s][b] = new_w
            cells.append(dict(shape=s, bucket=int(b), adopted=old_w, stock=ko['stock'][s][b], tuned=new_w,
                              changed=str(old_w) != str(new_w)))
        new['meta'].setdefault('families', {})['mixed'] = dict(
            note='kernel-opt amendment 17, part B: the adopted rows with the re-tuned cells', cells=cells,
            tuned=str(args.tuned / f'{slug}.json'), method=tuned['meta'].get('method'), kernels=tuned['meta'].get('kernels'))
        (args.out_dir / 'table_b0.json').write_text(json.dumps(new, indent=1, sort_keys=True) + '\n')
        (args.out_dir / 'cells.json').write_text(json.dumps(cells, indent=1) + '\n')
        changed = [f"{c['shape']}@{c['bucket']}" for c in cells if c['changed']]
        print(','.join(changed))
        return
    b0 = json.loads((args.out_dir / 'table_b0.json').read_text())
    cells = json.loads((args.out_dir / 'cells.json').read_text())
    changed = {(c['shape'], str(c['bucket'])) for c in cells if c['changed']}
    if changed:
        sch = json.loads((args.scheduled / f'{slug}.json').read_text())
        for s, b in changed:
            b0['schedule']['mixed_ko'].setdefault(s, {})[b] = sch['schedule']['mixed_ko'][s][b]
        b0['meta'].setdefault('schedule', {})['mixed_ko_part_b'] = sch['meta']['schedule']['mixed_ko']
    else:
        assert args.scheduled is None, 'no cell changed: nothing to schedule'
    # it differs from the adopted table only in the cells
    for fam in ko:
        if fam in ('meta', 'mixed', 'schedule'):
            continue
        assert b0[fam] == ko[fam], fam
    for s, row in ko['mixed'].items():
        for b, w in row.items():
            assert str(b0['mixed'][s][b]) == str(w) or (s, b) in set(CELLS), (s, b)
    for f, rows in ko['schedule'].items():
        for s, row in rows.items():
            for b, v in row.items():
                assert b0['schedule'][f][s][b] == v or (f == 'mixed_ko' and (s, b) in changed), (f, s, b)
    (args.out_dir / 'table_b.json').write_text(json.dumps(b0, indent=1, sort_keys=True) + '\n')
    print('table_b.json:', len(changed), 'cells changed width:', sorted(changed))


if __name__ == '__main__':
    main()

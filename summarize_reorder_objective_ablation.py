"""Collect the objective-ablation reports into one comparison table.

Reads report.json files written by run_reorder_objective_ablation.py. Pure file
aggregation: no search, no model, no new selection. The ratio column is the
decision-relevant one, because each variant normalizes its objective by its own
fit-derived scales and absolute values are not comparable across variants.
"""
import argparse
import json
from pathlib import Path

from run_reorder_objective_ablation import MATCHED_NULL, VARIANTS


def collect(root):
    rows = []
    for path in sorted(root.rglob('report.json')):
        report = json.loads(path.read_text())
        if report.get('status') != 'complete' or 'variant' not in report:
            continue
        rows.append(report)
    return rows


def excess_over_placebo(rows):
    """Fit objective relative to the variant's OWN matched null, per matrix.

    A variant whose real fit objective barely exceeds its own null is fitting
    noise, whatever its absolute value. The null must be matched, because
    collapsing the CE/KL conjunction onto one channel removes a constraint and
    raises the attainable objective by itself; scoring `kl` against the `ce_kl`
    null would credit it for that.
    """
    nulls = {key(r) + (r['variant'],): r['fit_objective'] for r in rows}
    out = {}
    for row in rows:
        matched = MATCHED_NULL.get(row['variant'])
        if matched is None:
            continue
        null = nulls.get(key(row) + (matched,))
        if null is None:
            continue
        out[key(row) + (row['variant'],)] = (
            row['fit_objective'] / null if null > 0 else float('inf'))
    return out


def key(report):
    """Identify a cell by matrix, type-tile shape AND fitting method.

    A sweep puts several tile shapes and grouping methods under one root, and a
    null is only a null for its own cell. Shrinking the tile changes how many
    atoms each bound is computed over, and changing the method changes how much
    search freedom the null gets to exploit, which is precisely what is being
    compared. `spectral` runs the deployed search with its refinement disabled,
    so it reports rounds=0; keying on method keeps it distinct anyway.
    """
    config = report.get('config', {})
    return (report['module'], config.get('tile_rows'), config.get('tile_cols'),
            report.get('method', 'search'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()

    rows = collect(args.root)
    if not rows:
        raise SystemExit(f'No complete ablation reports under {args.root}')
    ratios = excess_over_placebo(rows)

    header = (f'{"module":<26s} {"tile":>8s} {"method":<10s} {"variant":<10s} '
              f'{"fit":>12s} {"election":>12s} {"elect/fit":>10s} {"identity":>11s} '
              f'{"beats id":>9s} {"tiles":>6s} {"id tiles":>9s} {"fit/null":>9s}')
    print(header)
    print('-' * len(header))
    for cell in sorted({key(r) for r in rows}, key=lambda k: (k[0], -(k[1] or 0), k[3])):
        module, tile_rows, tile_cols, method = cell
        for variant in VARIANTS:
            found = [r for r in rows if key(r) == cell and r['variant'] == variant]
            if not found:
                continue
            r = found[0]
            ratio = r['election_fit_ratio']
            null = ratios.get(cell + (variant,))
            print(f'{module[-26:]:<26s} {f"{tile_rows}x{tile_cols}":>8s} {method:<10s} '
                  f'{variant:<10s} {r["fit_objective"]:>12.4f} '
                  f'{r["election_objective"]:>12.4f} '
                  f'{"n/a" if ratio is None else f"{ratio:>10.4f}":>10s} '
                  f'{r["election_identity_objective"]:>11.4f} '
                  f'{str(r["beats_identity_on_election"]):>9s} '
                  f'{r["elected_tiles"]:>6d} {r["identity_elected_tiles"]:>9d} '
                  f'{"-" if null is None else f"{null:>9.2f}":>9s}')
        print()

    summary = dict(reports=len(rows), root=str(args.root.resolve()), rows=rows,
                   fit_over_placebo={f'{m}|{tr}x{tc}|{me}|{v}': x
                                     for (m, tr, tc, me, v), x in ratios.items()})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + '\n')
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()

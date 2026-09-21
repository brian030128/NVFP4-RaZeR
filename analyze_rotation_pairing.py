"""Does Hadamard rotation pair better with E0M3 tiles than with E2M1 tiles?

There is a mechanical reason to expect so. E2M1 is log spaced with coarse top
codes, so an outlier inside a block is absorbed by the sparse top of the grid.
E0M3 is uniform with no such headroom, and its scale is `block_max / 7`, so one
outlier costs resolution everywhere in the block. Rotation suppresses exactly
that, which predicts an asymmetry: rotating the tiles that elected E0M3 should
help more than rotating the tiles that stayed E2M1.

Job 404773 already measured both arms and its report retains per-document losses,
but only the `mix` arms were paired. This pairs every arm on the same 64
development documents, so the asymmetry is readable:

    <scope>_unrot   the frozen arrangement with no rotation
    <scope>_mix     rotate the tiles that elected E0M3
    <scope>_e2      rotate the tiles that stayed E2M1

`mix` and `e2` are compared against `unrot`, which holds the arrangement, the
election and the background fixed so only the rotated set differs. That is the
comparison the original report did not make.

Reused development documents. Nothing here is fresh data, and nothing is
promoted: job 404773 recorded `selected = None` and did not pass its calibration
check. This only asks whether the E0M3 pairing is real, to decide whether the
closed rotation branch deserves reopening under finite-loss selection.
"""
import argparse
import json
import math
from pathlib import Path


def paired(values, reference):
    n = len(values)
    differences = [a - b for a, b in zip(values, reference)]
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    se = math.sqrt(variance / n)
    return dict(n=n, mean=mean, se=se, t=mean / se if se else float('nan'),
                upper_2se=mean + 2 * se)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    report = json.loads(args.report.read_text())
    metrics, sources = report['metrics'], report['sequence_sources']

    def series(name, quantity='ce'):
        value = metrics[name]
        return value[quantity] if isinstance(value, dict) else value

    rows = []
    for scope in ('up', 'down', 'both'):
        unrot = f'{scope}_unrot'
        if unrot not in metrics:
            continue
        for arm in ('mix', 'e2'):
            name = f'{scope}_{arm}'
            if name not in metrics:
                continue
            for reference, label in ((unrot, 'vs unrotated'), ('raw256', 'vs raw256')):
                for domain in ('all', 'math', 'code'):
                    idx = [i for i, s in enumerate(sources) if domain == 'all' or s == domain]
                    stat = paired([series(name)[i] for i in idx],
                                  [series(reference)[i] for i in idx])
                    rows.append(dict(scope=scope, arm=arm, reference=label,
                                     domain=domain, **stat))

    header = f'{"scope":<6s} {"arm":<5s} {"reference":<14s} {"domain":<6s} {"n":>4s} {"mean dCE":>12s} {"SE":>11s} {"t":>8s}'
    print(header)
    print('-' * len(header))
    for row in rows:
        if row['domain'] != 'all' and row['reference'] != 'vs unrotated':
            continue
        print(f'{row["scope"]:<6s} {row["arm"]:<5s} {row["reference"]:<14s} '
              f'{row["domain"]:<6s} {row["n"]:>4d} {row["mean"]:>+12.6f} '
              f'{row["se"]:>11.6f} {row["t"]:>+8.2f}')

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(
            source=str(args.report.resolve()), source_job=report.get('job_id'),
            selected_by_source=report.get('selected'),
            passed_calibration_check=report.get('passed_calibration_check'),
            development_only=True, rows=rows), indent=2) + '\n')
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()

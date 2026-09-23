"""Pair the native run against the simulator, window by window.

Aggregate perplexities can agree while per-window losses differ, and they can
disagree by an amount that is entirely within window-to-window spread. The
simulated run retains its per-window negative log likelihoods, so the honest
comparison is paired: same window, same policy, native minus simulated.

The pairing is only legitimate if both runs saw the same tokens, so every
window's `token_sha256` is checked against the simulated report's record before
any statistic is computed. A mismatch aborts rather than reporting a delta
between different evaluation sets.

Reported per domain: the mean paired NLL difference with its two-standard-error
interval, and the aggregate perplexities either side. The interval is descriptive
across windows and carries no multiplicity adjustment.
"""
import argparse
import json
import math
from pathlib import Path


def paired(native, simulated):
    n = len(native)
    differences = [a - b for a, b in zip(native, simulated)]
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    se = math.sqrt(variance / n)
    return dict(windows=n, mean=mean, se=se, two_se=2 * se,
                interval=[mean - 2 * se, mean + 2 * se],
                t=mean / se if se else float('nan'),
                agrees_within_2se=abs(mean) < 2 * se)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--native', type=Path, required=True)
    ap.add_argument('--simulated', type=Path, required=True)
    ap.add_argument('--native-policy', default='arranged')
    ap.add_argument('--simulated-policy', default='refined')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()

    native = json.loads(args.native.read_text())
    simulated = json.loads(args.simulated.read_text())
    assert native['status'] == 'complete' and simulated['status'] == 'complete'
    assert native['revision'] == simulated['revision'], 'different checkpoint revisions'

    # The windows must be the same tokens, not merely the same count.
    for domain in ('wiki', 'c4_paper'):
        mine = native['data'][domain]['token_sha256']
        theirs = simulated['data'][domain]['token_sha256']
        assert mine == theirs, f'{domain} token windows differ between the runs'

    rows = {}
    for domain, key in (('wiki', 'wiki'), ('c4_paper', 'c4')):
        a = native['evaluation'][args.native_policy][key]
        b = simulated['evaluation'][args.simulated_policy][key]
        assert len(a['nll']) == len(b['nll'])
        rows[key] = dict(paired=paired(a['nll'], b['nll']),
                         native_ppl=a['ppl'], simulated_ppl=b['ppl'],
                         ppl_difference=a['ppl'] - b['ppl'])

    header = (f'{"domain":<6s} {"windows":>8s} {"mean dNLL":>12s} {"2SE":>11s} '
              f'{"t":>8s} {"within 2SE":>11s} {"native PPL":>12s} {"sim PPL":>11s} {"dPPL":>10s}')
    print(header)
    print('-' * len(header))
    for key, row in rows.items():
        p = row['paired']
        print(f'{key:<6s} {p["windows"]:>8d} {p["mean"]:>+12.6f} {p["two_se"]:>11.6f} '
              f'{p["t"]:>+8.2f} {str(p["agrees_within_2se"]):>11s} '
              f'{row["native_ppl"]:>12.6f} {row["simulated_ppl"]:>11.6f} '
              f'{row["ppl_difference"]:>+10.6f}')

    summary = dict(native_report=str(args.native.resolve()),
                   simulated_report=str(args.simulated.resolve()),
                   native_policy=args.native_policy, simulated_policy=args.simulated_policy,
                   native_job=native.get('job'), simulated_job=simulated.get('job_id'),
                   token_windows_identical=True, domains=rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + '\n')
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Kernel-opt amendment 12 (the 8x64 plan's P3, with P4): the adopted table's 8x64 rows, and the decisive-margin
sensitivity of the width choice (CPU only).

    python experiments/kernel_opt/p3_tables.py compose --ko KO.json --widths W.json --out OUT.json
    python experiments/kernel_opt/p3_tables.py sensitivity --raw W.raw.json --widths W.json --out SENS.json

compose: the adopted table is the tracked '<gpu>.ko.json' with 'mixed_wB' rows added. Those are the width tuner's
'mixed_wB_ko' rows, the fastest median per (shape, bucket), as 4b. Every other key is kept byte for byte, and the
tuner's meta is recorded under meta['families']['mixed_wB'].

sensitivity: the width tuner's per-round times (us_rounds) give, per (shape, bucket), the table the decisive-margin rule
would have chosen. A width other than the default replaces the default only if its median is at least 0.5 % below the
default's and every one of its rounds is below every round of the default's (amendment 7's rule for scheduler rows).
Two defaults are used:
- 'dm128', the single 128-wide build (n8k64_wB): the rule's original proposal;
- 'dmfb', the fallback width (select.fallback_width: the narrowest power of two holding the bucket's tokens, capped at 128).
The output gives both tables ({'mixed_wB': rows}, for KernelSet), and every cell where the rule's choice differs from
the fastest. Each such cell lists both widths with their median and per-round times, and the difference.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
from mixfp4_sm120 import select as S  # noqa: E402

FAMILY, ROWS = 'mixed_wB_ko', 'mixed_wB'


def compose(args):
    ko = json.loads(Path(args.ko).read_text())
    w = json.loads(Path(args.widths).read_text())
    assert ROWS not in ko, f'{args.ko} already has {ROWS!r} rows'
    out = dict(ko)
    out[ROWS] = w[FAMILY]
    out['meta'] = dict(ko['meta'])
    out['meta'].setdefault('families', {})
    out['meta']['families'] = dict(out['meta']['families'], **{ROWS: dict(w['meta'], tuned_as=FAMILY, source=str(args.widths))})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(f'wrote {args.out}: {ROWS} rows for {len(out[ROWS])} shapes')


def decisive(rounds, best, default):
    """amendment 7's rule: best replaces default only if its median is >= 0.5 % below and all its rounds are below all
    of default's."""
    if best == default:
        return False
    mb, md = statistics.median(rounds[best]), statistics.median(rounds[default])
    return mb <= 0.995 * md and max(rounds[best]) < min(rounds[default])


def sensitivity(args):
    raw = json.loads(Path(args.raw).read_text())['us_rounds'][FAMILY]
    chosen = json.loads(Path(args.widths).read_text())[FAMILY]
    tables = {'dm128': {}, 'dmfb': {}}
    changed = {'dm128': [], 'dmfb': []}
    for shape, by_bucket in raw.items():
        for b, rounds in by_bucket.items():
            med = {w: statistics.median(v) for w, v in rounds.items()}
            fastest = str(chosen[shape][b])   # the tuner's choice (its saved per-round values are rounded to 0.01 us)
            assert fastest in rounds, (shape, b, fastest)
            for name, default in (('dm128', '128'), ('dmfb', str(S.fallback_width(int(b))))):
                pick = fastest if decisive(rounds, fastest, default) else default
                tables[name].setdefault(shape, {})[b] = S.parse_key(pick)
                if pick != fastest:
                    changed[name].append(dict(shape=shape, bucket=int(b), fastest=fastest, rule=pick,
                                              fastest_us=med[fastest], rule_us=med[pick],
                                              rule_vs_fastest_pct=100 * (med[pick] / med[fastest] - 1),
                                              fastest_rounds=rounds[fastest], rule_rounds=rounds[pick]))
    res = dict(rule='a non-default width replaces the default only if its median is >= 0.5 % below the default\'s and '
                    'all its rounds are below all of the default\'s (amendment 7\'s scheduler-row rule)',
               defaults=dict(dm128='the single 128-wide build (n8k64_wB_t0)', dmfb='select.fallback_width(bucket)'),
               source=dict(raw=str(args.raw), widths=str(args.widths)), cells=sum(len(v) for v in raw.values()),
               tables={k: {ROWS: v} for k, v in tables.items()}, changed=changed,
               summary={k: dict(cells_changed=len(v),
                                median_pct=statistics.median([c['rule_vs_fastest_pct'] for c in v]) if v else None,
                                max_pct=max([c['rule_vs_fastest_pct'] for c in v]) if v else None) for k, v in changed.items()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')
    for k, v in res['summary'].items():
        print(k, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('compose')
    c.add_argument('--ko', required=True)
    c.add_argument('--widths', required=True)
    c.add_argument('--out', required=True)
    s = sub.add_parser('sensitivity')
    s.add_argument('--raw', required=True)
    s.add_argument('--widths', required=True)
    s.add_argument('--out', required=True)
    args = ap.parse_args()
    (compose if args.cmd == 'compose' else sensitivity)(args)


if __name__ == '__main__':
    main()

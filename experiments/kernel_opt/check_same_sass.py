#!/usr/bin/env python3
"""Kernel-opt amendment 11: the self-test gate's rebuilt libraries carry the measured device code (CPU only).

    python experiments/kernel_opt/check_same_sass.py --root DIR --like DIR --configs A,B,... --out JSON

`build.py --selftest` rebuilds a configuration's library before building and running its self-test driver, so the chain
runs it in a separate directory (--root) and leaves the registered builds (--like) untouched. A rebuild's library file
differs in its ELF bytes; its device code must not. For every configuration named, --root's manifest must carry the
patched and unpatched SASS sha256, the patcher's per-site census and the extra defines of --like's manifest, and (unless
--no-selftest; amendment 12's tuning directory has no self-tests) a passed self-test. A difference exits non-zero.
"""
import argparse
import json
import sys
from pathlib import Path

KEYS = ('sass_sha256', 'unpatched_sass_sha256', 'patcher_sha256')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--like', required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--no-selftest', action='store_true', help='do not require a passed self-test in --root')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    res, bad = dict(root=args.root, like=args.like, configs={}), []
    for name in args.configs.split(','):
        a = json.loads((Path(args.root) / name / 'manifest.json').read_text())
        b = json.loads((Path(args.like) / name / 'manifest.json').read_text())
        row = {k: a.get(k) == b.get(k) for k in KEYS}
        row['sites'] = a['patch'].get('sites') == b['patch'].get('sites')
        row['extra_defines'] = a.get('extra_defines') == b.get('extra_defines')
        row['selftest'] = (a.get('selftest') or {}).get('gate')
        res['configs'][name] = row
        bad += [f'{name}: {k}' for k, v in row.items() if v is False]
        if not args.no_selftest and row['selftest'] != 'PASS':
            bad.append(f'{name}: self-test gate {row["selftest"]}')
    res['failures'], res['passed'] = bad, not bad
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    for name, row in res['configs'].items():
        print(name, row)
    print('PASSED' if not bad else 'FAILED: ' + '; '.join(bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

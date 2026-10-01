#!/usr/bin/env python3
"""Kernel-opt gates G1 (inert) and G2 (census), from the build manifests (CPU only).

    python experiments/kernel_opt/check_sass.py --after-root /home/dev/n16k64_campaign/kernel_opt/build \
        --tmopt-root /home/dev/n16k64_campaign/kernel_opt/build_tmopt --new n8k64_wB_m64,n8k64_wB_m32,n8k64_wB_m16 --out JSON

Protocol: results/kernel_opt/PROTOCOL.md.
- G1: every configuration of sm120/mixfp4_sm120/configs.py that is not new must exist in --after-root with the patched
  and unpatched SASS hashes (sass_sha256, unpatched_sass_sha256) of its before build: sm120/build if it has one, else
  the first of --before-roots that has one (amendment 3: the kernel-opt build directories of earlier optimizations),
  else --tmopt-root (built by the same build.py from a git archive of tm-opt 74058af).
- G2: every new configuration's patched census equals its expected census, and it has no predicated OMMA.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
from mixfp4_sm120 import configs as CFG  # noqa: E402


def manifest(root, name):
    p = Path(root) / name / 'manifest.json'
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--after-root', required=True)
    ap.add_argument('--tmopt-root', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--before-roots', default='', help='comma-separated build directories searched after sm120/build')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    new = args.new.split(',')
    extra = [r for r in args.before_roots.split(',') if r]
    res = dict(after_root=args.after_root, tmopt_root=args.tmopt_root, before_roots=extra, g1={}, g2={})
    ok = True
    for name in CFG.CONFIGS:
        after = manifest(args.after_root, name)
        if name in new:
            cfg = CFG.get(name)
            pred = after['patch']['census_patched']['predicated'] if after else None
            if not cfg.patch:
                # kernel-opt #4: a new unpatched (stock) build -- E2M1 OMMAs only, none predicated
                got = after['patch']['census_patched']['formats'] if after else None
                want = {'E2M1xE2M1': after['patch']['census_patched']['total']} if after else None
            else:
                got = {int(k): v for k, v in after['patch']['sites'].items()} if after else None
                want = {int(k): v for k, v in (cfg.expected_census or {}).items()}
            passed = bool(after) and got == want and pred == 0
            res['g2'][name] = dict(census=got, expected=want, predicated=pred, passed=passed,
                                   sass_sha256=after and after['sass_sha256'],
                                   stages=after and after['compiled_description'].get('mainloop_stages'),
                                   tile_mnk=after and after['compiled_description'].get('tile_mnk'))
            ok &= passed
            continue
        before, source = manifest(REPO / 'sm120' / 'build', name), 'sm120/build'
        for root in extra:
            if before is None:
                before, source = manifest(root, name), root
        if before is None:
            before, source = manifest(args.tmopt_root, name), 'tm-opt 74058af sources'
        row = dict(before_source=source, present=bool(after and before))
        if after and before:
            for key in ('sass_sha256', 'unpatched_sass_sha256'):
                row[key] = dict(before=before[key], after=after[key], equal=before[key] == after[key])
            row['passed'] = all(row[k]['equal'] for k in ('sass_sha256', 'unpatched_sass_sha256'))
        else:
            row['passed'] = False
        res['g1'][name] = row
        ok &= row['passed']
    res['passed'] = bool(ok)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    for g in ('g1', 'g2'):
        for name, row in res[g].items():
            print(g.upper(), name, 'PASS' if row['passed'] else 'FAIL', row.get('before_source', row.get('census')))
    print('G1+G2', 'PASSED' if ok else 'FAILED')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

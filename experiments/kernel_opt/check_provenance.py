#!/usr/bin/env python3
"""Kernel-opt gate G0 (amendment 9): the registered files and builds are the ones measured.

    python experiments/kernel_opt/check_provenance.py --registration results/kernel_opt/registration_9.json --out JSON

- Every file the registration lists has its registered sha256 (sources, tables, scripts, maps).
- Every build the registration lists (per build directory and configuration): the library file's sha256 equals its
  manifest's library_sha256, and the manifest's library_sha256, sass_sha256, unpatched_sass_sha256 and extra_defines equal
  the registered ones.
- With 'sass_of' (build directory -> an earlier registration JSON and its key): each listed build's patched and
  unpatched SASS equal that registration's, i.e. the device code is the one gated there.
A failure exits non-zero.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--registration', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    reg = json.loads(args.registration.read_text())
    res, bad = dict(registration=str(args.registration), files={}, builds={}, sass_of={}), []
    for f, h in reg['files'].items():
        p = Path(f) if Path(f).is_absolute() else REPO / f
        got = sha(p) if p.exists() else None
        res['files'][f] = got == h
        if got != h:
            bad.append(f'{f}: sha256 {got} != registered {h}')
    for root, cfgs in reg['builds'].items():
        for name, want in cfgs.items():
            d = Path(root) / name
            man = json.loads((d / 'manifest.json').read_text())
            lib = sha(d / man['library'])
            ok = dict(library_file=lib == man['library_sha256'])
            for key in ('library_sha256', 'sass_sha256', 'unpatched_sass_sha256', 'extra_defines'):
                ok[key] = man.get(key) == want.get(key)
            res['builds'][f'{root}/{name}'] = ok
            bad += [f'{root}/{name}: {k}' for k, v in ok.items() if not v]
    for root, (path, key) in reg.get('sass_of', {}).items():
        ref = json.loads((REPO / path).read_text())[key]
        for name in reg['builds'][root]:
            man = json.loads((Path(root) / name / 'manifest.json').read_text())
            ok = all(man[k] == ref[name][k] for k in ('sass_sha256', 'unpatched_sass_sha256'))
            res['sass_of'][f'{root}/{name}'] = ok
            if not ok:
                bad.append(f'{root}/{name}: SASS differs from {path} [{key}]')
    res['failures'] = bad
    res['status'] = 'pass' if not bad else 'fail'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print(f"files {sum(res['files'].values())}/{len(res['files'])}, builds "
          f"{sum(all(v.values()) for v in res['builds'].values())}/{len(res['builds'])}, SASS as gated "
          f"{sum(res['sass_of'].values())}/{len(res['sass_of'])}")
    for b in bad:
        print('FAIL', b)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

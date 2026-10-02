#!/usr/bin/env python3
"""Kernel-opt amendment 11 (the 8x64 plan's P2): the patcher check on the weights-on-B builds, CPU only. It is amendment
6b's validation (results/kernel_opt/t0/6b/validate_6b.py) extended to the wB family.

    python experiments/kernel_opt/check_patcher_sites.py --tagged-roots DIR,DIR,... --t0-roots DIR,DIR --out JSON

1. Tagged binaries, where the strict parser is the ground truth. Every weights-on-B mixed build generated with the site-0
   tags (TAG0 unset) in --tagged-roots: its unpatched library (lib*.so.unpatched) and, if built, its unpatched self-test
   executable (selftest.unpatched). parse_ommas_cfg (reaching definitions, jump tables resolved) must give every OMMA the
   site that parse_ommas (strict: the nearest tagged writer) gives it. The self-test executables carry the BRX jump
   table; the libraries carry none.
2. t0 binaries. Every weights-on-B build generated with TAG0=0 in --t0-roots: parse_ommas_cfg must classify the OMMAs of
   its unpatched library and, if built, of its unpatched self-test executable with per-site counts equal to the
   configuration's census (configs.py expected_census; a no-dispatch build: site 0 only). The library's counts must
   also equal the patcher's record in the manifest (patch.sites).

The patcher is the repository's sm120/kernel/scripts/patch_mixed_nvfp4_gemm.py; its sha256 must equal every manifest's
patcher_sha256 (--t0-roots). Any mismatch exits non-zero.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
from mixfp4_sm120 import configs as CFG  # noqa: E402

PATCHER = REPO / 'sm120' / 'kernel' / 'scripts' / 'patch_mixed_nvfp4_gemm.py'
_spec = importlib.util.spec_from_file_location('patcher', PATCHER)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
CUOBJDUMP = Path(os.environ.get('CUDA_HOME', '/home/dev/.conda/envs/mixfp4-cuda131')) / 'bin' / 'cuobjdump'


def sass_of(binary):
    return subprocess.run([str(CUOBJDUMP), '--dump-sass', str(binary)], check=True, capture_output=True, text=True).stdout


def wb_builds(root):
    """[(config name, manifest, build dir)] of the weights-on-B mixed builds in root."""
    out = []
    for man in sorted(Path(root).glob('*/manifest.json')):
        m = json.loads(man.read_text())
        if m.get('kind') == 'mixed' and m.get('weight_operand') == 1:
            out.append((m['config'], m, man.parent))
    return out


def binaries(d, m):
    out = [('library', d / f"{m['library']}.unpatched")]
    if (d / 'selftest.unpatched').exists():
        out.append(('selftest', d / 'selftest.unpatched'))
    return out


def check_tagged(job):
    root, name, kind, binary = job
    sass = sass_of(binary)
    strict = sorted((a, s) for a, _, _, s, _ in P.parse_ommas(sass))
    cfg = sorted((a, s) for a, _, _, s, _ in P.parse_ommas_cfg(sass, P.jump_tables(CUOBJDUMP, binary)))
    return dict(root=str(root), config=name, binary=kind, ommas=len(strict), brx=' BRX ' in sass,
                sites=dict(sorted(collections.Counter(s for _, s in cfg).items())), passed=bool(strict) and strict == cfg)


def check_t0(job):
    root, name, kind, binary, want, recorded = job
    sass = sass_of(binary)
    got = dict(sorted(collections.Counter(s for _, _, _, s, _ in P.parse_ommas_cfg(sass, P.jump_tables(CUOBJDUMP, binary))).items()))
    ok = got == want and (recorded is None or got == recorded)
    return dict(root=str(root), config=name, binary=kind, brx=' BRX ' in sass, sites=got, census=want,
                manifest_sites=recorded, passed=ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tagged-roots', required=True, help='comma-separated build directories (part 1)')
    ap.add_argument('--t0-roots', required=True, help='comma-separated build directories (part 2)')
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    psha = hashlib.sha256(PATCHER.read_bytes()).hexdigest()
    tagged, t0, bad = [], [], []
    for root in args.tagged_roots.split(','):
        for name, m, d in wb_builds(root):
            if str((m.get('blob_gen') or {}).get('TAG0', 1)) != '0':
                tagged += [(root, name, kind, b) for kind, b in binaries(d, m)]
    for root in args.t0_roots.split(','):
        for name, m, d in wb_builds(root):
            if str((m.get('blob_gen') or {}).get('TAG0', 1)) != '0':
                continue
            if m.get('patcher_sha256') != psha:
                bad.append(f'{root}/{name}: manifest patcher_sha256 differs from {PATCHER}')
            census = CFG.get(name).expected_census
            want = {int(k): v for k, v in census.items()} if census else {0: m['patch']['census_unpatched']['total']}
            recorded = {int(k): v for k, v in m['patch']['sites'].items()} if m['patch'].get('sites') else None
            t0 += [(root, name, kind, b, want, recorded if kind == 'library' else None) for kind, b in binaries(d, m)]
    with concurrent.futures.ProcessPoolExecutor(args.jobs) as ex:
        res_tagged = list(ex.map(check_tagged, tagged))
        res_t0 = list(ex.map(check_t0, t0))
    for r in res_tagged + res_t0:
        if not r['passed']:
            bad.append(f"{r['root']}/{r['config']} {r['binary']}: {r}")
    res = dict(patcher=str(PATCHER.relative_to(REPO)), patcher_sha256=psha, cuobjdump=str(CUOBJDUMP),
               tagged=res_tagged, t0=res_t0,
               summary=dict(tagged_binaries=len(res_tagged), tagged_with_brx=sum(r['brx'] for r in res_tagged),
                            t0_binaries=len(res_t0), t0_with_brx=sum(r['brx'] for r in res_t0)),
               failures=bad, passed=not bad)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    for r in res_tagged:
        print('tagged', 'same' if r['passed'] else 'DIFF', Path(r['root']).name, r['config'], r['binary'], 'OMMAs',
              r['ommas'], 'BRX' if r['brx'] else 'no BRX', r['sites'])
    for r in res_t0:
        print('t0', 'OK ' if r['passed'] else 'BAD', Path(r['root']).name, r['config'], r['binary'], r['sites'],
              'BRX' if r['brx'] else 'no BRX')
    print(f"tagged: {len(res_tagged)} binaries ({res['summary']['tagged_with_brx']} with a BRX); t0: {len(res_t0)} "
          f"binaries; failures: {len(bad)}")
    for b in bad:
        print('FAIL', b)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

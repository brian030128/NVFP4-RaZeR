#!/usr/bin/env python3
"""Kernel-opt amendments 17 (U) and 18 (V), gate G2u: the uniform-branch dispatch's SASS conditions on the amendment's
build directory (CPU only).

    python experiments/kernel_opt/check_uniform_sass.py --root build_U --like16 build_7freq --like8 build_P3freq --out JSON
    python experiments/kernel_opt/check_uniform_sass.py --set V --root build_V --like-u build_U --like-a1 build_A1 \
        --like-stock build_7 --out JSON

Set V (amendment 18's deployment directory, 22 builds): amendment 17's ten carry build_U's patched and unpatched SASS;
A''s four g32 builds (the paper-table 'mixed256', for 'paper_256') carry build_A1's; stock_ko's four carry build_7's;
the 256x64 path's four g32 t0 builds have MIXFP4_UNIFORM_DISPATCH=1 and the uniform conditions below.

For every GEMM library in --root, from the patched library's SASS (cuobjdump) and its manifest:
- every build: no predicated OMMA, and its census equals its configuration's expected census (stock: E2M1 only);
- MIXFP4_UNIFORM_DISPATCH=1 builds: no BSSY, BSYNC or WARPSYNC between the first and the last OMMA (the k_tile loop and
  its peeled last iteration), and at least one REDUX there;
- builds with today's defines (#2's dispatch only, or none for stock_wB_e64): patched and unpatched SASS equal to the
  same configuration in --like16 (16x64) / --like8 (8x64, stock_wB_e64);
- the registered define set per configuration (part A: the uniform dispatch at 16x64 widths 64 and 128, the pipelined
  flags at 8x64 widths 64, '128x64' and 128; widths 16 and 32 unchanged).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
from mixfp4_sm120 import configs as CFG  # noqa: E402

CUOBJDUMP = '/home/dev/.conda/envs/mixfp4-cuda131/bin/cuobjdump'
FREQ = {'MIXFP4_DISPATCH_FREQ': 1}
EXPECT = {'n16k64_wA_n16_t0': FREQ, 'n16k64_wA_n32_t0': FREQ,
          'n16k64_wA_n64_t0': dict(FREQ, MIXFP4_UNIFORM_DISPATCH=1), 'n16k64_wA_e64_t0': dict(FREQ, MIXFP4_UNIFORM_DISPATCH=1),
          'n8k64_wB_m16_t0': FREQ, 'n8k64_wB_m32_t0': FREQ,
          'n8k64_wB_m64_t0': dict(FREQ, MIXFP4_PIPE_FLAGS=1), 'n8k64_wB_n64_t0': dict(FREQ, MIXFP4_PIPE_FLAGS=1),
          'n8k64_wB_t0': dict(FREQ, MIXFP4_PIPE_FLAGS=1), 'stock_wB_e64': {}}
G32_T0 = ['n16k64_wA_g32_n16_t0', 'n16k64_wA_g32_n32_t0', 'n16k64_wA_g32_n64_t0', 'n16k64_wA_g32_e64_t0']
A1 = ['n16k64_wA_g32', 'n16k64_wA_g32_n64', 'n16k64_wA_g32_n32', 'n16k64_wA_g32_n16']
STOCK_A = ['stock_wA_n16', 'stock_wA_n32', 'stock_wA_n64', 'stock_wA_e64']
EXPECT_V = dict(EXPECT, **{c: {'MIXFP4_UNIFORM_DISPATCH': 1} for c in G32_T0}, **{c: {} for c in A1 + STOCK_A})
KINDS = ('BSSY', 'BSYNC', 'WARPSYNC', 'REDUX')


def sass_stats(so):
    sass = subprocess.run([CUOBJDUMP, '-sass', str(so)], check=True, capture_output=True, text=True).stdout
    funcs = [f for f in sass.split('Function :')[1:] if 'device_kernel' in f.splitlines()[0]]
    assert len(funcs) == 1, (so, len(funcs))
    ins = [(int(a, 16), (p or '').strip(), op) for a, p, op in
           re.findall(r'/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)', funcs[0])]
    om = [a for a, _, op in ins if op.startswith('OMMA')]
    lo, hi = min(om), max(om)
    return dict(instructions=len(ins), omma=len(om),
                predicated_omma=sum(1 for _, p, op in ins if op.startswith('OMMA') and p),
                total={k: sum(op.startswith(k) for _, _, op in ins) for k in KINDS},
                in_omma_span={k: sum(op.startswith(k) for a, _, op in ins if lo <= a <= hi) for k in KINDS})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', choices=('U', 'V'), default='U')
    ap.add_argument('--root', required=True)
    ap.add_argument('--like16', help='set U: the 16x64 builds with today\'s defines')
    ap.add_argument('--like8', help='set U: the 8x64 builds and stock_wB_e64')
    ap.add_argument('--like-u', help="set V: amendment 17's build directory")
    ap.add_argument('--like-a1', help="set V: A''s build directory")
    ap.add_argument('--like-stock', help="set V: stock_ko's build directory")
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = Path(args.root)
    expect = EXPECT if args.set == 'U' else EXPECT_V
    present = sorted(d.name for d in root.iterdir() if (d / 'manifest.json').exists())
    res = dict(set=args.set, root=str(root), like16=args.like16, like8=args.like8, like_u=args.like_u,
               like_a1=args.like_a1, like_stock=args.like_stock, builds={}, passed=True)
    if present != sorted(expect):
        res['passed'] = False
        res['error'] = f'builds present {present}, registered {sorted(expect)}'
    for name in present:
        man = json.loads((root / name / 'manifest.json').read_text())
        cfg = CFG.get(name)
        so = root / name / Path(man['library']).name if 'library' in man else next(
            p for p in (root / name).glob('lib*.so') if not p.name.endswith('.unpatched'))
        st = sass_stats(so)
        defines = man.get('extra_defines') or {}
        row = dict(defines=defines, sass_sha256=man['sass_sha256'], **st, failures=[])
        if defines != expect.get(name):
            row['failures'].append(f'defines {defines}, registered {expect.get(name)}')
        if st['predicated_omma']:
            row['failures'].append(f"{st['predicated_omma']} predicated OMMA")
        if cfg.patch:
            got = {int(k): v for k, v in man['patch']['sites'].items()}
            want = {int(k): v for k, v in (cfg.expected_census or {}).items()}
        else:
            got = man['patch']['census_patched']['formats']
            want = {'E2M1xE2M1': man['patch']['census_patched']['total']}
        row['census'] = dict(got=got, expected=want)
        if got != want:
            row['failures'].append('census differs')
        if defines.get('MIXFP4_UNIFORM_DISPATCH'):
            span = st['in_omma_span']
            if span['BSSY'] or span['BSYNC'] or span['WARPSYNC']:
                row['failures'].append(f'reconvergence code between the first and the last OMMA: {span}')
            if not span['REDUX']:
                row['failures'].append('no REDUX between the first and the last OMMA')
        like = None
        if args.set == 'V':
            like = (args.like_u if name in EXPECT else args.like_a1 if name in A1 else args.like_stock if name in STOCK_A
                    else None)
        elif not defines.get('MIXFP4_UNIFORM_DISPATCH') and defines in (FREQ, {}):
            like = args.like16 if name.startswith('n16k64') else args.like8
        if like is not None:
            like = Path(like)
            ref = json.loads((like / name / 'manifest.json').read_text())
            row['like'] = dict(root=str(like), **{k: dict(ref=ref[k], got=man[k], equal=ref[k] == man[k])
                                                  for k in ('sass_sha256', 'unpatched_sass_sha256')})
            if not all(row['like'][k]['equal'] for k in ('sass_sha256', 'unpatched_sass_sha256')):
                row['failures'].append(f'SASS differs from {like}')
        row['passed'] = not row['failures']
        res['builds'][name] = row
        res['passed'] &= row['passed']
        print(f"{name:20s} {'PASS' if row['passed'] else 'FAIL'} defines={defines} span={st['in_omma_span']} "
              f"predicated={st['predicated_omma']} {row['failures']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print('G2u', 'PASSED' if res['passed'] else 'FAILED')
    sys.exit(0 if res['passed'] else 1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Exploratory (disclosed): the CPU SASS check of the uniform-branch dispatch. For each build, the GEMM kernel's SASS
(cuobjdump of the patched library): instructions, BSSY/BSYNC/WARPSYNC/REDUX in total and between the first and the last
OMMA (the k_tile loop and its peeled last iteration), predicated OMMAs, and the patcher's census from the manifest.

    python results/kernel_opt/U/exploration/sass_U.py > results/kernel_opt/U/exploration/sass_U.json
"""
import json
import re
import subprocess
import sys

KO = '/home/dev/n16k64_campaign/kernel_opt'
CB = '/home/dev/.conda/envs/mixfp4-cuda131/bin/cuobjdump'
BUILDS = [(r, c) for c in ('n16k64_wA_n16_t0', 'n16k64_wA_n32_t0', 'n16k64_wA_n64_t0', 'n16k64_wA_e64_t0')
          for r in ('build_7freq', 'build_U1')] + \
         [(r, c) for c in ('n8k64_wB_m16_t0', 'n8k64_wB_m32_t0', 'n8k64_wB_m64_t0', 'n8k64_wB_n64_t0', 'n8k64_wB_t0')
          for r in ('build_P3freq', 'build_U1', 'build_U1p', 'build_U0p')]


def check(root, cfg):
    so = f'{KO}/{root}/{cfg}/libmixfp4_sm120_{cfg}.so'
    sass = subprocess.run([CB, '-sass', so], check=True, capture_output=True, text=True).stdout
    body = sass.split('Function :')[-1]
    assert 'device_kernel' in body.splitlines()[0]
    ins = [(int(a, 16), (p or '').strip(), op) for a, p, op in
           re.findall(r'/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)', body)]
    om = [a for a, _, op in ins if op.startswith('OMMA')]
    lo, hi = min(om), max(om)
    man = json.load(open(f'{KO}/{root}/{cfg}/manifest.json'))
    kinds = ('BSSY', 'BSYNC', 'WARPSYNC', 'REDUX')
    return dict(root=root, config=cfg, defines=man.get('extra_defines'), sass_sha256=man['sass_sha256'],
                census=man['patch'].get('census'), instructions=len(ins), omma=len(om),
                predicated_omma=sum(1 for _, p, op in ins if op.startswith('OMMA') and p),
                total={k: sum(op.startswith(k) for _, _, op in ins) for k in kinds},
                in_omma_span={k: sum(op.startswith(k) for a, _, op in ins if lo <= a <= hi) for k in kinds})


rows = [check(r, c) for r, c in BUILDS]
json.dump(rows, sys.stdout, indent=1)
print()
for r in rows:
    print(f"{r['root']:13s} {r['config']:18s} instr {r['instructions']:5d} predicated OMMA {r['predicated_omma']} "
          f"total {r['total']} in OMMA span {r['in_omma_span']}", file=sys.stderr)

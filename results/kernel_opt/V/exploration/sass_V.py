#!/usr/bin/env python3
"""Exploratory (disclosed): the CPU SASS check of the 256x64 (A') exploration builds. For each build: instructions,
BSSY/BSYNC/WARPSYNC/REDUX in total and between the first and the last OMMA, predicated OMMAs, the patcher's census,
mainloop stages and shared memory, from the patched library (cuobjdump) and its manifest.

    python results/kernel_opt/V/exploration/sass_V.py > results/kernel_opt/V/exploration/sass_V.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'experiments' / 'kernel_opt'))
from check_uniform_sass import sass_stats  # noqa: E402

KO = Path('/home/dev/n16k64_campaign/kernel_opt')
ROOTS = ['build_A1', 'build_VD0', 'build_VF', 'build_VU', 'build_VFU']
CFGS = ['n16k64_wA_g32', 'n16k64_wA_g32_n16', 'n16k64_wA_g32_n32', 'n16k64_wA_g32_n64', 'n16k64_wA_g32_e64',
        'n16k64_wA_g32_n16_t0', 'n16k64_wA_g32_n32_t0', 'n16k64_wA_g32_n64_t0', 'n16k64_wA_g32_e64_t0']
rows = []
for r in ROOTS:
    for c in CFGS:
        d = KO / r / c
        if not (d / 'manifest.json').exists():
            continue
        m = json.loads((d / 'manifest.json').read_text())
        st = sass_stats(d / m['library'])
        desc = m['compiled_description']
        rows.append(dict(root=r, config=c, defines=m.get('extra_defines'), sass_sha256=m['sass_sha256'],
                         census=m['patch'].get('sites'), stages=desc['mainloop_stages'], smem=desc['shared_storage_bytes'],
                         **st))
        print(f"{r:10s} {c:22s} defs={m.get('extra_defines') or {}} instr={st['instructions']:5d} stages={desc['mainloop_stages']} "
              f"smem={desc['shared_storage_bytes']} pred={st['predicated_omma']} span={st['in_omma_span']} total={st['total']}",
              file=sys.stderr)
json.dump(rows, sys.stdout, indent=1)
print()

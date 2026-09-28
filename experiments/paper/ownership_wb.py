#!/usr/bin/env python3
"""The ownership check of an E2M1-only artifact (NVFP4, FourOverSix) on the weights-on-B stock kernel (stock_wB).

    PAPER_PYTHON experiments/paper/ownership_wb.py --artifact ART --out ART/ownership_stock_wB.json

export_map_artifact.py --ownership checks these artifacts on stock_wA, their accuracy kernel. The same artifact also
runs on stock_wB, the same-placement latency reference of the 8x64 maps (step 05): an artifact's scale layout does
not depend on the placement (sf_buffer_size(rows, k) for either operand), and NativeLinear places the scales for the
kernel it is given. This runs sm120/eval/ownership_check.check_module on stock_wB with an all-E2M1 map: every weight
element's hardware-decoded value must equal the stored code times its scale, and no element may run as E0M3.
"""
import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
from mixfp4_sm120 import artifact as A  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location('ownership_check', REPO / 'sm120' / 'eval' / 'ownership_check.py')
    oc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oc)
    kern = Kernel.load('stock_wB')
    assert kern.weight_operand == 1, kern.cfg.name
    t0 = time.time()
    meta, weights = A.load(args.artifact)
    assert meta['type_block'] is None, 'E2M1-only artifacts only (maps are checked on their own kernel by the export)'
    rows = {}
    for name, pw in weights.items():
        n, k = pw.shape
        rows[name] = oc.check_module(kern, dataclasses.replace(pw, type_block=(16, 64)),
                                     torch.zeros(-(-n // 16), k // 64, dtype=torch.bool))
    bad = sorted(n for n, r in rows.items() if not r['exact'] or r['format_mismatches'] or r['e0m3_elements_observed'])
    summary = dict(modules=len(rows), all_exact=all(r['exact'] for r in rows.values()),
                   format_mismatches=sum(r['format_mismatches'] for r in rows.values()),
                   informative_elements=sum(r['informative_elements'] for r in rows.values()),
                   e0m3_elements_observed=sum(r['e0m3_elements_observed'] for r in rows.values()), failing_modules=bad,
                   kernel=kern.cfg.name, kernel_sha256=kern.sha256, artifact_weights_sha256=meta['weights_sha256'],
                   check='sm120/eval/ownership_check.check_module with an all-E2M1 map, weights on operand B',
                   seconds=round(time.time() - t0, 1), returncode=1 if bad else 0)
    args.out.write_text(json.dumps(dict(summary=summary, modules=rows), indent=1) + '\n')
    print('OWNERSHIP stock_wB', json.dumps(summary), flush=True)
    sys.exit(summary['returncode'])


if __name__ == '__main__':
    main()

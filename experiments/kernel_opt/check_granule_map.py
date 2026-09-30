#!/usr/bin/env python3
"""Kernel-opt amendment 3 (A'), gate G-span: the executed granule map of the 4-arm builds, measured.

    python experiments/kernel_opt/check_granule_map.py --a1-root DIR --out JSON

For every build (n16k64_wA from sm120/build for contrast, and the four n16k64_wA_g32 widths from --a1-root), a weight
of n rows x 128 columns is tagged E0M3 on one 16-row block b only (both k_blocks), and the decode probe of
sm120/tests/test_gemm.py (an exact identity activation operand) shows which 16-row blocks the tensor core executed as
E0M3. Rule:
- n16k64_wA executes exactly block b for every b;
- a g32 build executes blocks b and b + 4 when b is a warp's first m-atom (b mod 8 < 4) and no block when it is the
  second (b mod 8 >= 4, whose own tag is never read), i.e. its granules are {8p + w, 8p + w + 4}, w < 4, for every
  128-row panel p; so every granule lies inside one 128-row panel and inside one 256-row tile, and every 256x64 tile
  is a union of granules (n = 512: two 256-row tiles);
- on a 48-row weight (Qwen3.8's partial panel) every real block executes its own tag and nothing else.
"""
import argparse
import datetime
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
sys.path.insert(0, str(REPO / 'sm120' / 'tests'))
import common as B  # noqa: E402
import test_gemm as TG  # noqa: E402  (the decode probe's operands)
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402

G32 = ('n16k64_wA_g32', 'n16k64_wA_g32_n64', 'n16k64_wA_g32_n32', 'n16k64_wA_g32_n16')
K = 128


def executed(kern, n, block):
    m16 = torch.zeros(n // 16, K // 64, dtype=torch.bool)
    m16[block] = True
    wnib, wsb = TG.random_weight_operand(n, K, m16, (16, 64), seed=block)
    xnib, xsb = TG.identity_operand(K, K)
    d = TG.linear_gemm(kern, N.pack_nibbles(wnib), TG.place(wsb, K), 1.0, N.pack_nibbles(xnib), TG.place(xsb, K), None,
                       n, K, K)
    got = d.double().t()
    e0m3 = (got != N.decode_exact(wnib, wsb & 0x7F, 1.0)).reshape(n // 16, 16 * K).any(1)
    return sorted(int(b) for b in torch.nonzero(e0m3).flatten())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a1-root', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    builds = {'n16k64_wA': Kernel.load('n16k64_wA')}
    assert Path(builds['n16k64_wA'].path).parent.parent == REPO / 'sm120' / 'build'
    builds.update({nm: Kernel.load(nm, build_root=args.a1_root) for nm in G32})
    res = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), gpu=B.gpu_info(),
               builds={nm: dict(sha256=k.sha256, root=str(Path(k.path).parent.parent), type_block=list(k.type_block),
                                map_tile_rows=k.cfg.map_tile_rows) for nm, k in builds.items()}, maps={}, checks={})
    ok = True
    for nm, kern in builds.items():
        rec = res['maps'][nm] = {}
        for n in (512, 48):
            got = {b: executed(kern, n, b) for b in range(n // 16)}
            if kern.cfg.map_tile_rows is None or n == 48:
                want = {b: [b] for b in got}
            else:
                want = {b: ([b, b + 4] if b % 8 < 4 else []) for b in got}
            rec[str(n)] = {str(b): v for b, v in got.items()}
            passed = got == want
            if passed and kern.cfg.map_tile_rows and n == 512:
                granules = [set(v) for v in got.values() if v]
                passed = (sorted(b for gr in granules for b in gr) == list(range(n // 16))
                          and all(len({b // 8 for b in gr}) == 1 and len({b // 16 for b in gr}) == 1 for gr in granules))
                res['granules_' + nm] = [sorted(gr) for gr in granules]
            res['checks'][f'{nm} n={n}'] = passed
            ok &= passed
            print(nm, n, 'PASS' if passed else 'FAIL', flush=True)
    res['passed'] = bool(ok)
    res['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    B.write(args.out, res)
    print('G-span', 'PASSED' if ok else 'FAILED')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Kernel-opt amendment 17 (U), measurement M1: amendment 17's builds (part A) and its re-tuned 16x64 width cells
(part B) against today's adopted paths, both families, and the gap to stock.

    python experiments/kernel_opt/bench_U_isolated.py --model llama8b --u-root DIR --b7 DIR --b7freq DIR --p3freq DIR \
        --c3k DIR --p5 DIR --table-b JSON --artifact fo6=ART --artifact tc_16x64=ART --artifact tc_8x64=ART --out JSON

The deviation-2 method (cold weights by rotation + a 512 MiB flush, activations quantized after the flush, CUDA-event
time of isolated launches, 3 rotated rounds x 30) through bench_ab_isolated.run(), as amendments 10-13. Every set but
the part-B ones reads the tracked adopted table (sm120/configs/<gpu>.ko.json). Configurations (activations FourOverSix):
  stock_ko          KernelSet('stock_ko') from --b7: the target (stock, weights on A)
  c16               KernelSet('nodisp_ko') from --c3k on the FourOverSix artifact: the 16x64 no-dispatch ceiling
  A16_{v}           KernelSet('mixed_ko') from --b7freq: today's adopted 16x64 path (t0, #2's dispatch)
  U16_{v}           KernelSet('mixed_ko') from --u-root: part A (widths 64 and 128 with MIXFP4_UNIFORM_DISPATCH=1)
  UB16_{v}          the same on --table-b: part B (the re-tuned width cells and their scheduler rows)
  c8                KernelSet('nodisp_wB_ko') from --p5 on the FourOverSix artifact: the 8x64 no-dispatch ceiling
  A8_{v}            KernelSet('mixed_wB_ko') from --p3freq: today's adopted 8x64 path (t0, #2's dispatch)
  U8_{v}            KernelSet('mixed_wB_ko') from --u-root: part A (widths 64, '128x64' and 128 with MIXFP4_PIPE_FLAGS=1)
with v = typical (the lower-median module per projection) and worst (the densest). Added checks, bitwise on the timed
operands: U16 and UB16 equal A16, U8 equals A8, on the same tags.
"""
import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments' / 'kernel_opt'))
sys.path.insert(0, str(REPO / 'experiments' / 'paper'))
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import bench_ab_isolated as AB  # noqa: E402
import bench_gemm_isolated as G  # noqa: E402
import common as B  # noqa: E402
from mixfp4_sm120 import select as S  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

Q = 'four_over_six_rows'
CONFIGS = OrderedDict(stock_ko=('fo6', 'first', 'stock_ko', Q), c16=('fo6', 'first', 'c16', Q),
                      c8=('fo6', 'first', 'c8', Q))
for v in ('typical', 'worst'):
    CONFIGS[f'A16_{v}'] = ('tc_16x64', v, 'A16', Q)
    CONFIGS[f'U16_{v}'] = ('tc_16x64', v, 'U16', Q)
    CONFIGS[f'UB16_{v}'] = ('tc_16x64', v, 'UB16', Q)
    CONFIGS[f'A8_{v}'] = ('tc_8x64', v, 'A8', Q)
    CONFIGS[f'U8_{v}'] = ('tc_8x64', v, 'U8', Q)
PAIRS = [(f'A16_{v}', f'U16_{v}') for v in ('typical', 'worst')] + \
        [(f'A16_{v}', f'UB16_{v}') for v in ('typical', 'worst')] + [(f'A8_{v}', f'U8_{v}') for v in ('typical', 'worst')]
FREQ = {'MIXFP4_DISPATCH_FREQ': 1}
# the registered defines per family and width: amendment 17's part A
DEFINES = {('A16', w): FREQ for w in (16, 32, 64, 128)}
DEFINES.update({('U16', w): FREQ for w in (16, 32)})
DEFINES.update({('U16', w): dict(FREQ, MIXFP4_UNIFORM_DISPATCH=1) for w in (64, 128)})
DEFINES.update({('A8', w): FREQ for w in (16, 32, 64, 128, '128x64')})
DEFINES.update({('U8', w): FREQ for w in (16, 32)})
DEFINES.update({('U8', w): dict(FREQ, MIXFP4_PIPE_FLAGS=1) for w in (64, 128, '128x64')})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--u-root', required=True, help="amendment 17's build directory (build_U)")
    ap.add_argument('--b7', required=True, help="amendment 7's build directory (stock_ko)")
    ap.add_argument('--b7freq', required=True, help="today's adopted 16x64 path's build directory")
    ap.add_argument('--p3freq', required=True, help="today's adopted 8x64 path's build directory")
    ap.add_argument('--c3k', required=True, help="the 16x64 ceiling's build directory (nodisp_ko)")
    ap.add_argument('--p5', required=True, help="the 8x64 ceiling's build directory (nodisp_wB_ko)")
    ap.add_argument('--table-b', required=True, help="part B's table (U_tables.py compose)")
    ap.add_argument('--tokens', default=','.join(map(str, AB.TOKENS)))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--configs', default=None)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.after_root = args.u_root
    B.require_idle()
    ko = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    kernels = dict(stock_ko=KernelSet('stock_ko', build_root=args.b7, table=ko),
                   c16=KernelSet('nodisp_ko', build_root=args.c3k, table=ko),
                   c8=KernelSet('nodisp_wB_ko', build_root=args.p5, table=ko),
                   A16=KernelSet('mixed_ko', build_root=args.b7freq, table=ko),
                   U16=KernelSet('mixed_ko', build_root=args.u_root, table=ko),
                   UB16=KernelSet('mixed_ko', build_root=args.u_root, table=args.table_b),
                   A8=KernelSet('mixed_wB_ko', build_root=args.p3freq, table=ko),
                   U8=KernelSet('mixed_wB_ko', build_root=args.u_root, table=ko))
    for (name, w), want in DEFINES.items():
        k = kernels[name].kernels[w]
        assert (k.manifest.get('extra_defines') or {}) == want, (name, w, k.manifest.get('extra_defines'))
        assert str(k.manifest['blob_gen'].get('TAG0')) == '0', (name, w)
    for name in ('A16', 'U16', 'UB16', 'A8', 'U8'):
        assert set(kernels[name].kernels) == {w for (nm, w) in DEFINES if nm == name.replace('UB', 'U')}, name
    assert kernels['U16'].sha256 == kernels['UB16'].sha256             # the same libraries: only the table differs
    for name, root in (('stock_ko', args.b7), ('c16', args.c3k), ('c8', args.p5), ('A16', args.b7freq),
                       ('U16', args.u_root), ('UB16', args.u_root), ('A8', args.p3freq), ('U8', args.u_root)):
        assert all(Path(k.path).parent.parent == Path(root) for k in kernels[name].kernels.values()), name
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = 'kernel-opt amendment 17 (U: uniform-branch dispatch, part A; re-tuned 16x64 width cells, part B), M1'
    res['roots'] = dict(u=args.u_root, b7=args.b7, b7freq=args.b7freq, p3freq=args.p3freq, c3k=args.c3k, p5=args.p5)
    res['tables'] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (ko, Path(args.table_b))}
    B.write(args.out, res)


if __name__ == '__main__':
    main()

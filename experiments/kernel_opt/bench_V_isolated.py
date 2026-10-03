#!/usr/bin/env python3
"""Kernel-opt amendment 18 (V), measurement M1: the 256x64 path brought to the 16x64 state against today's 256x64 path
and stock.

    python experiments/kernel_opt/bench_V_isolated.py --model llama8b --v-root DIR --a1 DIR --b7 DIR --c3k DIR \
        --table-v0 JSON --table-v JSON --artifact fo6=ART --artifact tc_256x64=ART --out JSON

The deviation-2 method (cold weights by rotation + a 512 MiB flush, activations quantized after the flush, CUDA-event
time of isolated launches, 3 rotated rounds x 30) through bench_ab_isolated.run(), as amendments 10-17. Configurations
(activations FourOverSix; the mixed ones on the TC 256x64 artifact):
  stock_ko     KernelSet('stock_ko') from --b7 on the adopted table: the target
  ceil         KernelSet('nodisp256_ko') from --c3k on --table-v: the candidate's tiles, widths and scheduler rows with
               the dispatch compiled out (FourOverSix weights, E2M1 only)
  A_{v}        KernelSet('mixed256') from --a1 on the paper table: today's 256x64 path (A')
  B_{v}        KernelSet('mixed256_ko') from --v-root on --table-v0: the new builds at today's widths (the build effect)
  C_{v}        KernelSet('mixed256_ko') from --v-root on --table-v: the candidate (the re-tuned widths and scheduler rows)
with v = typical (the lower-median module per projection) and worst (the densest). Added checks, bitwise on the timed
operands: B and C equal A on the same tags.
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
CONFIGS = OrderedDict(stock_ko=('fo6', 'first', 'stock_ko', Q), ceil=('fo6', 'first', 'ceil', Q))
for v in ('typical', 'worst'):
    CONFIGS[f'A_{v}'] = ('tc_256x64', v, 'A', Q)
    CONFIGS[f'B_{v}'] = ('tc_256x64', v, 'B', Q)
    CONFIGS[f'C_{v}'] = ('tc_256x64', v, 'C', Q)
PAIRS = [(f'A_{v}', f'{x}_{v}') for v in ('typical', 'worst') for x in ('B', 'C')]
UNIFORM = {'MIXFP4_UNIFORM_DISPATCH': 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--v-root', required=True, help="amendment 18's build directory (build_V)")
    ap.add_argument('--a1', required=True, help="A''s build directory (today's mixed256)")
    ap.add_argument('--b7', required=True, help="amendment 7's build directory (stock_ko)")
    ap.add_argument('--c3k', required=True, help="the ceiling's build directory (the nodisp t0 builds)")
    ap.add_argument('--table-v0', required=True, help="V_tables.py today: today's widths as 'mixed256' rows")
    ap.add_argument('--table-v', required=True, help='V_tables.py finalize: the candidate table')
    ap.add_argument('--tokens', default=','.join(map(str, AB.TOKENS)))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--configs', default=None)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.after_root = args.v_root
    B.require_idle()
    ko = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    paper = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    kernels = dict(stock_ko=KernelSet('stock_ko', build_root=args.b7, table=ko),
                   ceil=KernelSet('nodisp256_ko', build_root=args.c3k, table=args.table_v),
                   A=KernelSet('mixed256', build_root=args.a1, table=paper),
                   B=KernelSet('mixed256_ko', build_root=args.v_root, table=args.table_v0),
                   C=KernelSet('mixed256_ko', build_root=args.v_root, table=args.table_v))
    for name in ('B', 'C'):
        for k in kernels[name].kernels.values():
            assert (k.manifest.get('extra_defines') or {}) == UNIFORM, (name, k.cfg.name, k.manifest.get('extra_defines'))
            assert str(k.manifest['blob_gen'].get('TAG0')) == '0', (name, k.cfg.name)
    for k in kernels['A'].kernels.values():
        assert not k.manifest.get('extra_defines'), k.cfg.name
    assert kernels['B'].sha256 == kernels['C'].sha256                  # the same libraries: only the table differs
    assert not kernels['A'].schedules and not kernels['B'].schedules and kernels['C'].schedules
    for name, root in (('stock_ko', args.b7), ('ceil', args.c3k), ('A', args.a1), ('B', args.v_root), ('C', args.v_root)):
        assert all(Path(k.path).parent.parent == Path(root) for k in kernels[name].kernels.values()), name
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = 'kernel-opt amendment 18 (V: the 256x64 path brought to the 16x64 state), M1'
    res['roots'] = dict(v=args.v_root, a1=args.a1, b7=args.b7, c3k=args.c3k)
    res['tables'] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                     for p in (paper, ko, Path(args.table_v0), Path(args.table_v))}
    B.write(args.out, res)


if __name__ == '__main__':
    main()

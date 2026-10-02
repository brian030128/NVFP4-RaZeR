#!/usr/bin/env python3
"""Kernel-opt amendment 12b (the 8x64 plan's P3b, option C), measurement M1: a fresh confirmation of the reduced 8x64
table (1b's rows with the 11 selected P3 widths, no scheduler rows) against P2's adopted path, and the gap to stock.

    python experiments/kernel_opt/bench_w8p3b_isolated.py --model llama8b --p3freq-root DIR --b7 DIR --table P3B.json \
        --artifact fo6=ART --artifact tc_8x64=ART --out JSON

The deviation-2 method, exactly as bench_w8_isolated.py (amendment 10), through bench_ab_isolated.run(). Both 8x64
configurations run the same libraries (build_P3freq's t0 builds with #2's dispatch), so only the table differs.
Configurations (activation quantizer FourOverSix):
  stock_ko          KernelSet('stock_ko') from --b7 on the tracked adopted table: the target (stock, weights on A)
  stock_wB          stock_wB (sm120/build), the paper's same-placement reference
  stock_wB_ko       KernelSet('stock_wB_ko') on --table: stock_wB_e64 with its P3 scheduler rows (adopted)
  wBp2_{v}          KernelSet('mixed_wB_t0') on the paper table: 1b's widths, no scheduler rows (P2's adopted path)
  wBp3b_{v}         KernelSet('mixed_wB_ko') on --table: the reduced table, no scheduler rows (the candidate)
Added checks, bitwise on the timed operands: wBp3b equals wBp2 on the same tags, and stock_wB_ko equals stock_wB.
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
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

Q = 'four_over_six_rows'
CONFIGS = OrderedDict(stock_ko=('fo6', 'first', 'stock_ko', Q), stock_wB=('fo6', 'first', 'stock_wB', Q),
                      stock_wB_ko=('fo6', 'first', 'stock_wB_ko', Q))
for v in ('typical', 'worst'):
    CONFIGS[f'wBp2_{v}'] = ('tc_8x64', v, 'p2', Q)
    CONFIGS[f'wBp3b_{v}'] = ('tc_8x64', v, 'p3b', Q)
PAIRS = [('wBp2_typical', 'wBp3b_typical'), ('wBp2_worst', 'wBp3b_worst'), ('stock_wB', 'stock_wB_ko')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--p3freq-root', required=True, help="the adopted 8x64 path's build directory (and stock_wB_e64)")
    ap.add_argument('--b7', required=True, help="amendment 7's build directory (stock_ko)")
    ap.add_argument('--table', required=True, help='the reduced table (p3b_tables.py compose: table_p3b)')
    ap.add_argument('--tokens', default=','.join(map(str, AB.TOKENS)))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--configs', default=None)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.after_root = args.p3freq_root
    B.require_idle()
    paper_table = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    ko_table = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    root = Path(args.p3freq_root)
    kernels = dict(stock_ko=KernelSet('stock_ko', build_root=args.b7, table=ko_table),
                   stock_wB=Kernel.load('stock_wB'),
                   stock_wB_ko=KernelSet('stock_wB_ko', build_root=root, table=args.table),
                   p2=KernelSet('mixed_wB_t0', build_root=root, table=paper_table),
                   p3b=KernelSet('mixed_wB_ko', build_root=root, table=args.table))
    assert Path(kernels['stock_wB'].path).parent.parent == REPO / 'sm120' / 'build'
    assert all(Path(k.path).parent.parent == Path(args.b7) for k in kernels['stock_ko'].kernels.values())
    for name in ('p2', 'p3b', 'stock_wB_ko'):
        assert all(Path(k.path).parent.parent == root for k in kernels[name].kernels.values()), name
    for name in ('p2', 'p3b'):
        ks = kernels[name]
        assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1 for k in ks.kernels.values()), name
        assert all(str(k.manifest['blob_gen'].get('TAG0')) == '0' for k in ks.kernels.values()), name
    assert kernels['p2'].sha256 == kernels['p3b'].sha256                  # the same libraries: only the table differs
    assert not kernels['p2'].schedules and not kernels['p3b'].schedules and kernels['stock_wB_ko'].schedules
    cells = sum(str(w) != str(kernels['p2'].table[s][b]) for s, row in kernels['p3b'].table.items() for b, w in row.items())
    assert cells == 11, cells
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = "kernel-opt amendment 12b (the 8x64 plan's P3b: the reduced table's confirmation), M1"
    res['roots'] = dict(p3freq=str(root), b7=str(args.b7), paper=str(REPO / 'sm120' / 'build'))
    res['tables'] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (paper_table, ko_table, Path(args.table))}
    B.write(args.out, res)


if __name__ == '__main__':
    main()

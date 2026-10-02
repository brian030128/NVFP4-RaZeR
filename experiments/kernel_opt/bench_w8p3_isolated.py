#!/usr/bin/env python3
"""Kernel-opt amendment 12 (the 8x64 plan's P3, with P4), measurement M1: the adopted 8x64 path on its re-tuned table
against P2's rows, stock_wB tuned alike, and the decisive-margin sensitivity of the widths.

    python experiments/kernel_opt/bench_w8p3_isolated.py --model llama8b --p3freq-root DIR --b7 DIR --table P3.json \
        --sensitivity SENS.json --artifact fo6=ART --artifact tc_8x64=ART --out JSON

The deviation-2 method, exactly as bench_w8_isolated.py (amendment 10), through bench_ab_isolated.run(). Every 8x64
configuration runs the same libraries: the t0 builds with #2's dispatch in --p3freq-root, whose SASS is build_P2freq's
(the 8x64 path adopted after amendment 11). Only the table differs. Configurations (activation quantizer FourOverSix):
  stock_ko          KernelSet('stock_ko') from --b7 on the tracked adopted table: the target (stock, weights on A)
  stock_wB          stock_wB (sm120/build), the paper's same-placement reference
  stock_wB_ko       KernelSet('stock_wB_ko'): stock_wB_e64 with the P3 table's scheduler rows (stock tuned alike)
  wBp2_{v}          KernelSet('mixed_wB_t0') on the paper table: 1b's widths, no scheduler rows (P2's adopted path)
  wBko_{v}          KernelSet('mixed_wB_ko') on the P3 table: its widths and scheduler rows (the candidate)
  wBw_typical       the P3 widths without scheduler rows (separates the widths from the rows)
  wBdm128_typical   the decisive-margin widths, default the 128-wide build, no scheduler rows (sensitivity)
  wBdmfb_typical    the decisive-margin widths, default the fallback width, no scheduler rows (sensitivity)
Added checks, bitwise on the timed operands: every 8x64 configuration equals wBp2 on the same tags, and stock_wB_ko
equals stock_wB.
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
    CONFIGS[f'wBko_{v}'] = ('tc_8x64', v, 'ko', Q)
CONFIGS['wBw_typical'] = ('tc_8x64', 'typical', 'widths', Q)
CONFIGS['wBdm128_typical'] = ('tc_8x64', 'typical', 'dm128', Q)
CONFIGS['wBdmfb_typical'] = ('tc_8x64', 'typical', 'dmfb', Q)
PAIRS = [('wBp2_typical', 'wBko_typical'), ('wBp2_worst', 'wBko_worst'), ('wBp2_typical', 'wBw_typical'),
         ('wBp2_typical', 'wBdm128_typical'), ('wBp2_typical', 'wBdmfb_typical'), ('stock_wB', 'stock_wB_ko')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--p3freq-root', required=True, help="P3's build directory: the 8x64 path's builds and stock_wB_e64")
    ap.add_argument('--b7', required=True, help="amendment 7's build directory (stock_ko)")
    ap.add_argument('--table', required=True, help='the P3 table (the adopted table with the 8x64 rows)')
    ap.add_argument('--sensitivity', required=True, help="p3_tables.py sensitivity's output")
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
    p3 = json.loads(Path(args.table).read_text())
    sens = json.loads(Path(args.sensitivity).read_text())
    root = Path(args.p3freq_root)
    kernels = dict(stock_ko=KernelSet('stock_ko', build_root=args.b7, table=ko_table),
                   stock_wB=Kernel.load('stock_wB'),
                   stock_wB_ko=KernelSet('stock_wB_ko', build_root=root, table=args.table),
                   p2=KernelSet('mixed_wB_t0', build_root=root, table=paper_table),
                   ko=KernelSet('mixed_wB_ko', build_root=root, table=args.table),
                   widths=KernelSet('mixed_wB_ko', build_root=root, table={'mixed_wB': p3['mixed_wB']}),
                   dm128=KernelSet('mixed_wB_ko', build_root=root, table=sens['tables']['dm128']),
                   dmfb=KernelSet('mixed_wB_ko', build_root=root, table=sens['tables']['dmfb']))
    assert Path(kernels['stock_wB'].path).parent.parent == REPO / 'sm120' / 'build'
    assert all(Path(k.path).parent.parent == Path(args.b7) for k in kernels['stock_ko'].kernels.values())
    wb = ('p2', 'ko', 'widths', 'dm128', 'dmfb')
    for name in wb + ('stock_wB_ko',):
        assert all(Path(k.path).parent.parent == root for k in kernels[name].kernels.values()), name
    for name in wb:
        ks = kernels[name]
        assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1 for k in ks.kernels.values()), name
        assert all(str(k.manifest['blob_gen'].get('TAG0')) == '0' for k in ks.kernels.values()), name
        assert ks.sha256 == kernels['ko'].sha256, name          # the same libraries: only the table differs
    assert not (kernels['stock_wB_ko'].primary.manifest.get('extra_defines') or {})
    assert kernels['ko'].schedules and kernels['stock_wB_ko'].schedules
    assert not any(kernels[n].schedules for n in ('p2', 'widths', 'dm128', 'dmfb'))
    assert kernels['p2'].table and kernels['ko'].table == kernels['widths'].table
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = "kernel-opt amendment 12 (the 8x64 plan's P3, with P4), M1"
    res['roots'] = dict(p3freq=str(root), b7=str(args.b7), paper=str(REPO / 'sm120' / 'build'))
    res['tables'] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                     for p in (paper_table, ko_table, Path(args.table), Path(args.sensitivity))}
    B.write(args.out, res)


if __name__ == '__main__':
    main()

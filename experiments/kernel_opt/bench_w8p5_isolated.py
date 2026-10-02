#!/usr/bin/env python3
"""Kernel-opt amendment 13 (the 8x64 plan's P5, decision a), measurement M1: the adopted 8x64 path against its own
no-dispatch ceiling at every width (the same-placement reference), and against stock.

    python experiments/kernel_opt/bench_w8p5_isolated.py --model llama8b --p3freq-root DIR --ceil-root DIR --b7 DIR \
        --artifact fo6=ART --artifact tc_8x64=ART --out JSON

The deviation-2 method, exactly as bench_w8_isolated.py (amendment 10), through bench_ab_isolated.run(). Every set reads
the tracked adopted table (sm120/configs/<gpu>.ko.json); the ceiling takes the 8x64 path's widths and scheduler rows
(select.TABLE_FAMILY / SCHEDULE_FAMILY; amendment 12b adopted no scheduler rows for the 8x64 path), so the two run the
same tiles in the same order. Configurations (activation
quantizer FourOverSix):
  stock_ko        KernelSet('stock_ko') from --b7: the target (stock, weights on A)
  stock_wB_ko     KernelSet('stock_wB_ko') from --p3freq-root: stock_wB tuned alike (#4's epilogue tile, its rows)
  ceiling         KernelSet('nodisp_wB_ko') from --ceil-root on the FourOverSix artifact (E2M1 only): the 8x64 path's
                  tiles with the format dispatch compiled out
  wB_{v}          KernelSet('mixed_wB_ko') from --p3freq-root on the TM-OPT+TC 8x64 artifact: the adopted 8x64 path (t0,
                  #2's dispatch, the adopted widths)
The ceiling computes E2M1 everywhere, so it has no bitwise counterpart; the report checks that it ran the 8x64 path's
width and scheduler setting in every cell.
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
CONFIGS = OrderedDict(stock_ko=('fo6', 'first', 'stock_ko', Q), stock_wB_ko=('fo6', 'first', 'stock_wB_ko', Q),
                      ceiling=('fo6', 'first', 'nodisp_wB_ko', Q))
for v in ('typical', 'worst'):
    CONFIGS[f'wB_{v}'] = ('tc_8x64', v, 'mixed_wB_ko', Q)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--p3freq-root', required=True, help="the adopted 8x64 path's build directory (and stock_wB_e64)")
    ap.add_argument('--ceil-root', required=True, help="the build directory of the ceiling's builds (nodisp_wB_ko)")
    ap.add_argument('--b7', required=True, help="amendment 7's build directory (stock_ko)")
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
    ko_table = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    kernels = dict(stock_ko=KernelSet('stock_ko', build_root=args.b7),
                   stock_wB_ko=KernelSet('stock_wB_ko', build_root=args.p3freq_root),
                   nodisp_wB_ko=KernelSet('nodisp_wB_ko', build_root=args.ceil_root),
                   mixed_wB_ko=KernelSet('mixed_wB_ko', build_root=args.p3freq_root))
    roots = dict(stock_ko=Path(args.b7), stock_wB_ko=Path(args.p3freq_root), nodisp_wB_ko=Path(args.ceil_root),
                 mixed_wB_ko=Path(args.p3freq_root))
    for name, ks in kernels.items():
        assert ks.table_source == str(ko_table), (name, ks.table_source)
        assert all(Path(k.path).parent.parent == roots[name] for k in ks.kernels.values()), name
    assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1 for k in kernels['mixed_wB_ko'].kernels.values())
    assert kernels['nodisp_wB_ko'].table == kernels['mixed_wB_ko'].table
    assert kernels['nodisp_wB_ko'].schedules == kernels['mixed_wB_ko'].schedules   # the adopted 8x64 rows have none
    assert set(kernels['nodisp_wB_ko'].kernels) == set(kernels['mixed_wB_ko'].kernels)
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=[])
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = "kernel-opt amendment 13 (the 8x64 plan's P5: the no-dispatch ceiling at every width), M1"
    res['roots'] = {k: str(v) for k, v in roots.items()}
    res['tables'] = {str(ko_table): hashlib.sha256(ko_table.read_bytes()).hexdigest()}
    B.write(args.out, res)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Kernel-opt amendment 3 (A', 256x64 maps on a 4-arm kernel), measurement M1: GEMM time, today's path vs A'.

    python experiments/kernel_opt/bench_abA1_isolated.py --model llama8b --a1-root DIR \
        --artifact fo6=ART --artifact tc_256x64=ART --out JSON

The deviation-2 method, exactly as bench_ab_isolated.py (whose run() this calls): isolated launches, cold weights by
rotation + a 512 MiB read-flush, CUPTI, 3 rotated rounds x 30 repetitions, typical and worst tags, telemetry, the
registered checks, and each after configuration's output bitwise equal to its before configuration's on the timed
operands. Configurations (activation quantizer FourOverSix everywhere):
  stock_wA                   FourOverSix on 'auto_stock' (sm120/build): the reference
  mixed_256x64_{v}           TM-OPT+TC 256x64 on KernelSet('mixed') from sm120/build: today's path (before)
  g32_256x64_{v}             the same maps on KernelSet('mixed256') from --a1-root: the 4-arm 32-row-granule builds,
                             with the 'mixed' tile-table rows, so each call runs at the same width as before (after)
"""
import argparse
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
from mixfp4_sm120.select import KernelSet  # noqa: E402

Q = 'four_over_six_rows'
CONFIGS = OrderedDict(stock_wA=('fo6', 'first', 'stock', Q))
for v in ('typical', 'worst'):
    CONFIGS[f'mixed_256x64_{v}'] = ('tc_256x64', v, 'mixed', Q)
    CONFIGS[f'g32_256x64_{v}'] = ('tc_256x64', v, 'mixed256', Q)
PAIRS = [(f'mixed_256x64_{v}', f'g32_256x64_{v}') for v in ('typical', 'worst')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--a1-root', required=True, help="build directory of the A' builds")
    ap.add_argument('--tokens', default=','.join(map(str, AB.TOKENS)))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--configs', default=None)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.after_root = args.a1_root
    B.require_idle()
    kernels = dict(stock=KernelSet('stock'), mixed=KernelSet('mixed'), mixed256=KernelSet('mixed256', build_root=args.a1_root))
    paper = REPO / 'sm120' / 'build'
    for name, root in (('stock', paper), ('mixed', paper), ('mixed256', Path(args.a1_root))):
        assert all(Path(k.path).parent.parent == root for k in kernels[name].kernels.values()), name
    for w, k in kernels['mixed256'].kernels.items():
        assert k.cfg.map_tile_rows == 128 and tuple(k.type_block) == (32, 64), k.cfg.name
    assert kernels['mixed256'].table == kernels['mixed'].table, "'mixed256' does not use the 'mixed' tile-table rows"
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = "kernel-opt amendment 3 (A')"
    B.write(args.out, res)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Kernel-opt amendment 5 (#4: epilogue tile and tile-scheduler order), measurement M1: GEMM time before vs after, for
the mixed 16x64 family and the stock family alike.

    python experiments/kernel_opt/bench_ab4_isolated.py --model llama8b --root4 DIR --freq-root DIR --a1-root DIR \
        --table4 JSON --artifact fo6=ART --artifact tc_16x64=ART --artifact tc_256x64=ART --out JSON

The deviation-2 method, exactly as bench_ab_isolated.py (whose run() this calls, with each set's scheduler setting
passed to the timed GEMM): isolated launches, cold weights by rotation + a 512 MiB read-flush, CUPTI, 3 rotated rounds x
30 repetitions, typical and worst tags, telemetry, the registered checks, and each after configuration's output bitwise
equal to its before configuration's on the timed operands. Configurations (activation quantizer FourOverSix everywhere):
  stock_wA                   FourOverSix on KernelSet('stock') from sm120/build, the current table: before
  stock_e                    FourOverSix on KernelSet('stock_e') from --root4 with --table4 (its schedule rows): after
  mixed_16x64_{v}            TM-OPT+TC 16x64 on KernelSet('mixed') from sm120/build, the current table: before
  e_16x64_{v}                the same maps on KernelSet('mixed_e') from --root4 with --table4: after
  freq_16x64_{v}             the same maps on KernelSet('mixed') from --freq-root (#2), the current table: a reference
  m256_{v}                   TM-OPT+TC 256x64 on KernelSet('mixed256') from --a1-root (A', adopted), the current table:
                             unchanged by #4, a reference for the 256x64 gap to stock before and after
Both tables have the same width rows, so every call runs at the same width before and after.
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
from mixfp4_sm120 import select as S  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

Q = 'four_over_six_rows'
CONFIGS = OrderedDict(stock_wA=('fo6', 'first', 'stock', Q), stock_e=('fo6', 'first', 'stock_e', Q))
for v in ('typical', 'worst'):
    CONFIGS[f'mixed_16x64_{v}'] = ('tc_16x64', v, 'mixed', Q)
    CONFIGS[f'e_16x64_{v}'] = ('tc_16x64', v, 'mixed_e', Q)
    CONFIGS[f'freq_16x64_{v}'] = ('tc_16x64', v, 'mixed_freq', Q)
    CONFIGS[f'm256_{v}'] = ('tc_256x64', v, 'mixed256', Q)
PAIRS = ([('stock_wA', 'stock_e')] + [(f'mixed_16x64_{v}', f'e_16x64_{v}') for v in ('typical', 'worst')]
         + [(f'mixed_16x64_{v}', f'freq_16x64_{v}') for v in ('typical', 'worst')])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--root4', required=True)
    ap.add_argument('--freq-root', required=True)
    ap.add_argument('--a1-root', required=True)
    ap.add_argument('--table4', required=True, type=Path)
    ap.add_argument('--tokens', default=','.join(map(str, AB.TOKENS)))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--configs', default=None)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.after_root = args.root4
    B.require_idle()
    cur = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    kernels = dict(stock=KernelSet('stock', table=cur), stock_e=KernelSet('stock_e', build_root=args.root4, table=args.table4),
                   mixed=KernelSet('mixed', table=cur), mixed_e=KernelSet('mixed_e', build_root=args.root4, table=args.table4),
                   mixed_freq=KernelSet('mixed', build_root=args.freq_root, table=cur),
                   mixed256=KernelSet('mixed256', build_root=args.a1_root, table=cur))
    paper = REPO / 'sm120' / 'build'
    roots = dict(stock=paper, stock_e=Path(args.root4), mixed=paper, mixed_e=Path(args.root4), mixed_freq=Path(args.freq_root),
                 mixed256=Path(args.a1_root))
    for name, ks in kernels.items():
        assert all(Path(k.path).parent.parent == roots[name] for k in ks.kernels.values()), name
    assert kernels['mixed_e'].table == kernels['mixed'].table and kernels['stock_e'].table == kernels['stock'].table
    assert kernels['mixed_e'].schedules and kernels['stock_e'].schedules, '--table4 has no schedule rows'
    assert not any(kernels[f].schedules for f in ('mixed', 'stock', 'mixed_freq', 'mixed256'))
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = 'kernel-opt amendment 5 (#4)'
    res['tables'] = dict(before=str(cur), after=str(args.table4))
    B.write(args.out, res)


if __name__ == '__main__':
    main()

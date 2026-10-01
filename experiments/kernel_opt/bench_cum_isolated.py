#!/usr/bin/env python3
"""Kernel-opt amendment 9 (the cumulative registered run), measurement M1: GEMM time of the combined 16x64 build and of
the adopted stock set against today's paper builds.

    python experiments/kernel_opt/bench_cum_isolated.py --model llama8b --b7 DIR --b7freq DIR \
        --artifact fo6=ART --artifact tc_16x64=ART --out JSON

The deviation-2 method, exactly as bench_ab_isolated.py (whose run() this calls, with each set's scheduler setting
passed to the timed GEMM): isolated launches, cold weights by rotation + a 512 MiB read-flush, CUPTI, 3 rotated rounds x
30 repetitions, typical and worst tags, telemetry, the registered checks, and each adopted configuration's output bitwise
equal to its paper configuration's on the timed operands. Configurations (activation quantizer FourOverSix everywhere):
  stock_paper          FourOverSix on KernelSet('stock') from sm120/build, the paper table: the paper reference
  stock_ko             FourOverSix on KernelSet('stock_ko') from --b7, the adopted table: the adopted reference
  paper_16x64_{v}      TM-OPT+TC 16x64 on KernelSet('mixed') from sm120/build, the paper table: today's paper build
  comb_16x64_{v}       the same maps on KernelSet('mixed_ko') from --b7freq (MIXFP4_DISPATCH_FREQ=1), the adopted table:
                       the combined build (#2's dispatch + t0 + #4 + 4b widths + scheduler rows)
  ko_16x64_{v}         the same maps on KernelSet('mixed_ko') from --b7 (default dispatch), the adopted table: the
                       adopted default-dispatch path, a reference for #2's dispatch vs the default on real maps
The adopted table has the 4b widths, so the adopted and the paper configurations may run at different widths; each row
records its width and scheduler setting.
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
CONFIGS = OrderedDict(stock_paper=('fo6', 'first', 'stock', Q), stock_ko=('fo6', 'first', 'stock_ko', Q))
for v in ('typical', 'worst'):
    CONFIGS[f'paper_16x64_{v}'] = ('tc_16x64', v, 'mixed', Q)
    CONFIGS[f'comb_16x64_{v}'] = ('tc_16x64', v, 'mixed_ko_freq', Q)
    CONFIGS[f'ko_16x64_{v}'] = ('tc_16x64', v, 'mixed_ko', Q)
PAIRS = ([('stock_paper', 'stock_ko')] + [(f'paper_16x64_{v}', f'comb_16x64_{v}') for v in ('typical', 'worst')]
         + [(f'paper_16x64_{v}', f'ko_16x64_{v}') for v in ('typical', 'worst')])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--b7', required=True, help="amendment 7's build directory (mixed_ko default dispatch, stock_ko)")
    ap.add_argument('--b7freq', required=True, help="amendment 7's mixed_ko builds with MIXFP4_DISPATCH_FREQ=1")
    ap.add_argument('--tokens', default=','.join(map(str, AB.TOKENS)))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--configs', default=None)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.after_root = args.b7freq
    B.require_idle()
    paper_table = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    ko_table = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    assert ko_table.exists(), ko_table
    kernels = dict(stock=KernelSet('stock', table=paper_table),
                   stock_ko=KernelSet('stock_ko', build_root=args.b7, table=ko_table),
                   mixed=KernelSet('mixed', table=paper_table),
                   mixed_ko_freq=KernelSet('mixed_ko', build_root=args.b7freq, table=ko_table),
                   mixed_ko=KernelSet('mixed_ko', build_root=args.b7, table=ko_table))
    paper = REPO / 'sm120' / 'build'
    roots = dict(stock=paper, stock_ko=Path(args.b7), mixed=paper, mixed_ko_freq=Path(args.b7freq), mixed_ko=Path(args.b7))
    for name, ks in kernels.items():
        assert all(Path(k.path).parent.parent == roots[name] for k in ks.kernels.values()), name
    assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1
               for k in kernels['mixed_ko_freq'].kernels.values())
    for name in ('mixed', 'mixed_ko', 'stock', 'stock_ko'):
        assert all('MIXFP4_DISPATCH_FREQ' not in (k.manifest.get('extra_defines') or {}) for k in kernels[name].kernels.values())
    assert kernels['mixed_ko_freq'].schedules and kernels['stock_ko'].schedules, 'the adopted table has no schedule rows'
    assert kernels['mixed_ko_freq'].schedules == kernels['mixed_ko'].schedules
    assert kernels['mixed_ko_freq'].table == kernels['mixed_ko'].table
    assert not kernels['mixed'].schedules and not kernels['stock'].schedules
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = 'kernel-opt amendment 9 (the cumulative registered run), M1'
    res['roots'] = {k: str(v) for k, v in roots.items()}
    res['tables'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (paper_table, ko_table)}
    B.write(args.out, res)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Kernel-opt amendment 10 (the 8x64 baseline), measurement M1: GEMM time of the 8x64 path on the current best kernels
against stock, weights on A (the deployment reference) and on B (the same placement).

    python experiments/kernel_opt/bench_w8_isolated.py --model llama8b --kopt-root DIR --freq-root DIR --b7 DIR \
        --artifact fo6=ART --artifact tc_8x64=ART --out JSON

The deviation-2 method, exactly as bench_ab_isolated.py (whose run() this calls): isolated launches, cold weights by
rotation + a 512 MiB read-flush, activations quantized after the flush, CUPTI, 3 rotated rounds x 30 repetitions,
typical and worst tags, telemetry, the registered checks. Configurations (activation quantizer FourOverSix everywhere):
  stock_paper      FourOverSix on KernelSet('stock') from sm120/build, the paper table: stock_wA as in the paper
  stock_ko         FourOverSix on KernelSet('stock_ko') from --b7, the adopted table: the deployment's stock_wA
  stock_wB         FourOverSix on stock_wB (sm120/build, one 128-wide build): the same-placement reference
  wB_{v}           TM-OPT+TC 8x64 on KernelSet('mixed_wB') from --kopt-root (optimization 1b's builds), the paper table
                   (1b's mixed_wB rows): today's best 8x64 path, default dispatch
  wBfreq_{v}       the same maps on KernelSet('mixed_wB') from --freq-root (#2's builds, MIXFP4_DISPATCH_FREQ=1), the same
                   table: #2's dispatch
Added check: wBfreq's output equals wB's bitwise on the timed operands.
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
CONFIGS = OrderedDict(stock_paper=('fo6', 'first', 'stock', Q), stock_ko=('fo6', 'first', 'stock_ko', Q),
                      stock_wB=('fo6', 'first', 'stock_wB', Q))
for v in ('typical', 'worst'):
    CONFIGS[f'wB_{v}'] = ('tc_8x64', v, 'mixed_wB', Q)
    CONFIGS[f'wBfreq_{v}'] = ('tc_8x64', v, 'mixed_wB_freq', Q)
PAIRS = [(f'wB_{v}', f'wBfreq_{v}') for v in ('typical', 'worst')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--kopt-root', required=True, help="optimization 1b's build directory (the mixed_wB set)")
    ap.add_argument('--freq-root', required=True, help="#2's build directory (mixed_wB with MIXFP4_DISPATCH_FREQ=1)")
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
    args.after_root = args.kopt_root
    B.require_idle()
    paper_table = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    ko_table = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    assert ko_table.exists(), ko_table
    kernels = dict(stock=KernelSet('stock', table=paper_table),
                   stock_ko=KernelSet('stock_ko', build_root=args.b7, table=ko_table),
                   stock_wB=Kernel.load('stock_wB'),
                   mixed_wB=KernelSet('mixed_wB', build_root=args.kopt_root, table=paper_table),
                   mixed_wB_freq=KernelSet('mixed_wB', build_root=args.freq_root, table=paper_table))
    paper = REPO / 'sm120' / 'build'
    roots = dict(stock=paper, stock_ko=Path(args.b7), stock_wB=paper, mixed_wB=Path(args.kopt_root),
                 mixed_wB_freq=Path(args.freq_root))
    for name, ks in kernels.items():
        libs = ks.kernels.values() if isinstance(ks, KernelSet) else [ks]
        assert all(Path(k.path).parent.parent == roots[name] for k in libs), name
    assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1
               for k in kernels['mixed_wB_freq'].kernels.values())
    for name in ('mixed_wB', 'stock', 'stock_ko'):
        assert all('MIXFP4_DISPATCH_FREQ' not in (k.manifest.get('extra_defines') or {}) for k in kernels[name].kernels.values())
    assert kernels['mixed_wB'].table and kernels['mixed_wB'].table == kernels['mixed_wB_freq'].table
    assert not kernels['mixed_wB'].schedules and not kernels['stock'].schedules and kernels['stock_ko'].schedules
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = 'kernel-opt amendment 10 (the 8x64 baseline), M1'
    res['roots'] = {k: str(v) for k, v in roots.items()}
    res['tables'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (paper_table, ko_table)}
    B.write(args.out, res)


if __name__ == '__main__':
    main()

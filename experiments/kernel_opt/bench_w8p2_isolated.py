#!/usr/bin/env python3
"""Kernel-opt amendment 11 (the 8x64 plan's P2), measurement M1: GEMM time of the weights-on-B family without the site-0
prmt tags (t0), with the default and #2's dispatch, against the tagged builds and against stock.

    python experiments/kernel_opt/bench_w8p2_isolated.py --model llama8b --kopt-root DIR --freq-root DIR --t0-root DIR \
        --t0freq-root DIR --b7 DIR --artifact fo6=ART --artifact tc_8x64=ART --out JSON

The deviation-2 method, exactly as bench_w8_isolated.py (amendment 10), through bench_ab_isolated.run(). Configurations
(activation quantizer FourOverSix everywhere; every 8x64 set reads the paper table's 'mixed_wB' rows, 1b's):
  stock_ko         FourOverSix on KernelSet('stock_ko') from --b7, the adopted table: the target (stock, weights on A)
  stock_wB         FourOverSix on stock_wB (sm120/build): the same-placement reference
  wB_{v}           TM-OPT+TC 8x64 on KernelSet('mixed_wB') from --kopt-root: optimization 1b's builds, default dispatch
  wBfreq_{v}       the same on KernelSet('mixed_wB') from --freq-root: #2's builds (MIXFP4_DISPATCH_FREQ=1)
  wBt0_{v}         the same on KernelSet('mixed_wB_t0') from --t0-root: P2's t0 builds, default dispatch
  wBt0freq_{v}     the same on KernelSet('mixed_wB_t0') from --t0freq-root: P2's t0 builds with #2's dispatch
Added checks: wBfreq, wBt0 and wBt0freq each equal wB bitwise on the timed operands.
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
CONFIGS = OrderedDict(stock_ko=('fo6', 'first', 'stock_ko', Q), stock_wB=('fo6', 'first', 'stock_wB', Q))
for v in ('typical', 'worst'):
    CONFIGS[f'wB_{v}'] = ('tc_8x64', v, 'mixed_wB', Q)
    CONFIGS[f'wBfreq_{v}'] = ('tc_8x64', v, 'mixed_wB_freq', Q)
    CONFIGS[f'wBt0_{v}'] = ('tc_8x64', v, 'mixed_wB_t0', Q)
    CONFIGS[f'wBt0freq_{v}'] = ('tc_8x64', v, 'mixed_wB_t0_freq', Q)
PAIRS = [(f'wB_{v}', f'{s}_{v}') for v in ('typical', 'worst') for s in ('wBfreq', 'wBt0', 'wBt0freq')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--kopt-root', required=True, help="optimization 1b's build directory (the mixed_wB set)")
    ap.add_argument('--freq-root', required=True, help="#2's build directory (mixed_wB with MIXFP4_DISPATCH_FREQ=1)")
    ap.add_argument('--t0-root', required=True, help="P2's build directory (mixed_wB_t0)")
    ap.add_argument('--t0freq-root', required=True, help="P2's freq build directory (mixed_wB_t0, MIXFP4_DISPATCH_FREQ=1)")
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
    args.after_root = args.t0_root
    B.require_idle()
    paper_table = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    ko_table = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    assert ko_table.exists(), ko_table
    kernels = dict(stock_ko=KernelSet('stock_ko', build_root=args.b7, table=ko_table),
                   stock_wB=Kernel.load('stock_wB'),
                   mixed_wB=KernelSet('mixed_wB', build_root=args.kopt_root, table=paper_table),
                   mixed_wB_freq=KernelSet('mixed_wB', build_root=args.freq_root, table=paper_table),
                   mixed_wB_t0=KernelSet('mixed_wB_t0', build_root=args.t0_root, table=paper_table),
                   mixed_wB_t0_freq=KernelSet('mixed_wB_t0', build_root=args.t0freq_root, table=paper_table))
    paper = REPO / 'sm120' / 'build'
    roots = dict(stock_ko=Path(args.b7), stock_wB=paper, mixed_wB=Path(args.kopt_root), mixed_wB_freq=Path(args.freq_root),
                 mixed_wB_t0=Path(args.t0_root), mixed_wB_t0_freq=Path(args.t0freq_root))
    for name, ks in kernels.items():
        libs = ks.kernels.values() if isinstance(ks, KernelSet) else [ks]
        assert all(Path(k.path).parent.parent == roots[name] for k in libs), name
    for name in ('mixed_wB_freq', 'mixed_wB_t0_freq'):
        assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1 for k in kernels[name].kernels.values())
    for name in ('mixed_wB', 'mixed_wB_t0', 'stock_ko'):
        assert all('MIXFP4_DISPATCH_FREQ' not in (k.manifest.get('extra_defines') or {}) for k in kernels[name].kernels.values())
    for name in ('mixed_wB_t0', 'mixed_wB_t0_freq'):
        assert all(str(k.manifest['blob_gen'].get('TAG0')) == '0' for k in kernels[name].kernels.values()), name
    for name in ('mixed_wB', 'mixed_wB_freq'):
        assert all('TAG0' not in k.manifest['blob_gen'] for k in kernels[name].kernels.values()), name
    tabs = [kernels[n].table for n in ('mixed_wB', 'mixed_wB_freq', 'mixed_wB_t0', 'mixed_wB_t0_freq')]
    assert tabs[0] and all(t == tabs[0] for t in tabs)
    assert not any(kernels[n].schedules for n in ('mixed_wB', 'mixed_wB_freq', 'mixed_wB_t0', 'mixed_wB_t0_freq'))
    assert kernels['stock_ko'].schedules
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = "kernel-opt amendment 11 (the 8x64 plan's P2: t0 for the wB family), M1"
    res['roots'] = {k: str(v) for k, v in roots.items()}
    res['tables'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (paper_table, ko_table)}
    B.write(args.out, res)


if __name__ == '__main__':
    main()

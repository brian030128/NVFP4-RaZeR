#!/usr/bin/env python3
"""Kernel-opt amendment 2 (#2, frequency-aware dispatch), measurement M1': GEMM time, default vs MIXFP4_DISPATCH_FREQ.

    python experiments/kernel_opt/bench_ab2_isolated.py --model llama8b --freq-root DIR --kopt-root DIR \
        --artifact fo6=ART --artifact tc_16x64=ART --artifact tc_256x64=ART --artifact tc_8x64=ART --out JSON

The deviation-2 method, exactly as bench_ab_isolated.py (whose run() this calls): isolated launches, cold weights by
rotation + a 512 MiB read-flush, CUPTI, 3 rotated rounds x 30 repetitions, typical and worst tags, telemetry, the
registered checks, and each after configuration's output bitwise equal to its before configuration's on the timed
operands. Configurations (activation quantizer FourOverSix everywhere):
  stock_wA, stock_wB                FourOverSix on 'auto_stock' / stock_wB (sm120/build): the references
  mixed_{16x64,256x64}_{v}          TM-OPT+TC on KernelSet('mixed') from sm120/build: before (the paper's builds)
  freq_{16x64,256x64}_{v}           the same maps on KernelSet('mixed') from --freq-root: after
  wB_{v}                            TM-OPT+TC 8x64 on KernelSet('mixed_wB') from --kopt-root (optimization 1b): before
  wBfreq_{v}                        the same on KernelSet('mixed_wB') from --freq-root: after
Both weights-on-B sets use the same tile table, so wB vs wBfreq isolates the dispatch change.
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
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

Q = 'four_over_six_rows'
CONFIGS = OrderedDict(stock_wA=('fo6', 'first', 'stock', Q), stock_wB=('fo6', 'first', 'stock_wB', Q))
for unit in ('16x64', '256x64'):
    for v in ('typical', 'worst'):
        CONFIGS[f'mixed_{unit}_{v}'] = (f'tc_{unit}', v, 'mixed', Q)
        CONFIGS[f'freq_{unit}_{v}'] = (f'tc_{unit}', v, 'mixed_freq', Q)
for v in ('typical', 'worst'):
    CONFIGS[f'wB_{v}'] = ('tc_8x64', v, 'auto_wB', Q)
    CONFIGS[f'wBfreq_{v}'] = ('tc_8x64', v, 'auto_wB_freq', Q)
PAIRS = ([(f'mixed_{u}_{v}', f'freq_{u}_{v}') for u in ('16x64', '256x64') for v in ('typical', 'worst')]
         + [(f'wB_{v}', f'wBfreq_{v}') for v in ('typical', 'worst')])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--freq-root', required=True)
    ap.add_argument('--kopt-root', required=True, help='optimization 1b build directory (the before weights-on-B set)')
    ap.add_argument('--tokens', default=','.join(map(str, AB.TOKENS)))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--configs', default=None)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.after_root = args.freq_root
    B.require_idle()
    kernels = dict(stock=KernelSet('stock'), stock_wB=Kernel.load('stock_wB'), mixed=KernelSet('mixed'),
                   mixed_freq=KernelSet('mixed', build_root=args.freq_root),
                   auto_wB=KernelSet('mixed_wB', build_root=args.kopt_root),
                   auto_wB_freq=KernelSet('mixed_wB', build_root=args.freq_root))
    paper = REPO / 'sm120' / 'build'
    for name, root in (('stock', paper), ('mixed', paper), ('mixed_freq', Path(args.freq_root)),
                       ('auto_wB', Path(args.kopt_root)), ('auto_wB_freq', Path(args.freq_root))):
        assert all(Path(k.path).parent.parent == root for k in kernels[name].kernels.values()), name
    assert Path(kernels['stock_wB'].path).parent.parent == paper
    for name in ('mixed_freq', 'auto_wB_freq'):
        for k in kernels[name].kernels.values():
            assert k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1, k.path
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = 'kernel-opt amendment 2 (#2)'
    B.write(args.out, res)


if __name__ == '__main__':
    main()

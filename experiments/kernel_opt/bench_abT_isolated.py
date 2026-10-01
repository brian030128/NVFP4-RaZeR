#!/usr/bin/env python3
"""Kernel-opt amendment 6 (t0: the site-0 prmt tags dropped), measurement M1: GEMM time before vs after, for the
no-dispatch ceiling, the 16x64 kernels (default and #2's frequency-aware dispatch) and the 256x64 (A') kernels.

    python experiments/kernel_opt/bench_abT_isolated.py --model llama8b --t0-root DIR --t0freq-root DIR --freq-root DIR \
        --a1-root DIR --artifact fo6=ART --artifact tc_16x64=ART --artifact tc_256x64=ART --out JSON

The deviation-2 method, exactly as bench_ab_isolated.py (whose run() this calls): isolated launches, cold weights by
rotation + a 512 MiB read-flush, CUPTI, 3 rotated rounds x 30 repetitions, typical and worst tags, telemetry, the
registered checks, and each after configuration's output bitwise equal to its before configuration's on the timed
operands. Every set reads the current tile table, so before and after run at the same width. Configurations
(activation quantizer FourOverSix everywhere):
  stock_wA            FourOverSix on KernelSet('stock') from sm120/build: the reference
  nodisp              FourOverSix on n16k64_wA_nodisp (sm120/build, width 128 only): before
  nodisp_t0           FourOverSix on n16k64_wA_nodisp_t0 from --t0-root: after
  mixed_16x64_{v}     TM-OPT+TC 16x64 on KernelSet('mixed') from sm120/build: before (default dispatch)
  t0_16x64_{v}        the same maps on KernelSet('mixed_t0') from --t0-root: after
  freq_16x64_{v}      the same maps on KernelSet('mixed') from --freq-root (#2): before (frequency-aware dispatch)
  freqt0_16x64_{v}    the same maps on KernelSet('mixed_t0') from --t0freq-root: after
  m256_{v}            TM-OPT+TC 256x64 on KernelSet('mixed256') from --a1-root (A'): before
  m256t0_{v}          the same maps on KernelSet('mixed256_t0') from --t0-root: after
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
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

Q = 'four_over_six_rows'
CONFIGS = OrderedDict(stock_wA=('fo6', 'first', 'stock', Q), nodisp=('fo6', 'first', 'nodisp', Q),
                      nodisp_t0=('fo6', 'first', 'nodisp_t0', Q))
for v in ('typical', 'worst'):
    CONFIGS[f'mixed_16x64_{v}'] = ('tc_16x64', v, 'mixed', Q)
    CONFIGS[f't0_16x64_{v}'] = ('tc_16x64', v, 'mixed_t0', Q)
    CONFIGS[f'freq_16x64_{v}'] = ('tc_16x64', v, 'mixed_freq', Q)
    CONFIGS[f'freqt0_16x64_{v}'] = ('tc_16x64', v, 'mixed_freq_t0', Q)
    CONFIGS[f'm256_{v}'] = ('tc_256x64', v, 'mixed256', Q)
    CONFIGS[f'm256t0_{v}'] = ('tc_256x64', v, 'mixed256_t0', Q)
PAIRS = [('nodisp', 'nodisp_t0'), ('stock_wA', 'nodisp_t0')]
for v in ('typical', 'worst'):
    PAIRS += [(f'mixed_16x64_{v}', f't0_16x64_{v}'), (f'freq_16x64_{v}', f'freqt0_16x64_{v}'), (f'm256_{v}', f'm256t0_{v}')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--t0-root', required=True)
    ap.add_argument('--t0freq-root', required=True)
    ap.add_argument('--freq-root', required=True)
    ap.add_argument('--a1-root', required=True)
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
    cur = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    paper = REPO / 'sm120' / 'build'
    kernels = dict(stock=KernelSet('stock', table=cur), nodisp=Kernel.load('n16k64_wA_nodisp'),
                   nodisp_t0=Kernel.load('n16k64_wA_nodisp_t0', build_root=args.t0_root),
                   mixed=KernelSet('mixed', table=cur), mixed_t0=KernelSet('mixed_t0', build_root=args.t0_root, table=cur),
                   mixed_freq=KernelSet('mixed', build_root=args.freq_root, table=cur),
                   mixed_freq_t0=KernelSet('mixed_t0', build_root=args.t0freq_root, table=cur),
                   mixed256=KernelSet('mixed256', build_root=args.a1_root, table=cur),
                   mixed256_t0=KernelSet('mixed256_t0', build_root=args.t0_root, table=cur))
    roots = dict(stock=paper, nodisp=paper, nodisp_t0=Path(args.t0_root), mixed=paper, mixed_t0=Path(args.t0_root),
                 mixed_freq=Path(args.freq_root), mixed_freq_t0=Path(args.t0freq_root), mixed256=Path(args.a1_root),
                 mixed256_t0=Path(args.t0_root))
    for name, ks in kernels.items():
        libs = ks.kernels.values() if isinstance(ks, KernelSet) else [ks]
        assert all(Path(k.path).parent.parent == roots[name] for k in libs), name
        if isinstance(ks, KernelSet):
            assert ks.table and not ks.schedules, name
    for name in ('mixed_freq', 'mixed_freq_t0'):
        assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1 for k in kernels[name].kernels.values())
    for name in ('mixed', 'mixed_t0', 'mixed256', 'mixed256_t0'):
        assert all('MIXFP4_DISPATCH_FREQ' not in (k.manifest.get('extra_defines') or {}) for k in kernels[name].kernels.values())
    assert kernels['mixed_t0'].table == kernels['mixed'].table == kernels['mixed256_t0'].table
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = 'kernel-opt amendment 6 (t0)'
    res['roots'] = {k: str(v) for k, v in roots.items()}
    B.write(args.out, res)


if __name__ == '__main__':
    main()

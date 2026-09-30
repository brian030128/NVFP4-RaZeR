#!/usr/bin/env python3
"""Kernel-opt amendment 4 (tile-table re-tune), measurement M1: GEMM time under the current vs the re-tuned table.

    python experiments/kernel_opt/bench_ab_retune.py --model llama8b --new-table JSON --freq-root DIR --a1-root DIR \
        --artifact fo6=ART --artifact tc_16x64=ART --artifact tc_256x64=ART --out JSON

The deviation-2 method, exactly as bench_ab_isolated.py (whose run() this calls): isolated launches, cold weights by
rotation + a 512 MiB read-flush, CUPTI, 3 rotated rounds x 30 repetitions, typical and worst tags, telemetry, the
registered checks. The same builds run under the current table (sm120/configs/<gpu>.json, 'cur') and under --new-table
('new'); every 'new' call's output must equal its 'cur' call's bitwise on the timed operands (the widths are
interchangeable). Configurations (activation quantizer FourOverSix everywhere):
  stock_wA_{cur,new}              FourOverSix on KernelSet('stock') from sm120/build
  m16_{v}_{cur,new}               TM-OPT+TC 16x64 on KernelSet('mixed') from --freq-root (#2's builds, the 16x64 path)
  m256_{v}_{cur,new}              TM-OPT+TC 256x64 on KernelSet('mixed256') from --a1-root (A', adopted), which reads the
                                  'mixed' rows
Each row records the width its call ran at, so the cells whose width changed can be listed.
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
CONFIGS = OrderedDict()
for tb in ('cur', 'new'):
    CONFIGS[f'stock_wA_{tb}'] = ('fo6', 'first', f'stock_{tb}', Q)
for v in ('typical', 'worst'):
    for tb in ('cur', 'new'):
        CONFIGS[f'm16_{v}_{tb}'] = ('tc_16x64', v, f'mixed_{tb}', Q)
    for tb in ('cur', 'new'):
        CONFIGS[f'm256_{v}_{tb}'] = ('tc_256x64', v, f'mixed256_{tb}', Q)
PAIRS = [('stock_wA_cur', 'stock_wA_new')] + [(f'{u}_{v}_cur', f'{u}_{v}_new') for u in ('m16', 'm256')
                                               for v in ('typical', 'worst')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--new-table', required=True, type=Path)
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
    args.after_root = args.freq_root
    B.require_idle()
    cur = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    kernels = {}
    for tb, table in (('cur', cur), ('new', args.new_table)):
        kernels[f'stock_{tb}'] = KernelSet('stock', table=table)
        kernels[f'mixed_{tb}'] = KernelSet('mixed', build_root=args.freq_root, table=table)
        kernels[f'mixed256_{tb}'] = KernelSet('mixed256', build_root=args.a1_root, table=table)
    paper = REPO / 'sm120' / 'build'
    for name, ks in kernels.items():
        root = paper if name.startswith('stock') else Path(args.freq_root if name.startswith('mixed_') else args.a1_root)
        assert all(Path(k.path).parent.parent == root for k in ks.kernels.values()), name
        assert ks.table, f'{name}: empty table'
    for tb in ('cur', 'new'):
        assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1
                   for k in kernels[f'mixed_{tb}'].kernels.values())
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = 'kernel-opt amendment 4 (tile-table re-tune)'
    res['tables'] = dict(cur=str(cur), new=str(args.new_table))
    B.write(args.out, res)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Exploratory (disclosed, not registered): the 256x64 (A') build variants per T at today's widths, by M1's method
(bench_ab_isolated.run: cold weights by rotation + flush, activations quantized after the flush, CUDA-event time of
isolated launches, rotated rounds), on one model's typical and worst modules of the TC 256x64 artifact.

    python results/kernel_opt/V/exploration/quick_V1.py --model llama8b --out JSON

Every mixed set reads the paper table's 'mixed' rows (today's mixed256 widths), so only the builds differ:
  today        KernelSet('mixed256') from build_A1: today's adopted 256x64 path (A', default dispatch, site-0 tags)
  e            KernelSet('mixed256_e') from build_VD0: #4's 64 x 64 epilogue tile at width 128 (n16k64_wA_g32_e64)
  eF, eU, eFU  the same from build_VF / build_VU / build_VFU: + #2's dispatch / the uniform index / both
  *_t0         KernelSet('mixed256_e_t0'): the same four without the site-0 prmt tags
  stock_ko     KernelSet('stock_ko') from build_7 (the adopted table): the target
  ceil         KernelSet('nodisp_ko') from build_C3k on the paper table: the tile with the dispatch compiled out (E2M1)
Bitwise pairs on the timed operands: every variant equals today's path.
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

WT = '/home/dev/n16k64_campaign/kernel_opt/wtV'
REPO = Path('/home/dev/NVFP4-RaZeR')
sys.path.insert(0, f'{WT}/sm120')
sys.path.insert(1, str(REPO / 'experiments' / 'kernel_opt'))
sys.path.insert(2, str(REPO / 'experiments' / 'paper'))
sys.path.insert(3, f'{WT}/sm120/bench')
import mixfp4_sm120  # noqa: E402
assert '/wtV/' in mixfp4_sm120.__file__
import bench_ab_isolated as AB  # noqa: E402
import bench_gemm_isolated as G  # noqa: E402
import common as B  # noqa: E402
from mixfp4_sm120 import select as S  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

KO = Path('/home/dev/n16k64_campaign/kernel_opt')
ART = Path('/home/dev/n16k64_campaign/paper/artifacts')
Q = 'four_over_six_rows'
VARIANTS = OrderedDict([('e', ('mixed256_e', 'build_VD0')), ('eF', ('mixed256_e', 'build_VF')),
                        ('eU', ('mixed256_e', 'build_VU')), ('eFU', ('mixed256_e', 'build_VFU')),
                        ('e_t0', ('mixed256_e_t0', 'build_VD0')), ('eF_t0', ('mixed256_e_t0', 'build_VF')),
                        ('eU_t0', ('mixed256_e_t0', 'build_VU')), ('eFU_t0', ('mixed256_e_t0', 'build_VFU'))])
DEFINES = dict(e={}, eF={'MIXFP4_DISPATCH_FREQ': 1}, eU={'MIXFP4_UNIFORM_DISPATCH': 1},
               eFU={'MIXFP4_DISPATCH_FREQ': 1, 'MIXFP4_UNIFORM_DISPATCH': 1})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--tokens', default=','.join(map(str, AB.TOKENS)))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--configs', default=None)
    ap.add_argument('--tags', default='typical,worst')
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.artifact = [f'{k}={ART}/{args.model}_{k}' for k in ('fo6', 'tc_256x64')]
    args.after_root = str(KO / 'build_VFU')
    B.require_idle()
    paper = S.TABLE_DIR / f'{S.gpu_slug()}.json'
    ko = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    kernels = dict(stock_ko=KernelSet('stock_ko', build_root=KO / 'build_7', table=ko),
                   ceil=KernelSet('nodisp_ko', build_root=KO / 'build_C3k', table=paper),
                   today=KernelSet('mixed256', build_root=KO / 'build_A1', table=paper))
    for name, (fam, root) in VARIANTS.items():
        kernels[name] = KernelSet(fam, build_root=KO / root, table=paper)
        want = DEFINES[name.replace('_t0', '')]
        for k in kernels[name].kernels.values():
            assert (k.manifest.get('extra_defines') or {}) == want, (name, k.cfg.name, k.manifest.get('extra_defines'))
    configs = OrderedDict(stock_ko=('fo6', 'first', 'stock_ko', Q), ceil=('fo6', 'first', 'ceil', Q))
    pairs = []
    for v in args.tags.split(','):
        configs[f'today_{v}'] = ('tc_256x64', v, 'today', Q)
        for name in VARIANTS:
            configs[f'{name}_{v}'] = ('tc_256x64', v, name, Q)
            pairs.append((f'today_{v}', f'{name}_{v}'))
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=configs, kernels=kernels, pairs=pairs)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = 'kernel-opt V (256x64), exploratory per-T timing of the build variants (not registered)'
    B.write(args.out, res)


if __name__ == '__main__':
    main()

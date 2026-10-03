#!/usr/bin/env python3
"""Exploratory (disclosed, not registered): the uniform-branch dispatch candidates per T through the adopted tables, by
M1's method (bench_ab_isolated.run: cold weights by rotation + flush, activations quantized after the flush, CUDA-event
time of isolated launches, rotated rounds), on one model's typical modules.

    python results/kernel_opt/U/exploration/quick_U3.py --model llama8b --out JSON [--rounds 3] [--tokens ...]

Every set reads the tracked adopted table (sm120/configs/<gpu>.ko.json). Configurations (activations FourOverSix):
  stock_ko            KernelSet('stock_ko') from build_7: the target
  c16                 KernelSet('nodisp_ko') from build_C3k on FourOverSix (E2M1 only): the 16x64 ceiling
  k16_freq_typical    KernelSet('mixed_ko') from build_7freq: today's adopted 16x64 path (t0, #2's dispatch)
  k16_uni_typical     the same family from build_U1: + MIXFP4_UNIFORM_DISPATCH=1 (redux.sync.or index)
  c8                  KernelSet('nodisp_wB_ko') from build_P5 on FourOverSix: the 8x64 ceiling
  k8_freq_typical     KernelSet('mixed_wB_ko') from build_P3freq: today's adopted 8x64 path (t0, #2's dispatch)
  k8_uni_typical      the same family from build_U1: + MIXFP4_UNIFORM_DISPATCH=1 (the non-pipelined dispatch)
  k8_unip_typical     from build_U1p: + MIXFP4_PIPE_FLAGS=1 + MIXFP4_UNIFORM_DISPATCH=1
  k8_pipe_typical     from build_U0p: + MIXFP4_PIPE_FLAGS=1 (control: the pipelined flags alone)
  k8_hyb_typical      from build_U8h (added after the Llama run, chosen on it): build_U1p's builds at widths 16, 32,
                      64 and '128x64', build_U0p's at width 128
Bitwise pairs on the timed operands: every candidate equals today's adopted path of its family.
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path('/home/dev/NVFP4-RaZeR')
sys.path.insert(0, str(REPO / 'experiments' / 'kernel_opt'))
sys.path.insert(0, str(REPO / 'experiments' / 'paper'))
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import bench_ab_isolated as AB  # noqa: E402
import bench_gemm_isolated as G  # noqa: E402
import common as B  # noqa: E402
from mixfp4_sm120 import select as S  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

KO = Path('/home/dev/n16k64_campaign/kernel_opt')
ART = Path('/home/dev/n16k64_campaign/paper/artifacts')
Q = 'four_over_six_rows'
CONFIGS = OrderedDict(stock_ko=('fo6', 'first', 'stock_ko', Q),
                      c16=('fo6', 'first', 'c16', Q),
                      k16_freq_typical=('tc_16x64', 'typical', 'k16_freq', Q),
                      k16_uni_typical=('tc_16x64', 'typical', 'k16_uni', Q),
                      c8=('fo6', 'first', 'c8', Q),
                      k8_freq_typical=('tc_8x64', 'typical', 'k8_freq', Q),
                      k8_uni_typical=('tc_8x64', 'typical', 'k8_uni', Q),
                      k8_unip_typical=('tc_8x64', 'typical', 'k8_unip', Q),
                      k8_pipe_typical=('tc_8x64', 'typical', 'k8_pipe', Q),
                      k8_hyb_typical=('tc_8x64', 'typical', 'k8_hyb', Q),
                      # added after the Phi-4 run: the worst (densest) modules, run with --configs
                      k16_freq_worst=('tc_16x64', 'worst', 'k16_freq', Q),
                      k16_uni_worst=('tc_16x64', 'worst', 'k16_uni', Q),
                      k8_freq_worst=('tc_8x64', 'worst', 'k8_freq', Q),
                      k8_pipe_worst=('tc_8x64', 'worst', 'k8_pipe', Q))
PAIRS = [('k16_freq_typical', 'k16_uni_typical'), ('k8_freq_typical', 'k8_uni_typical'),
         ('k8_freq_typical', 'k8_unip_typical'), ('k8_freq_typical', 'k8_pipe_typical'),
         ('k8_freq_typical', 'k8_hyb_typical'), ('k16_freq_worst', 'k16_uni_worst'), ('k8_freq_worst', 'k8_pipe_worst')]
DEFINES = dict(k16_freq={'MIXFP4_DISPATCH_FREQ': 1},
               k16_uni={'MIXFP4_DISPATCH_FREQ': 1, 'MIXFP4_UNIFORM_DISPATCH': 1},
               k8_freq={'MIXFP4_DISPATCH_FREQ': 1},
               k8_uni={'MIXFP4_DISPATCH_FREQ': 1, 'MIXFP4_UNIFORM_DISPATCH': 1},
               k8_unip={'MIXFP4_DISPATCH_FREQ': 1, 'MIXFP4_PIPE_FLAGS': 1, 'MIXFP4_UNIFORM_DISPATCH': 1},
               k8_pipe={'MIXFP4_DISPATCH_FREQ': 1, 'MIXFP4_PIPE_FLAGS': 1})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--tokens', default=','.join(map(str, AB.TOKENS)))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--configs', default=None)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.artifact = [f'{k}={ART}/{args.model}_{k}' for k in ('fo6', 'tc_16x64', 'tc_8x64')]
    args.after_root = str(KO / 'build_U1')
    B.require_idle()
    table = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    kernels = dict(stock_ko=KernelSet('stock_ko', build_root=KO / 'build_7', table=table),
                   c16=KernelSet('nodisp_ko', build_root=KO / 'build_C3k', table=table),
                   k16_freq=KernelSet('mixed_ko', build_root=KO / 'build_7freq', table=table),
                   k16_uni=KernelSet('mixed_ko', build_root=KO / 'build_U1', table=table),
                   c8=KernelSet('nodisp_wB_ko', build_root=KO / 'build_P5', table=table),
                   k8_freq=KernelSet('mixed_wB_ko', build_root=KO / 'build_P3freq', table=table),
                   k8_uni=KernelSet('mixed_wB_ko', build_root=KO / 'build_U1', table=table),
                   k8_unip=KernelSet('mixed_wB_ko', build_root=KO / 'build_U1p', table=table),
                   k8_pipe=KernelSet('mixed_wB_ko', build_root=KO / 'build_U0p', table=table),
                   k8_hyb=KernelSet('mixed_wB_ko', build_root=KO / 'build_U8h', table=table))
    for name, want in DEFINES.items():
        for k in kernels[name].kernels.values():
            got = k.manifest.get('extra_defines') or {}
            assert got == want, (name, k.cfg.name, got)
            assert str(k.manifest['blob_gen'].get('TAG0')) == '0', (name, k.cfg.name)
    for w, k in kernels['k8_hyb'].kernels.items():
        want = DEFINES['k8_pipe'] if w == 128 else DEFINES['k8_unip']
        assert (k.manifest.get('extra_defines') or {}) == want, ('k8_hyb', w, k.manifest.get('extra_defines'))
    assert kernels['k16_freq'].schedules == kernels['k16_uni'].schedules
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        AB.run(args, tel, CONFIGS=CONFIGS, kernels=kernels, pairs=PAIRS)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    res['suite'] = 'kernel-opt U (uniform-branch dispatch), exploratory per-T timing (not registered)'
    B.write(args.out, res)


if __name__ == '__main__':
    main()

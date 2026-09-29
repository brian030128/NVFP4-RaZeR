#!/usr/bin/env python3
"""GEMM and activation-quantizer kernel times per text-Linear shape (step 06), one process per model.

    PAPER_PYTHON experiments/paper/bench_gemm.py --model llama8b --artifact fo6=ART --artifact nvfp4=ART \
        --artifact tc_8x64=ART --artifact tc_16x64=ART --artifact tc_256x64=ART --tokens 128,2048 --out JSON

**Configurations.** Each timed call is one NativeLinear forward (the deployment path: the per-token activation
quantizer, then the FP4 GEMM with its fused epilogue). Per call, CUPTI gives each kernel's device time: the median of
--iters calls, after 5 warm-ups (sm120/bench/common.kernel_times).

**Order (results/paper/PROTOCOL.md, deviation 1).** At each (projection, T), every configuration is measured in each
of --rounds rounds (default 3). Round r starts the configuration list at position r * len / rounds (rotated), so each
configuration is measured early, mid and late. The reported time is the median of the per-round medians. Measured
back-to-back in one fixed order, the same kernel ran up to 12 % slower when measured second (a clock or power state
that grows with the GEMM's load).

| configuration | kernel | weights | activations |
|---|---|---|---|
| stock_wA | 'auto_stock', width from the tile table | FourOverSix artifact | four_over_six_rows |
| stock_wA_nvfp4 | the same stock GEMM | NVFP4 artifact | nvfp4_rows |
| stock_wB | stock_wB (the 8x64 same-placement reference) | FourOverSix artifact | four_over_six_rows |
| mixed_16x64 | 'auto' (n16k64_wA), width from the tile table | TM-OPT+TC 16x64 artifact | four_over_six_rows |
| mixed_256x64 | 'auto' (n16k64_wA), 16x64 granules | TM-OPT+TC 256x64 artifact | four_over_six_rows |
| n8k64_wB | n8k64_wB (its single build) | TM-OPT+TC 8x64 artifact | four_over_six_rows |

- **FourOverSix and NVFP4 share the stock GEMM.** They differ in the weight values and the activation quantizer's
  kernel (quant_rows_kernel<1> / <0>). stock_wA_nvfp4 times that quantizer and repeats the stock GEMM.
- **The weights (the real map tags):** per projection, from the module with the most E0M3 tiles in the artifact, as
  the tile table's tuning did. For E2M1-only artifacts, the projection's first module.
- **Per-forward sums:** each projection's time times its number of modules. They cover the quantized text Linears
  only; the head and the rest of the model are BF16 in every policy. A quantizer launch is shared when consecutive
  projections read the same input: the q/k/v and gate/up projections in NativeLinear's quantization cache. Step 07
  applies the reuse counts measured in step 05. Without them, it states that no reuse is assumed (an upper bound).
- **The activations** are seeded random BF16 [T, in]; T is the number of tokens (batch x prompt).
"""
import argparse
import dataclasses
import json
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402  (sm120/bench/common.py)
from mixfp4_sm120 import artifact as A  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import NativeLinear  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

# configuration -> (artifact kind, kernel, activation quantizer)
CONFIGS = OrderedDict(stock_wA=('fo6', 'stock', 'four_over_six_rows'), stock_wA_nvfp4=('nvfp4', 'stock', 'nvfp4_rows'),
                      stock_wB=('fo6', 'stock_wB', 'four_over_six_rows'),
                      mixed_16x64=('tc_16x64', 'mixed', 'four_over_six_rows'),
                      mixed_256x64=('tc_256x64', 'mixed', 'four_over_six_rows'),
                      n8k64_wB=('tc_8x64', 'n8k64_wB', 'four_over_six_rows'))


def classify(name):
    n = name.lower()
    if 'quant_rows_kernel' in n:
        return 'quant'
    if 'cutlass' in n or 'device_kernel' in n or 'gemm' in n:
        return 'gemm'
    return 'other'


def representatives(meta, weights):
    """{projection: (module name, E0M3 tiles, module count, shape)}: the module with the most E0M3 tiles."""
    out = OrderedDict()
    for m in meta['modules']:
        proj = m['name'].rsplit('.', 1)[-1]
        name, tiles, count, shape = out.get(proj, (None, -1, 0, None))
        if m['e0m3_tiles'] > tiles:
            name, tiles = m['name'], m['e0m3_tiles']
        out[proj] = (name, tiles, count + 1, tuple(weights[m['name']].shape))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--tokens', default='128,256,512,1024,2048,4096,8192')
    ap.add_argument('--projections', default=None, help='comma-separated subset (default: every text projection)')
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--rounds', type=int, default=3, help='rounds per (projection, T), in rotated order')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    arts = dict(spec.split('=', 1) for spec in args.artifact)
    kernels = dict(stock=KernelSet('stock'), mixed=KernelSet('mixed'), stock_wB=Kernel.load('stock_wB'),
                   n8k64_wB=Kernel.load('n8k64_wB'))
    res = dict(status='running', gpu=B.gpu_info(), model=args.model, artifacts=arts, iters=args.iters,
               protocol=dict(rounds=args.rounds, order='rotated: round r starts at position r * len / rounds',
                             value='median over rounds of the per-round CUPTI median of --iters calls'),
               kernels={k: (v.sha256 if isinstance(v, Kernel) else v.describe()) for k, v in kernels.items()},
               configs={c: dict(artifact=a, kernel=k, activation_quantizer=q) for c, (a, k, q) in CONFIGS.items() if a in arts},
               projections={}, rows=[])
    reps = {}
    for kind, path in arts.items():
        meta, weights = A.load(path, device='cpu')
        sel = representatives(meta, weights)
        reps[kind] = {proj: weights[name] for proj, (name, *_) in sel.items()}
        for proj, (name, tiles, count, shape) in sel.items():
            p = res['projections'].setdefault(proj, dict(shape=list(shape), modules=count, weights={}))
            assert p['shape'] == list(shape) and p['modules'] == count, (kind, proj)
            p['weights'][kind] = dict(module=name, e0m3_tiles=tiles, tiles_in_module=None if not meta['type_block'] else
                                      -(-shape[0] // meta['type_block'][0]) * (shape[1] // meta['type_block'][1]),
                                      type_block=meta['type_block'], artifact_weights_sha256=meta['weights_sha256'])
        del weights
    projs = list(res['projections'])
    if args.projections:
        projs = [p for p in projs if p in args.projections.split(',')]
    tokens = [int(t) for t in args.tokens.split(',')]
    for proj in projs:
        n, k = res['projections'][proj]['shape']
        for t in tokens:
            x = torch.randn(t, k, generator=torch.Generator('cpu').manual_seed(n + k + t)).to('cuda', torch.bfloat16)
            cfgs = [c for c, (kind, _, _) in CONFIGS.items() if kind in reps]
            per = {c: [] for c in cfgs}
            for r in range(args.rounds):
                shift = (r * len(cfgs) // args.rounds) % len(cfgs)
                for pos, cfg in enumerate(cfgs[shift:] + cfgs[:shift]):
                    kind, kname, act = CONFIGS[cfg]
                    pw = reps[kind][proj]
                    pw = dataclasses.replace(pw, packed=pw.packed.cuda(), scales=pw.scales.cuda(),
                                             bias=None if pw.bias is None else pw.bias.cuda())
                    lin = NativeLinear(pw, kernels[kname], act, name=f'{proj}@{cfg}')
                    lin.share_input = False          # quantize on every call, as the first projection of a group does
                    times = B.kernel_times(lambda: lin(x), iters=args.iters)
                    by = {}
                    for name, v in times.items():
                        c = classify(name)
                        by[c] = by.get(c, 0.0) + v['us']
                    per[cfg].append(dict(round=r, position=pos, gemm_us=by.get('gemm'), quant_us=by.get('quant'),
                                         other_us=by.get('other', 0.0),
                                         kernels={name: round(v['us'], 3) for name, v in times.items()}))
                    del lin
            for cfg in cfgs:
                kind, kname, act = CONFIGS[cfg]
                kern = kernels[kname]
                med = lambda key: statistics.median(v[key] for v in per[cfg]) if all(v[key] is not None for v in per[cfg]) else None  # noqa: E731
                res['rows'].append(dict(proj=proj, out=n, inp=k, tokens=t, config=cfg, kernel=kname, act=act,
                                        width=kern.width(n, k, t) if isinstance(kern, KernelSet) else None,
                                        gemm_us=med('gemm_us'), quant_us=med('quant_us'), other_us=med('other_us'),
                                        rounds=[{k2: v[k2] for k2 in ('round', 'position', 'gemm_us', 'quant_us')} for v in per[cfg]],
                                        kernels=per[cfg][-1]['kernels']))
            print(f"{args.model} {proj} T={t} " + ' '.join(f"{r['config']}={r['gemm_us']:.1f}+{r['quant_us']:.1f}us"
                                                           for r in res['rows'] if r['proj'] == proj and r['tokens'] == t), flush=True)
            B.write(args.out, res)
    res['kernel_sets_after'] = {k: v.describe() for k, v in kernels.items() if isinstance(v, KernelSet)}
    res['gpu_end'] = B.gpu_info()
    res['status'] = 'complete'
    B.write(args.out, res)


if __name__ == '__main__':
    main()

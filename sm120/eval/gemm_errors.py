#!/usr/bin/env python3
"""Raw operator-level error numbers (checklist section 6), saved as JSON.

Per (model shape, token count, tag pattern, weight distribution), on synthetic LLM-like operands:
    native_vs_exact    NativeLinear output vs the FP64 product of the exactly decoded operands
                       -> native arithmetic error (FP32 tensor-core accumulation + one bf16 rounding)
    fake_vs_exact      fake-quant F.linear (BF16 dequantized operands) vs the same FP64 product
                       -> the fake path's own rounding
    quant_error        fake-quant output vs the unquantized BF16 layer -> quantization error
    max_elem_ratio     max over elements of |native - exact| / (2^-8 |exact| + 2^-20 sum|a b|),
                       the tolerance the kernel tests assert (must be <= 1)
All as relative Frobenius norms unless named otherwise.

    python sm120/eval/gemm_errors.py --out sm120/results/errors/gemm_errors_rtx5090.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from common import MODEL_SHAPES, gpu_info  # noqa: E402  (bench/common.py)

from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.artifact import fake_quant_weight, pack_module  # noqa: E402
from mixfp4_sm120.linear import NativeLinear  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


def weight(n, k, dist, g):
    if dist == 'normal':
        w = torch.randn(n, k, generator=g) * 0.02
    else:   # heavy-tailed with outlier input channels
        w = torch.distributions.StudentT(torch.tensor(3.0)).sample((n, k)) * 0.01
        w[:, torch.randperm(k, generator=g)[:max(1, k // 1024)]] *= 30
    return w.cuda().bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='llama8b,qwen4b')
    ap.add_argument('--tokens', default='1,128,2048')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ks = KernelSet('mixed')
    rows = []
    shapes = {}
    for m in args.models.split(','):
        for proj, s in MODEL_SHAPES[m].items():
            shapes.setdefault(s, f'{m}.{proj}')
    for (n, k), name in shapes.items():
        for dist in ('normal', 'heavy'):
            g = torch.Generator(device='cpu').manual_seed(n + k)
            w = weight(n, k, dist, g)
            grid = (n // 16, k // 64)
            for pattern, mask in (('e2m1', torch.zeros(grid, dtype=torch.bool)), ('sparse0.1%', torch.rand(grid, generator=g) < 0.001),
                                  ('random50', torch.rand(grid, generator=g) < 0.5), ('all_e0m3', torch.ones(grid, dtype=torch.bool))):
                pw = pack_module(name, w, None, 'map', mask, (16, 64))
                nl = NativeLinear(pw, ks, 'four_over_six_rows', name)
                wd = N.decode_exact(N.unpack_nibbles(pw.packed), pw.scales, pw.global_scale)
                wf = fake_quant_weight(w, 'map', mask, (16, 64))
                for t in [int(v) for v in args.tokens.split(',')]:
                    x = torch.randn(t, k, generator=g)
                    x[:, torch.randperm(k, generator=g)[:max(1, k // 512)]] *= 40
                    x = x.cuda().bfloat16()
                    xn, xsb, xgs = N.quantize_act(x, 'four_over_six_rows')
                    xd = N.decode_exact(xn, xsb, xgs[:, None])
                    exact = xd @ wd.t()
                    absdot = xd.abs() @ wd.abs().t()
                    with torch.no_grad():
                        y = nl(x)
                        fake = F.linear(N.fake_quant_act_rows(x, 'four_over_six_rows'), wf)
                        ref_bf16 = F.linear(x, w)
                    bound = 2.0 ** -8 * exact.abs() + 2.0 ** -20 * absdot + 1e-30
                    rows.append(dict(shape=name, out=n, inp=k, tokens=t, dist=dist, pattern=pattern,
                                     e0m3_tiles=int(mask.sum()), native_vs_exact=rel(y, exact), fake_vs_exact=rel(fake, exact),
                                     native_vs_fake=rel(y, fake), quant_error=rel(fake, ref_bf16),
                                     max_elem_ratio=float(((y.double() - exact).abs() / bound).max())))
            print(name, dist, 'done', flush=True)
    worst = max(r['max_elem_ratio'] for r in rows)
    summary = dict(cases=len(rows), max_elem_ratio=worst, all_within_tolerance=worst <= 1.0,
                   native_vs_exact_max=max(r['native_vs_exact'] for r in rows),
                   fake_vs_exact_max=max(r['fake_vs_exact'] for r in rows),
                   native_better_than_fake=sum(r['native_vs_exact'] <= r['fake_vs_exact'] for r in rows),
                   quant_error_range=[min(r['quant_error'] for r in rows), max(r['quant_error'] for r in rows)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(gpu=gpu_info(), kernel_set=ks.describe(), summary=summary, rows=rows), indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

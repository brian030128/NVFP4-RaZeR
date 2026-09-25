#!/usr/bin/env python3
"""Layer-by-layer native vs fake-quant comparison on real activations (one window).

The model runs the fake-quant policy. For every scoped Linear, a pre-hook captures its raw input
x (before the fake activation quantizer) and a forward hook compares, on that same x:
    fake    = F.linear(quantize_rows(x), W_fake)          (the fake path's own output)
    native  = NativeLinear(artifact)(x)                   (fused quantizer + patched GEMM)
    ref     = FP64 product of the exactly decoded native operands
and records rel. Frobenius errors native-vs-ref (native arithmetic), fake-vs-ref (fake-quant BF16
rounding), native-vs-fake, plus a bitwise check that the fused quantizer reproduced the reference
activation codes. It separates kernel/packing bugs (a layer far above ~1e-3) from the compounding
of tiny per-layer rounding differences through the network.

    python sm120/eval/layerwise.py --model qwen4b --map sm120/maps/qwen4b_seed0_n16_k3.mixfp4map \
        --artifact sm120/artifacts/qwen4b_n16_k3 --out sm120/results/layerwise/qwen4b_n16_k3.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from mixfp4_sm120 import artifact as A  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120 import quant_act as QA  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import NativeLinear  # noqa: E402


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(C.MODELS))
    ap.add_argument('--map', required=True)
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--kernel', default='n16k64_wA')
    ap.add_argument('--domain', default='wiki')
    ap.add_argument('--window', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = C.load_tokenizer(args.model)
    wins, _, _ = C.windows(tok, args.model, args.domain)
    ids = wins[args.window]
    model = C.load_model(args.model)
    mods = C.scope(model, args.model)
    header, masks, digest = C.read_map_for(args.model, args.map, mods)
    meta, weights = A.load(args.artifact)
    if meta['map']['sha256'] != digest:
        raise SystemExit('artifact was exported from a different map')
    fq = C.FakeQuant(mods)
    fq.install('map', masks, tuple(header['type_block']))
    kern = Kernel.load(args.kernel)
    native = {n: NativeLinear(weights[n], kern, meta['activation_quantizer'], name=n) for n in mods}
    rows = []
    raw = {}

    def pre(name):
        def f(mod, inp):
            raw[name] = inp[0].detach()
        return f

    def post(name):
        @torch.no_grad()
        def f(mod, inp, out):
            x = raw.pop(name)
            x2 = x.reshape(-1, x.shape[-1])
            y_nat = native[name](x)
            xnib, xsb, xgs = N.quantize_act(x2, 'four_over_six_rows')
            p, sf, gs = QA.quantize(x2, 'four_over_six_rows')
            codes_equal = torch.equal(p, N.pack_nibbles(xnib)) and torch.equal(gs, xgs)
            pw = weights[name]
            ref = N.decode_exact(xnib, xsb, xgs[:, None]) @ N.decode_exact(N.unpack_nibbles(pw.packed), pw.scales, pw.global_scale).t()
            if mod.bias is not None:
                ref = ref + mod.bias.double()
            fake = out.reshape(-1, out.shape[-1])
            yn = y_nat.reshape(-1, y_nat.shape[-1])
            rows.append(dict(name=name, shape=list(x2.shape) + [yn.shape[-1]], e0m3_tiles=pw.e0m3_tiles,
                             act_codes_bitwise=codes_equal, native_vs_ref=rel(yn, ref), fake_vs_ref=rel(fake, ref),
                             native_vs_fake=rel(yn, fake), max_abs_native_vs_fake=float((yn.float() - fake.float()).abs().max()),
                             ref_absmax=float(ref.abs().max())))
            del ref
        return f

    hs = []
    for n, m in mods.items():
        hs.append(m.register_forward_pre_hook(pre(n), prepend=True))
        hs.append(m.register_forward_hook(post(n)))
    with torch.no_grad():
        model(input_ids=ids.cuda(), use_cache=False)
    for h in hs:
        h.remove()
    worst = sorted(rows, key=lambda r: -r['native_vs_ref'])[:5]
    summary = dict(layers=len(rows), all_act_codes_bitwise=all(r['act_codes_bitwise'] for r in rows),
                   native_vs_ref_max=max(r['native_vs_ref'] for r in rows),
                   native_vs_ref_mean=sum(r['native_vs_ref'] for r in rows) / len(rows),
                   fake_vs_ref_max=max(r['fake_vs_ref'] for r in rows),
                   fake_vs_ref_mean=sum(r['fake_vs_ref'] for r in rows) / len(rows),
                   native_vs_fake_max=max(r['native_vs_fake'] for r in rows),
                   layers_native_worse_than_fake=sum(r['native_vs_ref'] > r['fake_vs_ref'] for r in rows),
                   worst_native=worst)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(model=args.model, map_sha256=digest, artifact=meta['weights_sha256'],
                                              domain=args.domain, window=args.window, summary=summary, layers=rows), indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

"""Exploratory (before any registration of kernel-opt #4, disclosed, not a result): the 64 x 64 epilogue tile at width
64 (CTA tile 128 x 64), where it costs a mainloop stage (6 -> 5), for the mixed 16x64 family (default dispatch) and
stock_wA alike: n16k64_wA_n64 vs n16k64_wA_n64_e64 and stock_wA_n64 vs stock_wA_n64_e64, all from build_4 (the #4
sources), at (shape, T) cells where the current table runs width 64. Timing as quick_E.py: weights cold by rotation +
512 MiB flush, activations quantized after the flush, CUPTI GEMM time, 3 rotated rounds x 30. Outputs are checked
bitwise against the auto-epilogue build first.
"""
import math
import statistics
import sys

sys.path.insert(0, '/home/dev/NVFP4-RaZeR/sm120')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import mixfp4_sm120  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402
assert mixfp4_sm120.__file__.startswith('/home/dev/NVFP4-RaZeR/sm120/')

B4 = '/home/dev/n16k64_campaign/kernel_opt/build_4'
FLUSH = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
L2 = torch.cuda.get_device_properties(0).L2_cache_size
_, m16l, _ = mapio.read_map('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_16x64.mixfp4map')
_, m16q, _ = mapio.read_map('/home/dev/n16k64_campaign/paper/artifacts/qwen27b_tc_16x64.mixfp4map')


def operands(n, k, t, mask, seed):
    g = torch.Generator('cpu').manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    kind = 'map' if mask is not None else 'four_over_six'
    wn, wsb, gsw = N.quantize_weight(w, kind, mask, (16, 64) if mask is not None else None)
    return dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw), x=x)


def cold(kern, op, n, k, t, iters=30, warmup=3):
    copies = [(op['wp'].clone(), op['wsf'].clone()) for _ in range(math.ceil(4 * L2 / (op['wp'].numel() + op['wsf'].numel())) + 1)]

    def one(i):
        FLUSH.sum()
        torch.cuda.synchronize()
        xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
        torch.cuda.synchronize()
        wp, wsf = copies[i % len(copies)]
        kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False)
        torch.cuda.synchronize()
    for i in range(warmup):
        one(i)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            one(warmup + i)
    d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    assert len(d) == iters, len(d)
    return statistics.median(d)


def out(kern, op, n, k, t):
    xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
    return kern.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False).clone()


# (label, n, k, mask, T list) at cells where the current table runs width 64 for both families
CELLS = [('llama gate/up', 14336, 4096, m16l['model.layers.0.mlp.gate_proj'], (64,)),
         ('llama q/o', 4096, 4096, m16l['model.layers.0.self_attn.o_proj'], (256,)),
         ('llama down', 4096, 14336, m16l['model.layers.0.mlp.down_proj'], (256, 1024)),
         ('llama k/v', 1024, 4096, m16l['model.layers.0.self_attn.k_proj'], (1024,)),
         ('qwen gate/up', 17408, 5120, m16q['model.language_model.layers.3.mlp.gate_proj'], (64,)),
         ('qwen down', 5120, 17408, m16q['model.language_model.layers.3.mlp.down_proj'], (128, 256))]
print('width 64: auto epilogue (6 stages) vs 64x64 (5 stages), us', flush=True)
for label, n, k, mask, ts in CELLS:
    for t in ts:
        line = []
        for fam, prefix in (('mixed', 'n16k64_wA_n64'), ('stock', 'stock_wA_n64')):
            op = operands(n, k, t, mask if fam == 'mixed' else None, n + k + t)
            ks = {e: Kernel.load(prefix + e, build_root=B4) for e in ('', '_e64')}
            ref = out(ks[''], op, n, k, t)
            assert torch.equal(out(ks['_e64'], op, n, k, t).view(torch.int16), ref.view(torch.int16)), (label, t, fam)
            res = {e: [] for e in ks}
            names = list(ks)
            for r in range(3):
                for e in (names if r % 2 == 0 else names[::-1]):
                    res[e].append(cold(ks[e], op, n, k, t))
            med = {e: statistics.median(v) for e, v in res.items()}
            line.append(f"{fam} auto {med['']:7.2f} e64 {med['_e64']:7.2f} ({100 * (med['_e64'] / med[''] - 1):+5.2f} %) "
                        f"rounds auto {[round(v, 2) for v in res['']]} e64 {[round(v, 2) for v in res['_e64']]}")
        print(f'{label:13s} T={t:5d}: ' + ' | '.join(line), flush=True)

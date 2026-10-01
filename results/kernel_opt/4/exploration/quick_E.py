"""Exploratory (before any registration of kernel-opt #4, disclosed, not a result): tile-scheduler raster order / swizzle
and the epilogue tile, for the mixed 16x64 family (default dispatch) and stock_wA alike. Builds from worktree wtE
(build_E: the same device code as sm120/build plus sm120_set_schedule; the e* builds change the epilogue tile).
Timing: M1's condition (weights cold by rotation + 512 MiB flush, activations quantized after the flush), CUPTI GEMM time,
3 rounds x 30. Every variant's output is checked bitwise against its family's default first.
Part 1: raster {heuristic, along M, along N} x swizzle {1, 2, 4, 8} at the 'mixed' / 'stock' table widths.
Part 2: epilogue tile 128x32 (auto) vs 128x64 / 64x32 / 64x64, 128-wide builds.
"""
import math
import statistics
import sys

sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtE/sm120')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import mixfp4_sm120  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402
assert '/wtE/' in mixfp4_sm120.__file__

BE, PAPER = '/home/dev/n16k64_campaign/kernel_opt/build_E', '/home/dev/NVFP4-RaZeR/sm120/build'
NAMES = {16: '_n16', 32: '_n32', 64: '_n64', 128: ''}
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


SHAPES = [('llama q/o', 4096, 4096, m16l['model.layers.0.self_attn.o_proj']),
          ('llama gate/up', 14336, 4096, m16l['model.layers.0.mlp.gate_proj']),
          ('llama down', 4096, 14336, m16l['model.layers.0.mlp.down_proj']),
          ('qwen gate/up', 17408, 5120, m16q['model.language_model.layers.3.mlp.gate_proj']),
          ('qwen down', 5120, 17408, m16q['model.language_model.layers.3.mlp.down_proj'])]
COMBOS = [(r, s) for r in (0, 1, 2) for s in (1, 2, 4, 8)]
print('part 1: raster / swizzle (us; default = heuristic, 1)', flush=True)
tables = {f: KernelSet(f, build_root=PAPER) for f in ('mixed', 'stock')}
for label, n, k, mask in SHAPES:
    for t in (128, 512, 2048):
        line = []
        for fam, prefix in (('mixed', 'n16k64_wA'), ('stock', 'stock_wA')):
            wd = tables[fam].width(n, k, t)
            kern = Kernel.load(prefix + NAMES[wd], build_root=BE)
            op = operands(n, k, t, mask if fam == 'mixed' else None, n + k + t)
            kern.set_schedule(0, 1)
            ref = out(kern, op, n, k, t)
            res = {c: [] for c in COMBOS}
            for c in COMBOS:
                kern.set_schedule(*c)
                y = out(kern, op, n, k, t)
                assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), (label, t, fam, c)
            for r in range(3):
                order = COMBOS[r * 4:] + COMBOS[:r * 4]
                for c in order:
                    kern.set_schedule(*c)
                    res[c].append(cold(kern, op, n, k, t))
            kern.set_schedule(0, 1)
            med = {c: statistics.median(v) for c, v in res.items()}
            best = min(med, key=med.get)
            line.append(f"{fam} w{wd:3d} default {med[(0, 1)]:7.2f} best r{best[0]}s{best[1]} {med[best]:7.2f} "
                        f"({100 * (med[best] / med[(0, 1)] - 1):+5.2f} %) [along M s1 {100 * (med[(1, 1)] / med[(0, 1)] - 1):+5.2f}, "
                        f"along N s1 {100 * (med[(2, 1)] / med[(0, 1)] - 1):+5.2f}]")
        print(f'{label:13s} T={t:5d}: ' + ' | '.join(line), flush=True)

print('part 2: epilogue tile, 128-wide builds (us; auto = 128x32)', flush=True)
for label, n, k, mask in SHAPES[:3]:
    for t in (256, 1024, 4096):
        line = []
        for fam, prefix in (('mixed', 'n16k64_wA'), ('stock', 'stock_wA')):
            op = operands(n, k, t, mask if fam == 'mixed' else None, n + k + t)
            ks = {e: Kernel.load(prefix + ('' if e == 'auto' else f'_e{e}'), build_root=BE) for e in ('auto', '128x64', '64x32', '64x64')}
            ref = out(ks['auto'], op, n, k, t)
            for e, kern in ks.items():
                y = out(kern, op, n, k, t)
                assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), (label, t, fam, e)
            res = {e: [] for e in ks}
            names = list(ks)
            for r in range(3):
                for e in names[r:] + names[:r]:
                    res[e].append(cold(ks[e], op, n, k, t))
            med = {e: statistics.median(v) for e, v in res.items()}
            line.append(f"{fam} auto {med['auto']:7.2f} " + ' '.join(f"{e} {100 * (med[e] / med['auto'] - 1):+5.2f}%" for e in names[1:]))
        print(f'{label:13s} T={t:5d}: ' + ' | '.join(line), flush=True)

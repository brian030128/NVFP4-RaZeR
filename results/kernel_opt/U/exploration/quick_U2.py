"""Exploratory (disclosed): the 8x64 pipelined variants of the uniform-branch dispatch. As quick_U.py, 8x64 only, with
  unip     MIXFP4_PIPE_FLAGS=1 + MIXFP4_UNIFORM_DISPATCH=1 (build_U1p): the next index computed mid-k_tile, through redux
  pipe     MIXFP4_PIPE_FLAGS=1 alone (build_U0p): the control
The rest of quick_U.py's docstring: the uniform-branch dispatch (MIXFP4_UNIFORM_DISPATCH=1,
the index through redux.sync.or) against today's adopted builds, both families. Builds from worktree wtU (build_U1, with
#2's dispatch as adopted) and the registered build directories:
  freq     the adopted path: 16x64 n16k64_wA_e64_t0 / _n16_t0 from build_7freq; 8x64 n8k64_wB_t0 / _m16_t0 from build_P3freq
  uni      the same configurations from build_U1 (MIXFP4_DISPATCH_FREQ=1 + MIXFP4_UNIFORM_DISPATCH=1)
  ceiling  the no-dispatch ceiling: n16k64_wA_nodisp_e64_t0 / _n16_t0 (build_C3k); n8k64_wB_nodisp_t0 / _m16_nodisp_t0 (build_P5)
  stock    stock_wA_e64 / stock_wA_n16 (build_7): stock_ko's builds at these widths (FourOverSix weights)
Maps: all-E2M1; the typical module (lower median of the E0M3 count) of the projection in Llama-3.1-8B's TC map of the family;
all-E0M3. Every uni output is checked bitwise against freq's first. Timing: M1's condition (cold weights by rotation +
512 MiB flush, activations quantized after the flush), CUPTI, 3 rotated rounds x 30; and back-to-back warm calls,
3 rounds x 50. Default scheduler setting (0, 1) for every kernel.
"""
import math
import statistics
import sys

sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtU/sm120')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import mixfp4_sm120  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402
assert '/wtU/' in mixfp4_sm120.__file__

KO = '/home/dev/n16k64_campaign/kernel_opt'
ART = '/home/dev/n16k64_campaign/paper/artifacts'
FLUSH = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
L2 = torch.cuda.get_device_properties(0).L2_cache_size
PROJ = {(4096, 4096): 'o_proj', (14336, 4096): 'gate_proj', (4096, 14336): 'down_proj'}
MAPS = {fam: mapio.read_map(f'{ART}/llama8b_tc_{fam}.mixfp4map') for fam in ('16x64', '8x64')}


def typical(fam, proj):
    header, masks, _ = MAPS[fam]
    mods = [m for m in header['modules'] if m['name'].rsplit('.', 1)[-1] == proj]
    order = sorted(range(len(mods)), key=lambda i: (mods[i]['selected'], i))
    m = mods[order[(len(mods) - 1) // 2]]
    return m['name'], masks[m['name']]


def kernels(fam, width):
    if fam == '16x64':
        mixed = 'n16k64_wA_e64_t0' if width == 128 else f'n16k64_wA_n{width}_t0'
        ceil = 'n16k64_wA_nodisp_e64_t0' if width == 128 else f'n16k64_wA_nodisp_n{width}_t0'
        return dict(freq=Kernel.load(mixed, build_root=f'{KO}/build_7freq'), uni=Kernel.load(mixed, build_root=f'{KO}/build_U1'),
                    ceiling=Kernel.load(ceil, build_root=f'{KO}/build_C3k'),
                    stock=Kernel.load('stock_wA_e64' if width == 128 else f'stock_wA_n{width}', build_root=f'{KO}/build_7'))
    mixed = 'n8k64_wB_t0' if width == 128 else f'n8k64_wB_m{width}_t0'
    ceil = 'n8k64_wB_nodisp_t0' if width == 128 else f'n8k64_wB_m{width}_nodisp_t0'
    return dict(freq=Kernel.load(mixed, build_root=f'{KO}/build_P3freq'), uni=Kernel.load(mixed, build_root=f'{KO}/build_U1'),
                unip=Kernel.load(mixed, build_root=f'{KO}/build_U1p'), pipe=Kernel.load(mixed, build_root=f'{KO}/build_U0p'),
                ceiling=Kernel.load(ceil, build_root=f'{KO}/build_P5'),
                stock=Kernel.load('stock_wA_e64' if width == 128 else f'stock_wA_n{width}', build_root=f'{KO}/build_7'))


def operands(n, k, t, mask, tb, seed):
    g = torch.Generator('cpu').manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    kind = 'map' if mask is not None else 'four_over_six'
    wn, wsb, gsw = N.quantize_weight(w, kind, mask, tb if mask is not None else None)
    return dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw), x=x)


def gemm(kern, wp, wsf, op, n, k, t):
    xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
    if kern.weight_operand == 0:
        return kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False)
    return kern.gemm(xp, xsf, wp, wsf, t, n, k, scale_m=gs, scale_n_default=op['gsw'], check=False)


def times(prof, n_):
    d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    assert len(d) == n_, len(d)
    return statistics.median(d)


def cold(kern, op, n, k, t, iters=30, warmup=3):
    copies = [(op['wp'].clone(), op['wsf'].clone()) for _ in range(math.ceil(4 * L2 / (op['wp'].numel() + op['wsf'].numel())) + 1)]

    def one(i):
        FLUSH.sum()
        torch.cuda.synchronize()
        xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
        torch.cuda.synchronize()
        wp, wsf = copies[i % len(copies)]
        if kern.weight_operand == 0:
            kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False)
        else:
            kern.gemm(xp, xsf, wp, wsf, t, n, k, scale_m=gs, scale_n_default=op['gsw'], check=False)
        torch.cuda.synchronize()
    for i in range(warmup):
        one(i)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            one(warmup + i)
    return times(prof, iters)


def b2b(kern, op, n, k, t, iters=50, warmup=5):
    xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
    if kern.weight_operand == 0:
        f = lambda: kern.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False)  # noqa: E731
    else:
        f = lambda: kern.gemm(xp, xsf, op['wp'], op['wsf'], t, n, k, scale_m=gs, scale_n_default=op['gsw'], check=False)  # noqa: E731
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            f()
        torch.cuda.synchronize()
    return times(prof, iters)


CELLS = [(128, 4096, 4096, 4096), (128, 4096, 4096, 1024), (128, 14336, 4096, 2048), (128, 4096, 14336, 512),
         (16, 4096, 4096, 16), (16, 14336, 4096, 16), (16, 4096, 14336, 16)]
names = ['freq', 'uni', 'unip', 'pipe', 'ceiling', 'stock']
print('us; % vs freq (the adopted path, #2\'s dispatch)', flush=True)
for fam in ('8x64',):
    tb = (16, 64) if fam == '16x64' else (8, 64)
    for width, n, k, t in CELLS:
        ks = kernels(fam, width)
        mod, real = typical(fam, PROJ[(n, k)])
        for label in ('e2m1', 'real', 'e0m3'):
            grid = (n // tb[0], k // 64)
            mask = {'e2m1': torch.zeros(grid, dtype=torch.bool), 'e0m3': torch.ones(grid, dtype=torch.bool), 'real': real}[label]
            op = operands(n, k, t, mask, tb, n + k + t)
            so = operands(n, k, t, None, tb, n + k + t)
            ref = gemm(ks['freq'], op['wp'], op['wsf'], op, n, k, t).clone()
            for c in ('uni', 'unip', 'pipe'):
                y = gemm(ks[c], op['wp'], op['wsf'], op, n, k, t)
                assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), (fam, width, n, k, t, label, c)
            res = {m: {c: [] for c in names} for m in ('cold', 'b2b')}
            for r in range(3):
                order = names[r:] + names[:r]
                for c in order:
                    res['cold'][c].append(cold(ks[c], so if c in ('ceiling', 'stock') else op, n, k, t))
                for c in order:
                    res['b2b'][c].append(b2b(ks[c], so if c in ('ceiling', 'stock') else op, n, k, t))
            for mode in ('cold', 'b2b'):
                med = {c: statistics.median(v) for c, v in res[mode].items()}
                rng = {c: (min(v), max(v)) for c, v in res[mode].items()}
                sep = max(res[mode]['unip']) < min(res[mode]['freq'])
                cells = ' '.join(f"{c} {100 * (med[c] / med['freq'] - 1):+5.2f}%" for c in names[1:])
                print(f"{fam} w{width:3d} {n}x{k} T={t:5d} {label:4s} {mode:4s}: freq {med['freq']:8.2f} | {cells} | unip below freq "
                      f"in every round: {sep} | real module {mod.split('.')[2]}", flush=True)

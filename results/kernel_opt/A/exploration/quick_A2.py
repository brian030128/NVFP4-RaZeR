"""Exploratory (before any registration of kernel-opt A, disclosed, not a result): the arm subset {0,1,2,4,8} + fallback
at every width, against today's builds and #2's freq builds, on real 16x64-map shapes. Isolated CUPTI, 3 rotated rounds
x 15 single calls, warm weights. Llama-3.1-8B layer-0 q_proj / down_proj / gate_proj with their own 16x64 tags; the width
is the 'mixed' tile table's for (shape, T). Every output is checked bitwise against today's build first.
"""
import statistics
import sys
import time

sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtA2/sm120')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import mixfp4_sm120  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402
assert '/wtA2/' in mixfp4_sm120.__file__

PAPER, FREQ, A2 = '/home/dev/NVFP4-RaZeR/sm120/build', '/home/dev/n16k64_campaign/kernel_opt/build_freq', '/home/dev/n16k64_campaign/kernel_opt/build_A2'
NAMES = {16: '_n16', 32: '_n32', 64: '_n64', 128: ''}
D = KernelSet('mixed', build_root=PAPER)
_, masks, _ = mapio.read_map('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_16x64.mixfp4map')


def iso(fn, reps=15):
    out = []
    for _ in range(reps):
        time.sleep(0.05)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        out += [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    return statistics.median(out)


for layer, (n, k) in (('model.layers.0.self_attn.q_proj', (4096, 4096)), ('model.layers.0.mlp.down_proj', (4096, 14336)),
                      ('model.layers.0.mlp.gate_proj', (14336, 4096))):
    for t in (1, 16, 64, 256):
        g = torch.Generator('cpu').manual_seed(n + k + t)
        w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
        x = torch.randn(t, k, generator=g).cuda().bfloat16()
        xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
        wn, wsb, gsw = N.quantize_weight(w, 'map', masks[layer], (16, 64))
        wp, wsf, xp, xsf = N.pack_nibbles(wn), place(wsb, k), N.pack_nibbles(xn), place(xsb, k)
        wd = D.width(n, k, t)
        K = dict(default=Kernel.load('n16k64_wA' + NAMES[wd], build_root=PAPER),
                 freq=Kernel.load('n16k64_wA' + NAMES[wd], build_root=FREQ),
                 as6=Kernel.load('n16k64_wA_as6' + NAMES[wd], build_root=A2))
        fns = {kk: (lambda kern=kern: kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=float(gsw), scale_n=gsx, check=False))
               for kk, kern in K.items()}
        ref = fns['default']()
        for kk in ('freq', 'as6'):
            y = fns[kk]()
            torch.cuda.synchronize()
            assert torch.equal(ref.view(torch.int16), y.view(torch.int16)), (layer, t, kk)
        res = {kk: [] for kk in fns}
        names = list(fns)
        for r in range(3):
            sh = r * len(names) // 3
            for kk in names[sh:] + names[:sh]:
                fns[kk]()
                torch.cuda.synchronize()
                res[kk].append(iso(fns[kk]))
        med = {kk: statistics.median(v) for kk, v in res.items()}
        print(f"{layer.split('.')[-1]:10s} {n}x{k} T={t:4d} width {wd:3d}: default {med['default']:6.1f} us, "
              f"freq {100 * (med['freq'] / med['default'] - 1):+6.2f} %, as6 {100 * (med['as6'] / med['default'] - 1):+6.2f} % "
              f"(bitwise equal)", flush=True)

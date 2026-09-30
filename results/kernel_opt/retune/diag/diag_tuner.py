"""Diagnostic (disclosed, not a result): why tune_tiles --cold ranks widths differently from M1 at small T.
Stock and mixed (#2's builds) widths on Llama-3.1-8B shapes, T in {32, 64, 128}, isolated CUPTI GEMM time, weights cold
(rotation over copies + 512 MiB flush), 3 rounds x 30:
  A  tune_tiles.cold_us: flush, then the GEMM on activations quantized once before (cold after the flush)
  B  M1's order: flush, then the activation quantizer (fresh, L2-warm activations), then the GEMM
"""
import math
import statistics
import sys

sys.path.insert(0, '/home/dev/NVFP4-RaZeR/sm120')
sys.path.insert(0, '/home/dev/NVFP4-RaZeR/sm120/bench')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
import tune_tiles as TT  # noqa: E402
from kernel import operands  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

FREQ = '/home/dev/n16k64_campaign/kernel_opt/build_freq'
flush = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
l2 = torch.cuda.get_device_properties(0).L2_cache_size


def warm_act_us(kern, op, n, k, t, iters=30, warmup=3):
    copies = [(op['wp'].clone(), op['wsf'].clone()) for _ in range(math.ceil(4 * l2 / (op['wp'].numel() + op['wsf'].numel())) + 1)]
    x = op['x']
    def one(i):
        flush.sum()
        torch.cuda.synchronize()
        xp, xsf, gs = kern.quant_rows(x, 'four_over_six_rows')
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


for fam, root in (('stock', None), ('mixed', FREQ)):
    ks = KernelSet(fam, table={}, build_root=root)
    for (n, k) in ((4096, 4096), (4096, 14336)):
        for t in (32, 64, 128):
            op = operands(n, k, t, None, None, seed=n + k + t)
            res = {}
            for w, kern in ks.kernels.items():
                if w == 128:
                    continue
                launch = lambda wp, wsf, kern=kern: kern.gemm(wp, wsf, op['xp'], op['xsf'], n, t, k, scale_m_default=op['gsw'],
                                                             scale_n=op['gsx'], check=False)
                a = statistics.median(TT.cold_us(launch, op['wp'], op['wsf'], iters=30) for _ in range(3))
                b = statistics.median(warm_act_us(kern, op, n, k, t) for _ in range(3))
                res[w] = (a, b)
            best_a = min(res, key=lambda w: res[w][0])
            best_b = min(res, key=lambda w: res[w][1])
            print(f"{fam:5s} {n}x{k} T={t:3d}: " + ' | '.join(f'w{w}: tuner {a:.2f} / M1-order {b:.2f}' for w, (a, b) in res.items())
                  + f'  -> best tuner w{best_a}, M1-order w{best_b}', flush=True)

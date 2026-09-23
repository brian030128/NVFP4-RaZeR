"""Where does one linear layer's time go: fake quant vs the native-kernel harness path.

Per Llama-3.1-8B shape (T = 2048 tokens), median over iterations of the GPU time of each segment
(CUDA events, each segment synchronized separately) plus the end-to-end CPU wall time of one call:

  fake    quantize_rows(x)              per-token FourOverSix, dequantized bf16
          F.linear(x_dq, w_dq)          cuBLAS BF16 GEMM on pre-dequantized weights
  native  act_four_over_six_rows(x)     same arithmetic, returns codes/scales/global scale
          encode + pack                 e2m1_nibbles (+ validity check, a host sync), pack_nibbles,
                                        scale_bytes (+ exactness check, a host sync)
          place_scales                  zero buffer + scatter into the kernel's SF layout
          kernel.gemm                   ctypes call + CUTLASS can_implement/initialize + launch
          epilogue                      D.float() * gs_x, transpose, bf16, reshape
"""
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/dev/NVFP4-RaZeR-n16k64/repro_local/realquant')
import rq  # noqa: E402
from campaign import quant as Q  # noqa: E402
from quantize.causal_four_over_six import quantize_rows  # noqa: E402

ITERS = 30


def gpu_ms(fn):
    """Median GPU time of fn() over ITERS runs, each bracketed by events and synchronized."""
    times = []
    for _ in range(ITERS + 5):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times[5:])


def wall_ms(fn):
    times = []
    for _ in range(ITERS + 5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(1e3 * (time.perf_counter() - t0))
    return statistics.median(times[5:])


@torch.no_grad()
def main():
    dev = 'cuda'
    torch.manual_seed(0)
    shapes = {'q_proj 4096x4096': (4096, 4096), 'k_proj 1024x4096': (1024, 4096),
              'up_proj 14336x4096': (14336, 4096), 'down_proj 4096x14336': (4096, 14336)}
    t = 2048
    for cfg in ('wt_as_A', 'b8x64'):
        kern = rq.Kernel(cfg)
        print(f'\n=== native kernel config {cfg} (weights on {"A" if kern.weight_operand == 0 else "B"}) ===')
        for label, (n, k) in shapes.items():
            w = (torch.randn(n, k, device=dev) * 0.02).bfloat16()
            x = torch.randn(t, k, device=dev).bfloat16()
            w_fake = Q.four_over_six(w)
            pw, sbytes = rq.pack_weight(w, 'four_over_six')
            pw.sf_bytes = (kern.place_scales(sbytes, 0, n, t, k) if kern.weight_operand == 0
                           else kern.place_scales(sbytes, 1, t, n, k))
            lin = rq.RealLinear(kern, pw, None, 'four_over_six_rows', check_calls=0)
            xq = quantize_rows(x)
            code, scale, gs = rq.act_four_over_six_rows(x)
            nib = rq.e2m1_nibbles(code)
            packed = rq.pack_nibbles(nib)
            sb = rq.scale_bytes(scale)
            if kern.weight_operand == 0:
                sf = kern.place_scales(sb, 1, n, t, k)
                gemm = lambda: kern.gemm(pw.packed, pw.sf_bytes, packed, sf, n, t, k, pw.gs)
                place = lambda: kern.place_scales(sb, 1, n, t, k)
            else:
                sf = kern.place_scales(sb, 0, t, n, k)
                gemm = lambda: kern.gemm(packed, sf, pw.packed, pw.sf_bytes, t, n, k, pw.gs)
                place = lambda: kern.place_scales(sb, 0, t, n, k)
            d = gemm()

            def epilogue():
                if kern.weight_operand == 0:
                    y = (d.float() * gs[None, :]).t()
                else:
                    y = d.float() * gs[:, None]
                return y.to(torch.bfloat16).reshape(1, t, n)

            seg = dict(
                fake_act=gpu_ms(lambda: quantize_rows(x)),
                fake_gemm=gpu_ms(lambda: F.linear(xq, w_fake)),
                nat_act=gpu_ms(lambda: rq.act_four_over_six_rows(x)),
                nat_encode=gpu_ms(lambda: (rq.pack_nibbles(rq.e2m1_nibbles(code)), rq.scale_bytes(scale))),
                nat_place=gpu_ms(place),
                nat_gemm=gpu_ms(gemm),
                nat_epi=gpu_ms(epilogue),
            )
            fake_total = wall_ms(lambda: F.linear(quantize_rows(x), w_fake))
            nat_total = wall_ms(lambda: lin(x))
            flops = 2 * t * n * k
            print(f'{label:22s} fake: act {seg["fake_act"]:.3f} + gemm {seg["fake_gemm"]:.3f} ms '
                  f'-> call {fake_total:.3f} ms | native: act {seg["nat_act"]:.3f} + encode {seg["nat_encode"]:.3f} '
                  f'+ place {seg["nat_place"]:.3f} + gemm {seg["nat_gemm"]:.3f} + epi {seg["nat_epi"]:.3f} ms '
                  f'-> call {nat_total:.3f} ms | GEMM TFLOP/s: bf16 {flops / seg["fake_gemm"] / 1e9:.0f}, '
                  f'native {flops / seg["nat_gemm"] / 1e9:.0f}', flush=True)


if __name__ == '__main__':
    main()

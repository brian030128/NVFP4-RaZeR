"""
    Does clipping buy its MSE reduction with COHERENT error?

    MSE treats a scale block's 16 errors as if they were going to be summed in quadrature. They are
    not: the layer computes y_i = sum_j x_j W_ij, so the errors are summed with the input as weights
    and their SIGNS matter. Split the per-block error into the two parts that a correlated-input
    Hessian S = sigma^2 [(1-r) I + r 11^T] charges for separately:

        incoherent   sum_j dW_j^2            <- all MSE can see
        coherent     (sum_j dW_j)^2          <- invisible to MSE, and what "corr<r>" prices

    A clipping candidate pulls every clipped element the same way, so it should raise the coherent
    share even where it lowers MSE. If that shows up on real weights, then an MSE-selected clip is
    systematically overrated and "corr<r>" should pick differently -- which is the hypothesis this
    script exists to test before spending GPU hours on it.

    Usage:
        python analyze_coherent_error.py --model_name llama-2-7b --max_layers 8
        python analyze_coherent_error.py                    # synthetic
"""

import argparse

import torch

from quantize.quantizer import (
    CLIP_PRESETS,
    _quant_e0m3,
    _quant_e2m1,
    _tile_type_blocks,
)

E2M1_MAX, E0M3_MAX = 6.0, 7.0
FP8_MAX, FP8_MIN = 448.0, 2 ** (-9)


@torch.no_grad()
def best_candidate(w_tiled, block_max, quant_fn, grid_max, alphas):
    """Per scale block, the alpha with the lowest squared error. Returns the dequantized tensor."""
    best_dq, best_err = None, None
    for alpha in alphas:
        scale = (block_max * (alpha / grid_max)).clamp(max=FP8_MAX, min=FP8_MIN)
        scale = scale.to(torch.float8_e4m3fn).to(w_tiled.dtype)
        dq    = quant_fn(w_tiled, scale)
        err   = (dq - w_tiled).pow(2).sum(dim=-1, keepdim=True)
        if best_dq is None:
            best_dq, best_err = dq, err
        else:
            take    = err < best_err
            best_dq = torch.where(take, dq, best_dq)
            best_err = torch.where(take, err, best_err)
    return best_dq


@torch.no_grad()
def split_error(w_tiled, dq):
    """(incoherent, coherent) totals over all scale blocks."""
    d = dq - w_tiled
    return d.pow(2).sum().item(), d.sum(dim=-1).pow(2).sum().item()


def model_tensors(model_name: str, max_layers: int):
    import torch.nn as nn
    from transformers import AutoModelForCausalLM
    from utils import model2path

    model = AutoModelForCausalLM.from_pretrained(
        model2path[model_name], torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    out, count = [], 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "head" not in name:
            out.append((name, mod.weight.data.clone()))
            count += 1
            if count >= max_layers:
                break
    del model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--max_layers", type=int, default=8)
    ap.add_argument("--presets", type=lambda s: s.split(","),
                    default=["base", "e0", "e2", "e2x", "bothx", "wide"])
    args = ap.parse_args()

    if args.model_name:
        tensors = model_tensors(args.model_name, args.max_layers)
    else:
        torch.manual_seed(0)
        x = torch.randn(4096, 4096)
        x[:, ::37] *= 20.0
        tensors = [("synthetic", x.to(torch.bfloat16))]

    print(f"{'layer':<34} {'preset':<7} {'grid':<5} "
          f"{'incoherent':>12} {'coherent':>12} {'coh/incoh':>10} {'d incoh %':>10}")
    print("-" * 96)

    for name, w in tensors:
        w32   = w.reshape(-1, w.shape[-1]).to(torch.float32)
        gs    = (w32.abs().amax() / (E2M1_MAX * FP8_MAX)).clamp(min=torch.finfo(torch.float32).tiny)
        tiled, _  = _tile_type_blocks(w32 / gs, 1, 16, 16)
        block_max = tiled.abs().amax(dim=-1, keepdim=True)

        ref = {}
        for preset in args.presets:
            alphas = CLIP_PRESETS[preset]
            for grid, fn, gmax, key in (("e2m1", _quant_e2m1, E2M1_MAX, "e2m1"),
                                        ("e0m3", _quant_e0m3, E0M3_MAX, "e0m3")):
                dq = best_candidate(tiled, block_max, fn, gmax, alphas[key])
                inc, coh = split_error(tiled, dq)
                if preset == "base":
                    ref[grid] = inc
                d_inc = 100.0 * (inc / ref[grid] - 1.0)
                print(f"{name[:34]:<34} {preset:<7} {grid:<5} "
                      f"{inc:>12.4e} {coh:>12.4e} {coh / inc:>10.4f} {d_inc:>+10.2f}")
        print()

    print("coh/incoh is the ratio the `corr<r>` metric charges for: loss = incoh + r (coh - incoh).")
    print("A preset that lowers `d incoh %` while raising coh/incoh is buying MSE with coherence.")


if __name__ == "__main__":
    main()

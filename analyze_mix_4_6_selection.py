"""
    Why does mix_4_6 only beat nvfp4_4over6 at small type blocks?

    By construction mix_4_6 should never lose: for any type block it can elect E2M1, whose
    per-scale-block 4/6 search IS nvfp4_4over6, and it only elects E0M3 when that lowers the tile's
    summed error. So MSE(mix_4_6) <= MSE(4over6) must hold at EVERY type-block size. Yet perplexity
    gets worse than 4over6 as soon as the type block exceeds one scale block.

    This script checks the guarantee and then measures what the tile decision does to individual
    scale blocks, which is where the two facts are reconciled:

      * "overruled" blocks -- scale blocks whose individually-best element type differs from the one
        their tile elected. These exist only when the type block spans more than one scale block.
      * the error the tile decision ADDS to those blocks, versus the error it REMOVES elsewhere.
        The sum is negative (that is why E0M3 was elected) but the two sides can both be large.

    Usage:
        python analyze_mix_4_6_selection.py --model_name llama-2-7b --max_layers 6
        python analyze_mix_4_6_selection.py            # synthetic tensors
"""

import argparse

import torch

from quantize.quantizer import (
    _quant_e2m1,
    _tile_type_blocks,
    quant_mix_4_6,
    quant_nvfp4_4over6,
)
from quantize.utils import parse_type_block

E2M1_MAX, E2M1_ALT, E0M3_MAX = 6.0, 4.0, 7.0
FP8_MAX, FP8_MIN = 448.0, 2**(-9)
TYPE_BLOCKS = ["1x16", "16x16", "8x64", "16x64", "32x64", "32x128"]


@torch.no_grad()
def block_errors(w_fp, type_block, groupsize=16):
    """Per-scale-block squared error of the E2M1 (best of 4/6) and E0M3 candidates."""
    block_m, block_k = parse_type_block(type_block)
    x = w_fp.reshape(-1, w_fp.shape[-1]).to(torch.float32)
    if x.shape[-1] % block_k != 0:
        block_k = x.shape[-1]

    gscale   = (x.abs().amax() / (E2M1_MAX * FP8_MAX)).clamp(min=torch.finfo(torch.float32).tiny)
    tiled, _ = _tile_type_blocks(x / gscale, block_m, block_k, groupsize)
    bmax     = tiled.abs().amax(dim=-1, keepdim=True)

    err_e2m1 = None
    for qmax in (E2M1_MAX, E2M1_ALT):
        scale = (bmax / qmax).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn).to(tiled.dtype)
        err   = (_quant_e2m1(tiled, scale) - tiled).pow(2).sum(dim=-1)
        err_e2m1 = err if err_e2m1 is None else torch.minimum(err_e2m1, err)

    scale0   = (bmax / E0M3_MAX).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn).to(tiled.dtype)
    dq0      = (tiled / scale0).round().clamp(-E0M3_MAX, E0M3_MAX) * scale0
    err_e0m3 = (dq0 - tiled).pow(2).sum(dim=-1)

    # peakiness of each scale block: max / rms, the thing E2M1's non-uniform grid is good at
    rms  = tiled.pow(2).mean(dim=-1).sqrt()
    peak = bmax.squeeze(-1) / rms.clamp(min=1e-30)

    return err_e2m1, err_e0m3, peak, gscale


def analyze(name, w_fp):
    print(f"\n=================== {name} ===================")

    nmse_ref = None
    ref = quant_nvfp4_4over6(w_fp, groupsize=16)
    den = w_fp.float().pow(2).sum()
    nmse_ref = ((w_fp.float() - ref.float()).pow(2).sum() / den).item()
    print(f"nvfp4_4over6 NMSE = {nmse_ref:.6e}")
    print(f"\n{'type block':<12}{'mix NMSE':>13}{'vs 4over6':>12}{'E0M3 tiles':>12}"
          f"{'forced E0M3':>11}{'gain %':>9}{'churn/net':>11}")

    for tb in TYPE_BLOCKS:
        dq   = quant_mix_4_6(w_fp, groupsize=16, type_block=tb)
        nmse = ((w_fp.float() - dq.float()).pow(2).sum() / den).item()

        err_e2m1, err_e0m3, peak, _ = block_errors(w_fp, tb)

        # tile decision: sum each candidate over the scale blocks of its type block
        tile_e2m1 = err_e2m1.sum(dim=-1, keepdim=True)
        tile_e0m3 = err_e0m3.sum(dim=-1, keepdim=True)
        elect_e0m3 = tile_e0m3 < tile_e2m1                       # (num_tile, 1)

        # what each scale block would have picked for itself
        best_e0m3 = err_e0m3 < err_e2m1                          # (num_tile, num_block)
        chosen_e0m3 = elect_e0m3.expand_as(best_e0m3)

        err_chosen = torch.where(chosen_e0m3, err_e0m3, err_e2m1)

        # Two directions of overruling, which are NOT symmetric:
        #  forced_e0m3: block prefers E2M1 but its tile elected E0M3. This is the only direction
        #               that can make mix_4_6 worse than 4over6 on a given block.
        #  forced_e2m1: block prefers E0M3 but its tile elected E2M1. Harmless -- the block simply
        #               behaves exactly like 4over6.
        forced_e0m3 = chosen_e0m3 & ~best_e0m3
        forced_e2m1 = ~chosen_e0m3 & best_e0m3

        # error mix_4_6 ADDS relative to 4over6 (only possible on forced_e0m3 blocks) ...
        worse = (err_chosen - err_e2m1).clamp(min=0).sum().item()
        # ... and the error it REMOVES relative to 4over6
        better = (err_e2m1 - err_chosen).clamp(min=0).sum().item()

        gain_pct = 100.0 * (nmse_ref - nmse) / nmse_ref
        net      = better - worse
        churn    = (better + worse) / net if net > 0 else float("inf")

        print(f"{tb:<12}{nmse:>13.6e}{nmse - nmse_ref:>+12.3e}"
              f"{elect_e0m3.float().mean().item()*100:>11.1f}%"
              f"{forced_e0m3.float().mean().item()*100:>10.2f}%"
              f"{gain_pct:>9.3f}{churn:>11.1f}")

    print("  E0M3 tiles  = share of type blocks electing E0M3")
    print("  forced E0M3 = share of scale blocks that preferred E2M1 but were overruled onto E0M3")
    print("                (the only way mix_4_6 can be worse than 4over6 on a block)")
    print("  gain %      = NMSE reduction vs 4over6 (the guaranteed-nonnegative quantity)")
    print("  churn/net   = (error added + error removed) / net error removed, vs 4over6.")
    print("                High churn means mix_4_6 moves a lot of weights for almost no net gain.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_layers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.model_name is None:
        torch.manual_seed(0)
        w = torch.randn(4096, 4096)
        w[:, ::13] *= 18.0
        tensors = {"gaussian+outlier [4096,4096]": w}
    else:
        import json, os
        from transformers import AutoModelForCausalLM
        m2p = json.load(open(os.path.join(os.path.dirname(__file__), "model2path.json")))
        model = AutoModelForCausalLM.from_pretrained(
            m2p[args.model_name], torch_dtype=torch.bfloat16, device_map="cpu"
        )
        tensors, n = {}, 0
        for nm, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "head" not in nm:
                tensors[f"{nm} {tuple(mod.weight.shape)}"] = mod.weight.data.clone()
                n += 1
                if n >= args.max_layers:
                    break

    for nm, w in tensors.items():
        analyze(nm, w.to(torch.bfloat16).to(args.device))


if __name__ == "__main__":
    main()

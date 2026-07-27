"""
    What a Hadamard rotation does to the E2M1-vs-E0M3 decision.

    Rotating a scale block by a normalized Hadamard mixes its 16 values, which spreads a lone outlier
    over the whole block and drops the block maximum the 4-bit grid has to span. That is why it
    lowers the error. The more interesting consequence for THIS project is what it does to the
    decision: E2M1's log spacing only pays when a block has a heavy upper tail, and a rotated block
    does not. So rotation should push the choice toward the uniform grid, E0M3 -- and it is exactly
    the tiles that were electing E0M3 almost never (q_proj, k_proj, o_proj), i.e. the tiles where
    MixFP4 had degenerated to plain 4over6, that have the most to gain.

    Reports, per layer, with and without rotation:
      * E0M3 per-block  -- share of 16-element scale blocks that individually prefer E0M3
                           (what a 1x16 type block would choose)
      * E0M3 per-tile   -- share of type blocks that elect E0M3
      * straddling      -- share of type blocks containing both preferences, i.e. tiles that must
                           overrule some of their own scale blocks

    Usage:
        python analyze_rotation.py --model_name llama-2-7b --max_layers 7
        python analyze_rotation.py --type_block 32x128
"""

import argparse

import torch

from quantize.quantizer import (
    CLIP_PRESETS,
    _elect_e0m3,
    _quant_e0m3,
    _quant_e2m1,
    _rotate_chunks,
    _selection_loss,
    _tile_type_blocks,
)
from quantize.utils import parse_type_block

E2M1_MAX, E0M3_MAX = 6.0, 7.0
FP8_MAX, FP8_MIN = 448.0, 2 ** (-9)


@torch.no_grad()
def decision_stats(w, type_block, rotate_size, clip="base", elect="argmin", margin=0.0):
    block_m, block_k = parse_type_block(type_block)
    w2 = w.reshape(-1, w.shape[-1]).float()
    if rotate_size:
        w2 = _rotate_chunks(w2, rotate_size)
    gs = (w2.abs().amax() / (E2M1_MAX * FP8_MAX)).clamp(min=torch.finfo(torch.float32).tiny)
    tiled, _ = _tile_type_blocks(w2 / gs, block_m, block_k, 16)
    block_max = tiled.abs().amax(dim=-1, keepdim=True)

    def best(quant_fn, grid_max, alphas):
        best_err = None
        for alpha in alphas:
            scale = (block_max * (alpha / grid_max)).clamp(
                max=FP8_MAX, min=FP8_MIN
            ).to(torch.float8_e4m3fn).to(tiled.dtype)
            err = _selection_loss(tiled, quant_fn(tiled, scale), "mse")
            best_err = err if best_err is None else torch.minimum(best_err, err)
        return best_err

    alphas = CLIP_PRESETS[clip]
    gain = (best(_quant_e2m1, E2M1_MAX, alphas["e2m1"])
            - best(_quant_e0m3, E0M3_MAX, alphas["e0m3"]))

    return {
        "per_block": (gain > 0).float().mean().item(),
        "per_tile":  _elect_e0m3(gain, rule=elect, margin=margin).float().mean().item(),
        "straddle":  ((gain > 0).any(dim=1) & (gain <= 0).any(dim=1)).float().mean().item(),
        "nmse_gap":  gain.sum().item(),
    }


def model_tensors(model_name, max_layers):
    import torch.nn as nn
    from transformers import AutoModelForCausalLM
    from utils import model2path

    model = AutoModelForCausalLM.from_pretrained(
        model2path[model_name], torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "head" not in name:
            out.append((name, mod.weight.data.clone()))
            if len(out) >= max_layers:
                break
    del model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="llama-2-7b")
    ap.add_argument("--max_layers", type=int, default=7)
    ap.add_argument("--type_block", default="8x64")
    ap.add_argument("--rotate_sizes", type=lambda s: [int(v) for v in s.split(",")],
                    default=[0, 16, 64])
    ap.add_argument("--elect", default="argmin")
    ap.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args()

    print(f"type block {args.type_block}, election {args.elect}({args.margin})\n")
    print(f"{'layer':<28} {'rot':>5} {'E0M3/block':>11} {'E0M3/tile':>10} {'straddling':>11}")
    print("-" * 68)
    for name, w in model_tensors(args.model_name, args.max_layers):
        for size in args.rotate_sizes:
            s = decision_stats(w, args.type_block, size, elect=args.elect, margin=args.margin)
            tag = "none" if size == 0 else str(size)
            print(f"{name.split('.', 2)[-1][:28]:<28} {tag:>5} "
                  f"{s['per_block']:>10.1%} {s['per_tile']:>9.1%} {s['straddle']:>10.1%}")
        print()


if __name__ == "__main__":
    main()

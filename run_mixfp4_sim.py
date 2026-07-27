"""
    MixFP4 CPU simulation sweep.

    Fake-quantizes tensors on the CPU with MixFP4 for a list of type-block shapes and reports the
    quantization error against the NVFP4 / NVINT4(NVIF4) baselines. No GPU and no NVFP4 hardware
    support is required -- everything is emulated in FP32.

    Examples:
        # synthetic tensors
        python run_mixfp4_sim.py

        # real weights of a model listed in model2path.json
        python run_mixfp4_sim.py --model_name llama-2-7b --max_layers 4
"""

import argparse
import torch

from quantize.quantizer import (
    quant_mixfp4,
    quant_nvfp4,
    quant_nvif4,
)
from quantize.utils import parse_type_block


DEFAULT_TYPE_BLOCKS = ["1x16", "16x16", "256x16", "32x64", "32x128"]


def quant_error(w_fp, w_dq):
    """
        Relative quantization error: ||w - w_q||^2 / ||w||^2 , and the SQNR in dB.
    """
    w_fp  = w_fp.to(torch.float32)
    w_dq  = w_dq.to(torch.float32)
    noise = (w_fp - w_dq).pow(2).sum()
    signal = w_fp.pow(2).sum()
    nmse  = (noise / signal).item()
    sqnr  = float("inf") if nmse == 0 else -10.0 * torch.log10(noise / signal).item()
    return nmse, sqnr


@torch.no_grad()
def e0m3_fraction(w_fp, type_block, groupsize: int = 16):
    """
        Fraction of type blocks that select the E0M3 data type. Recomputed here (rather than
        returned by the quantizer) so that `quant_mixfp4` keeps the same signature as every other
        fake quantizer in this repo.
    """
    from quantize.quantizer import _tile_type_blocks

    block_m, block_k = parse_type_block(type_block)
    x = w_fp.reshape(-1, w_fp.shape[-1]).to(torch.float32)
    if x.shape[-1] % block_k != 0:
        block_k = x.shape[-1]

    global_scale = (x.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
    tiled, _     = _tile_type_blocks(x / global_scale, block_m, block_k, groupsize)
    block_max    = tiled.abs().amax(dim=-1, keepdim=True)

    scale_e2m1 = (block_max / 6.0).clamp(min=2**(-9), max=448.0).to(torch.float8_e4m3fn).to(tiled.dtype)
    scaled     = tiled / scale_e2m1
    exp        = torch.floor(torch.log2(torch.abs(scaled) + (scaled == 0).type(scaled.dtype))).clamp(min=0)
    man        = torch.sign(scaled / (2**exp) * 2) * torch.floor(torch.abs(scaled / (2**exp) * 2) + 0.5)
    dq_e2m1    = (man * (2**exp) / 2).clamp(-6.0, 6.0) * scale_e2m1

    scale_e0m3 = (block_max / 7.0).clamp(min=2**(-9), max=448.0).to(torch.float8_e4m3fn).to(tiled.dtype)
    dq_e0m3    = (tiled / scale_e0m3).round().clamp(-7.0, 7.0) * scale_e0m3

    err_e2m1 = (dq_e2m1 - tiled).pow(2).sum(dim=(-1, -2))
    err_e0m3 = (dq_e0m3 - tiled).pow(2).sum(dim=(-1, -2))

    return (err_e0m3 < err_e2m1).float().mean().item()


def synthetic_tensors(seed: int = 0):
    torch.manual_seed(seed)
    tensors = {
        "gaussian [4096, 4096]":        torch.randn(4096, 4096),
        "gaussian+outlier [4096,4096]": None,
        "laplace [1024, 4096]":         torch.distributions.Laplace(0.0, 1.0).sample((1024, 4096)),
        "student-t [512, 11008]":       torch.distributions.StudentT(3.0).sample((512, 11008)),
    }
    w = torch.randn(4096, 4096)
    # inject channel-wise outliers, which is the regime where the coarser type blocks matter
    outlier_col = torch.randperm(w.shape[1])[: w.shape[1] // 128]
    w[:, outlier_col] *= 20.0
    tensors["gaussian+outlier [4096,4096]"] = w

    return tensors


def model_tensors(model_name: str, max_layers: int):
    import json
    import os
    from transformers import AutoModelForCausalLM

    model2path = json.load(open(os.path.join(os.path.dirname(__file__), "model2path.json")))
    model = AutoModelForCausalLM.from_pretrained(
        model2path[model_name], torch_dtype=torch.bfloat16, device_map="cpu"
    )

    tensors, count = {}, 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("head" not in name):
            tensors[f"{name} {tuple(module.weight.shape)}"] = module.weight.data.clone()
            count += 1
            if count >= max_layers:
                break

    return tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None,
                        help="Optional model from model2path.json. If unset, synthetic tensors are used.")
    parser.add_argument("--max_layers", type=int, default=4, help="Number of Linear layers to sweep when --model_name is set.")
    parser.add_argument("--type_blocks", type=lambda s: s.split(","), default=DEFAULT_TYPE_BLOCKS,
                        help="Comma separated type-block shapes, e.g. \"1x16,16x16,32x128\".")
    parser.add_argument("--groupsize", type=int, default=16, help="NVFP4 scale-block size. Must be 16.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    type_blocks = [parse_type_block(tb) for tb in args.type_blocks]

    if args.model_name is None:
        tensors = synthetic_tensors(args.seed)
    else:
        tensors = model_tensors(args.model_name, args.max_layers)

    for name, w_fp in tensors.items():
        w_fp = w_fp.to(torch.bfloat16)
        print(f"\n=================== {name} ===================")
        print(f"{'format':<24}{'NMSE':>14}{'SQNR (dB)':>12}{'E0M3 blocks':>14}")

        nmse, sqnr = quant_error(w_fp, quant_nvfp4(w_fp, groupsize=args.groupsize))
        print(f"{'nvfp4 (E2M1 only)':<24}{nmse:>14.3e}{sqnr:>12.3f}{'-':>14}")

        nmse, sqnr = quant_error(w_fp, quant_nvif4(w_fp, groupsize=args.groupsize))
        print(f"{'nvif4 (per 16 blk)':<24}{nmse:>14.3e}{sqnr:>12.3f}{'-':>14}")

        for block_m, block_k in type_blocks:
            w_dq = quant_mixfp4(w_fp, groupsize=args.groupsize, type_block=(block_m, block_k))
            nmse, sqnr = quant_error(w_fp, w_dq)
            frac = e0m3_fraction(w_fp, (block_m, block_k), args.groupsize)
            print(f"{f'mixfp4 {block_m}x{block_k}':<24}{nmse:>14.3e}{sqnr:>12.3f}{frac*100:>13.1f}%")


if __name__ == "__main__":
    main()

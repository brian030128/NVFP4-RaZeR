"""
    Weight-MSE simulation for google/diffusiongemma-26B-A4B-it (DiffusionGemma, MoE + vision).

    "Just do MSE": for every quantizable weight of the *text decoder* we fake-quantize it with each
    FP4 format and report the relative quantization error (NMSE = ||w - w_q||^2 / ||w||^2) and SQNR.
    No model graph is built -- the tensors are read straight from the safetensors shards -- so the
    brand-new DiffusionGemma architecture does not need to be supported by transformers, and no
    diffusion / vision machinery is instantiated.

    Formats reported (all groupsize 16 = the NVFP4 scale block):
        nvfp4            plain NVFP4, E2M1 only, block scale = block_max / 6
        nvfp4_4over6     E2M1 with the FourOverSix two-point block-scale search {1, 1.5}   <-- baseline A
        mixfp4  <tb>     E2M1-vs-E0M3 chosen per type block, argmin MSE (block scale /6 vs /7)  <-- B
        mix_4_6 <tb>     same, but the E2M1 side also runs the 4over6 search before comparing

    The requested comparison is  nvfp4_4over6  vs  mixfp4 @ 8x64 on the weights.  Activation
    quantization ("keep activation 4over6") does not enter a weight-only MSE measurement and is
    therefore not simulated here -- it would only matter for a perplexity / end-to-end run.

    Orientation: NVFP4 groups along the reduction (K) dimension, and every quantizer in this repo
    groups along the *last* axis. 2D attention / dense-MLP weights are stored (out, in) so K = in is
    already last. The MoE expert weights are stacked 3D tensors (num_experts, ., .) whose reduction
    axis is the *middle* one; each expert is its own GEMM operand, so we slice per expert, move K to
    the last axis, and quantize (which also gives each expert its own NVFP4 per-tensor global scale
    and bounds GPU memory).
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import torch
from safetensors import safe_open

from quantize.quantizer import (
    quant_nvfp4,
    quant_nvfp4_4over6,
    quant_mixfp4,
    quant_mix_4_6,
)
from quantize.utils import parse_type_block, format_type_block


# --- which decoder weights are 4-bit-quantized (norms / router / embeddings / lm_head excluded) ---
ATTN_PROJ  = ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
              "self_attn.v_proj.weight", "self_attn.o_proj.weight")
MLP_PROJ   = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
EXPERT_2D  = ("experts.gate_up_proj", "experts.down_proj")  # stacked 3D: (num_experts, ., .)


def classify(key: str):
    """Return (coarse_group, fine_name) for a weight key, or None to skip it."""
    if ".experts.gate_up_proj" in key:
        return "moe_experts", "expert_gate_up"
    if ".experts.down_proj" in key:
        return "moe_experts", "expert_down"
    for p in ATTN_PROJ:
        if key.endswith(p):
            return "attention", p.split(".")[-2]      # q_proj / k_proj / v_proj / o_proj
    for p in MLP_PROJ:
        if key.endswith(p):
            return "dense_mlp", p.split(".")[-2]       # gate_proj / up_proj / down_proj
    return None


def expected_k(fine_name: str, hidden: int, moe_inter: int):
    """Reduction (in) dimension a stacked expert tensor should be oriented to have last."""
    if fine_name == "expert_gate_up":
        return hidden            # gate_up: in = hidden_size
    if fine_name == "expert_down":
        return moe_inter         # down: in = moe_intermediate_size
    return None


def orient_expert(mat2d: torch.Tensor, k: int) -> torch.Tensor:
    """Move the reduction axis (== k) to last for a per-expert 2D slice."""
    if mat2d.shape[-1] == k:
        return mat2d
    if mat2d.shape[0] == k:
        return mat2d.transpose(0, 1).contiguous()
    raise ValueError(f"expert slice {tuple(mat2d.shape)} has no axis == expected K {k}")


@torch.no_grad()
def errors_for(w: torch.Tensor, type_block, device):
    """
        Return {format: (noise_sum, e0m3_frac_or_None)} for one oriented weight matrix, plus the
        signal energy. noise/signal are float64 python scalars accumulated across the model.
    """
    w = w.to(device)
    signal = w.to(torch.float32).pow(2).sum().item()

    def noise(w_dq):
        return (w.to(torch.float32) - w_dq.to(torch.float32)).pow(2).sum().item()

    out = {}
    out["nvfp4"]        = (noise(quant_nvfp4(w, groupsize=16)), None)
    out["nvfp4_4over6"] = (noise(quant_nvfp4_4over6(w, groupsize=16)), None)

    tb = parse_type_block(type_block)
    out[f"mixfp4"]  = (noise(quant_mixfp4(w, groupsize=16, type_block=tb)),
                       _e0m3_frac(w, tb, four_over_six=False))
    out[f"mix_4_6"] = (noise(quant_mix_4_6(w, groupsize=16, type_block=tb)),
                       _e0m3_frac(w, tb, four_over_six=True))
    return signal, out


@torch.no_grad()
def _e0m3_frac(w, tb, four_over_six):
    """Fraction of type blocks (in this matrix) that elect E0M3. Runs on w's device."""
    from quantize.quantizer import _tile_type_blocks, _quant_e2m1

    block_m, block_k = tb
    x = w.reshape(-1, w.shape[-1]).to(torch.float32)
    if x.shape[-1] % block_k != 0:
        block_k = x.shape[-1]
    gscale = (x.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
    tiled, _ = _tile_type_blocks(x / gscale, block_m, block_k, 16)
    bmax = tiled.abs().amax(dim=-1, keepdim=True)

    err_e2m1 = None
    for qmax in ((6.0, 4.0) if four_over_six else (6.0,)):
        scale = (bmax / qmax).clamp(min=2 ** (-9), max=448.0).to(torch.float8_e4m3fn).to(tiled.dtype)
        err = (_quant_e2m1(tiled, scale) - tiled).pow(2).sum(dim=-1, keepdim=True)
        err_e2m1 = err if err_e2m1 is None else torch.minimum(err_e2m1, err)
    scale0 = (bmax / 7.0).clamp(min=2 ** (-9), max=448.0).to(torch.float8_e4m3fn).to(tiled.dtype)
    dq0 = (tiled / scale0).round().clamp(-7.0, 7.0) * scale0
    err_e0m3 = (dq0 - tiled).pow(2).sum(dim=-1, keepdim=True)

    elect = (err_e0m3.sum(dim=(-1, -2)) < err_e2m1.sum(dim=(-1, -2)))
    return float(elect.float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="Local snapshot dir of the model (safetensors + config.json + index).")
    ap.add_argument("--type_block", default="8x64", help="MixFP4 weight type block, e.g. 8x64.")
    ap.add_argument("--out", required=True, help="CSV output path (per-key rows).")
    ap.add_argument("--max_layers", type=int, default=0, help="0 = all decoder layers.")
    ap.add_argument("--max_experts", type=int, default=0,
                    help="0 = all experts per stacked tensor; else sample the first N (faster).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    tb = parse_type_block(args.type_block)
    print(f"device={device}  type_block={format_type_block(tb)}  model_dir={args.model_dir}")

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    hidden    = tcfg["hidden_size"]
    moe_inter = tcfg.get("moe_intermediate_size", tcfg.get("intermediate_size"))
    print(f"hidden_size={hidden}  moe_intermediate_size={moe_inter}")

    index_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"]

    # decoder weights only; group keys by shard for a single pass over each file
    targets = {}   # key -> (coarse, fine)
    for key in weight_map:
        if not key.startswith("model.decoder."):
            continue
        c = classify(key)
        if c is None:
            continue
        if args.max_layers:
            # keep only layers < max_layers (keys look like ...layers.<n>...)
            import re
            m = re.search(r"\.layers\.(\d+)\.", key)
            if m and int(m.group(1)) >= args.max_layers:
                continue
        targets[key] = c
    by_shard = defaultdict(list)
    for key in targets:
        by_shard[weight_map[key]].append(key)
    print(f"{len(targets)} decoder weight tensors selected across {len(by_shard)} shards")

    # accumulators
    formats = ["nvfp4", "nvfp4_4over6", "mixfp4", "mix_4_6"]
    agg_noise  = defaultdict(lambda: defaultdict(float))   # group -> fmt -> noise
    agg_signal = defaultdict(float)                         # group -> signal
    e0m3_accum = defaultdict(lambda: defaultdict(list))     # group -> fmt -> [fracs]
    rows = []

    t0 = time.time()
    done = 0
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(args.model_dir, shard), framework="pt", device="cpu") as f:
            for key in keys:
                coarse, fine = targets[key]
                t = f.get_tensor(key)
                k = expected_k(fine, hidden, moe_inter)

                sig_key = 0.0
                noise_key = {fmt: 0.0 for fmt in formats}
                frac_key = {fmt: [] for fmt in formats}

                if t.dim() == 3:                       # stacked experts: (E, ., .)
                    E = t.shape[0]
                    n_exp = E if args.max_experts == 0 else min(args.max_experts, E)
                    for e in range(n_exp):
                        w = orient_expert(t[e], k)
                        sig, out = errors_for(w, tb, device)
                        sig_key += sig
                        for fmt in formats:
                            noise_key[fmt] += out[fmt][0]
                            if out[fmt][1] is not None:
                                frac_key[fmt].append(out[fmt][1])
                else:                                  # 2D (out, in): K = in already last
                    sig, out = errors_for(t, tb, device)
                    sig_key += sig
                    for fmt in formats:
                        noise_key[fmt] += out[fmt][0]
                        if out[fmt][1] is not None:
                            frac_key[fmt].append(out[fmt][1])

                agg_signal[coarse] += sig_key
                agg_signal["ALL"]  += sig_key
                row = {"key": key, "group": coarse, "fine": fine,
                       "shape": "x".join(map(str, tuple(t.shape))), "signal": sig_key}
                for fmt in formats:
                    agg_noise[coarse][fmt] += noise_key[fmt]
                    agg_noise["ALL"][fmt]  += noise_key[fmt]
                    nmse = noise_key[fmt] / sig_key if sig_key > 0 else float("nan")
                    row[f"nmse_{fmt}"] = nmse
                    if frac_key[fmt]:
                        fr = sum(frac_key[fmt]) / len(frac_key[fmt])
                        row[f"e0m3_{fmt}"] = fr
                        e0m3_accum[coarse][fmt].extend(frac_key[fmt])
                        e0m3_accum["ALL"][fmt].extend(frac_key[fmt])
                rows.append(row)
                done += 1
                if done % 20 == 0 or done == len(targets):
                    print(f"  [{done}/{len(targets)}] {key}  ({time.time()-t0:.1f}s)", flush=True)

    # write per-key CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["key", "group", "fine", "shape", "signal"] + \
                 [f"nmse_{f}" for f in formats] + [f"e0m3_{f}" for f in ("mixfp4", "mix_4_6")]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {args.out}")

    # summary
    def sqnr(nmse):
        import math
        return float("inf") if nmse <= 0 else -10.0 * math.log10(nmse)

    print("\n================= AGGREGATE (energy-weighted NMSE = sum noise / sum signal) =================")
    order = ["ALL", "attention", "dense_mlp", "moe_experts"]
    header = f"{'group':<13}" + "".join(f"{f:>16}" for f in formats)
    print(header)
    for g in order:
        if agg_signal.get(g, 0) <= 0:
            continue
        line = f"{g:<13}"
        for fmt in formats:
            nmse = agg_noise[g][fmt] / agg_signal[g]
            line += f"{nmse:>10.3e}({sqnr(nmse):4.1f})"
        print(line)
    print("  (each cell: NMSE(SQNR dB).  smaller NMSE / larger SQNR = better)")

    print("\n----- headline: nvfp4_4over6 vs mixfp4 @ %s -----" % format_type_block(tb))
    for g in order:
        if agg_signal.get(g, 0) <= 0:
            continue
        a = agg_noise[g]["nvfp4_4over6"] / agg_signal[g]
        b = agg_noise[g]["mixfp4"] / agg_signal[g]
        rel = (b - a) / a * 100.0
        verdict = "mixfp4 better" if b < a else "4over6 better"
        print(f"  {g:<13} 4over6 NMSE={a:.4e}  mixfp4 NMSE={b:.4e}  "
              f"(mixfp4 {rel:+.2f}% vs 4over6, {verdict})")

    print("\n----- E0M3 election fraction (mean over type blocks) -----")
    for g in order:
        parts = []
        for fmt in ("mixfp4", "mix_4_6"):
            fr = e0m3_accum[g].get(fmt)
            if fr:
                parts.append(f"{fmt}={sum(fr)/len(fr)*100:.1f}%")
        if parts:
            print(f"  {g:<13} " + "  ".join(parts))

    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

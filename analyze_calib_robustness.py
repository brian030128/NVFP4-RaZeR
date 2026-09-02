"""
    How much calibration does `hess` actually need, and does the SOURCE matter?

    The rule search showed that what distinguishes the tiles `hess` declines is `imp_max` -- one
    very high-importance channel in the tile, 47% higher than in the tiles it keeps. Every
    weight-only feature is identical between the two groups (peakedness 2.809 vs 2.839, energy
    concentration 0.0620 vs 0.0622), so that decision is an ACTIVATION property and no weight
    statistic can stand in for it. No calibration-free rule beat the do-nothing baseline.

    That changes the question. Instead of removing calibration, ask how weak it can be:

      * does E[x_j^2] from RANDOM TOKEN IDS reproduce the wikitext estimate?
      * does ONE batch reproduce four?
      * does a different corpus (c4) reproduce wikitext?

    Activation outlier channels are widely reported to be a fixed, largely input-independent
    property of a trained model -- that is the premise SmoothQuant and AWQ rely on. If it holds
    here, "needs calibration" collapses to "needs one forward pass of arbitrary tokens", which is
    universal in every sense that matters for deployment.

    Reports, per source: rank correlation of per-channel importance against the wikitext reference,
    and -- what actually matters -- the fraction of 8x64 tiles whose E0M3 election is unchanged.
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantize.reorder import elect_mask, gain_features, scale_block_gain


def elect(gain, bm, chunks, rule, margin):
    M, N = gain.shape
    g = torch.nn.functional.pad(gain, (0, (-N) % chunks, 0, (-M) % bm))
    nrg, ncg = g.shape[0] // bm, g.shape[1] // chunks
    t = g.reshape(nrg, bm, ncg, chunks).permute(0, 2, 1, 3).reshape(-1, bm * chunks)
    return elect_mask(gain_features(t).sum(dim=1), rule, margin, bm * chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="llama-3.1-8b-local")
    ap.add_argument("--layer_stride", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--clip", type=str, default="a1")
    ap.add_argument("--rule", type=str, default="harm")
    ap.add_argument("--margin", type=float, default=1.5)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quantize import collect_importance

    here = os.path.dirname(os.path.abspath(__file__))
    path = json.load(open(os.path.join(here, "model2path.json")))[args.model_name]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    vocab = model.config.vocab_size

    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(wt["text"][:20000]), return_tensors="pt").input_ids
    c4 = load_dataset("allenai/c4",
                      data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                      split="validation")
    enc_c4 = tok("\n\n".join(c4[i]["text"] for i in range(2000)), return_tensors="pt").input_ids

    torch.manual_seed(0)
    L = args.seq_len
    sources = {
        "wikitext4": [enc[:, i * L:(i + 1) * L] for i in range(4)],
        "wikitext1": [enc[:, :L]],
        "c4x4":      [enc_c4[:, i * L:(i + 1) * L] for i in range(4)],
        "random4":   [torch.randint(0, vocab, (1, L)) for _ in range(4)],
    }
    imps = {k: collect_importance(model, v, device=model.device) for k, v in sources.items()}
    print(f"collected importance from {len(imps)} sources\n", flush=True)

    ref = "wikitext4"
    bm, chunks = 8, 4
    agree = {k: [0, 0] for k in sources if k != ref}
    rho = {k: [] for k in sources if k != ref}

    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or "head" in name:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None or int(m.group(1)) % args.layer_stride != 0:
            continue
        if imps[ref].get(name) is None:
            continue
        w = mod.weight.data.float().cpu()
        gs = (w.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
        ws = w / gs
        a = imps[ref][name].float().cpu()
        e_ref = elect(scale_block_gain(ws, 16, "mse", args.clip, importance=a),
                      bm, chunks, args.rule, args.margin)
        for k in agree:
            ik = imps[k][name].float().cpu()
            e_k = elect(scale_block_gain(ws, 16, "mse", args.clip, importance=ik),
                        bm, chunks, args.rule, args.margin)
            agree[k][0] += int((e_k == e_ref).sum())
            agree[k][1] += int(e_ref.numel())
            ra = torch.argsort(torch.argsort(a)).double()
            rb = torch.argsort(torch.argsort(ik)).double()
            ra, rb = ra - ra.mean(), rb - rb.mean()
            rho[k].append(float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=1e-30)))

    print(f"{'source':>12} {'rho vs wikitext4':>18} {'tile elections agree':>22}")
    print(f"{ref:>12} {1.0:18.4f} {1.0:22.4f}   <- reference")
    for k in agree:
        print(f"{k:>12} {sum(rho[k]) / len(rho[k]):18.4f} "
              f"{agree[k][0] / max(agree[k][1], 1):22.4f}")
    print("\nIf random tokens reproduce the wikitext elections, `hess` needs no real data at all.")


if __name__ == "__main__":
    main()

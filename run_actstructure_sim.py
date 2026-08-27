"""
    Is there per-channel structure in ACTIVATIONS that reordering can exploit?

    Two measurements on real weights have now failed for the same reason: the per-input-channel
    component of the variance is ~1-3% and everything else is per-element idiosyncratic.

        E0M3 preference (run_reorder_sim.py --diagnostics_only):  col_share 0.004
        log|W| magnitude (run_blockorder_sim.py):                 col_share 0.004 - 0.028

    A permutation of the reduction axis can only exploit a per-CHANNEL effect, so both died.

    Activations are the opposite case, and it is well established: outliers sit in a small number of
    FIXED channels, which is why SmoothQuant and AWQ scale per channel. CLAUDE.md measures it in
    this repo already -- the median per-layer max/min of diag(S) = E[x_j^2] is ~4.1e3, against a
    ~1.03x column spread in the weights.

    This script runs the SAME decomposition on real calibration activations that
    `blockorder.column_profile_agreement` runs on weights, so the two numbers are directly
    comparable. Rows are tokens instead of output channels; the question is identical -- do the rows
    agree about which columns are large?

    If `col_share` is large here, then reordering belongs on the ACTIVATION operand (W4A4), not the
    weight operand, and the failures above are evidence about where to aim rather than about the
    idea itself.

        python run_actstructure_sim.py --model_name llama-3.1-8b-local --nsamples 4
"""
import argparse
import csv
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantize.blockorder import column_profile_agreement  # noqa: E402
from quantize.reorder import scale_block_gain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="llama-3.1-8b-local")
    ap.add_argument("--nsamples", type=int, default=4, help="calibration sequences")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--layer_stride", type=int, default=8)
    ap.add_argument("--projections", type=lambda s: s.split(","),
                    default=["q_proj", "o_proj", "up_proj", "down_proj"])
    ap.add_argument("--max_tokens", type=int, default=2048,
                    help="tokens kept per layer for the decomposition")
    ap.add_argument("--clip", type=str, default="heade0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    here = os.path.dirname(os.path.abspath(__file__))
    model2path = json.load(open(os.path.join(here, "model2path.json")))
    path = model2path[args.model_name]

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(data["text"][:20000]), return_tensors="pt").input_ids

    captured = {}

    def hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float().abs().cpu()
            prev = captured.get(name)
            captured[name] = x if prev is None else torch.cat([prev, x], 0)[: args.max_tokens]
        return fn

    handles = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or "head" in name:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None or int(m.group(1)) % args.layer_stride != 0:
            continue
        if args.projections and not any(name.endswith(p) for p in args.projections):
            continue
        handles.append(mod.register_forward_hook(hook(re.sub(r"^model\.", "", name))))

    torch.manual_seed(args.seed)
    with torch.no_grad():
        for i in range(args.nsamples):
            lo = i * args.seq_len
            ids = enc[:, lo: lo + args.seq_len].to(model.device)
            model(ids)
    for h in handles:
        h.remove()

    print(f"\n[actstructure] {len(captured)} layers, {args.nsamples} x {args.seq_len} tokens\n")
    print(f"{'layer':>34} {'row(tok)':>9} {'col(chan)':>10} {'resid':>7} {'max/min':>9} "
          f"{'sortErr%':>9} {'randErr%':>9} {'sortBest%':>10}")
    rows = []
    for name, x in captured.items():
        r, c, e = column_profile_agreement(x, num_sample=args.max_tokens)
        m2 = x.pow(2).mean(dim=0)
        spread = float(m2.max() / m2[m2 > 0].min()) if (m2 > 0).any() else float("nan")

        # TRUE activation quantization error, identity vs sorted vs random control. `sum
        # block_max^2` was measured to be a broken proxy on weights (it improved 37.7% while the
        # real error more than doubled), so this reports only what the quantizer actually pays.
        gs = (x.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
        xs = (x / gs).float()
        gen = torch.Generator().manual_seed(args.seed)

        def aerr(cols):
            _, e2, e0 = scale_block_gain(xs if cols is None else xs[:, cols], 16, "mse",
                                         args.clip, return_losses=True)
            return (float(e2.to(torch.float64).sum()),
                    float(torch.minimum(e2, e0).to(torch.float64).sum()))

        b_e2, b_best = aerr(None)
        s_e2, s_best = aerr(torch.argsort(m2, descending=True))
        r_e2, _      = aerr(torch.randperm(x.shape[1], generator=gen))
        pct = lambda v, base: 100.0 * (base - v) / base
        print(f"{name:>34} {r:9.4f} {c:10.4f} {e:7.4f} {spread:9.2g} "
              f"{pct(s_e2, b_e2):8.3f}% {pct(r_e2, b_e2):8.3f}% {pct(s_best, b_best):9.3f}%")
        rows.append(dict(layer=name, row_share=round(r, 4), col_share=round(c, 4),
                         resid_share=round(e, 4), diagS_max_over_min=round(spread, 3),
                         sorted_e2m1_pct=round(pct(s_e2, b_e2), 4),
                         random_e2m1_pct=round(pct(r_e2, b_e2), 4),
                         sorted_best_pct=round(pct(s_best, b_best), 4),
                         tokens=int(x.shape[0]), channels=int(x.shape[1])))

    n = len(rows)
    a = lambda k: sum(r[k] for r in rows) / n
    print(f"\nmean col_share (activations) = {a('col_share'):.4f}")
    print(f"mean E2M1 error reduction from sorting channels = {a('sorted_e2m1_pct'):+.3f}%"
          f"   (random control {a('random_e2m1_pct'):+.3f}%)")
    print("compare: weights log|W| col_share ~ 0.004-0.028, E0M3 tag grid col_share ~ 0.004")
    print("A permutation of the reduction axis can only exploit the col_share component.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""
    Can we PREDICT which election rule a model needs, instead of sweeping them?

    Section 1 of MIXFP4_REPORT leaves one thing unresolved: the type block beats NVFP4 on every
    model, but the strictness that wins is model-dependent, and the wrong choice is expensive
    (`h1.5` is best on Llama-3.1-8B and costs +0.4231 wikitext on Qwen3-4B). Sweeping four rules per
    model is exactly what a deployable method cannot do.

    THE HYPOTHESIS, AND WHY IT IS NOT JUST CURVE-FITTING
    ---------------------------------------------------
    The election compares an estimate of how much a tile's quantization raises the LAYER OUTPUT
    error. For Y = X W^T that error is tr(dW S dW^T) with S = E[x x^T]. What the election actually
    computes is the DIAGONAL surrogate, sum_j S_jj dW_ij^2, because a full S per layer is not
    affordable. CLAUDE.md states the resulting certificate: with D = diag(S), the surrogate errs by
    at most

        ||S - D||_2 * (||dW_A||_F^2 + ||dW_B||_F^2)

    So the amount by which a tile's measured gain can be wrong scales with the OFF-DIAGONAL MASS of
    S. And `h<lambda>` is precisely a margin against that error: elect only when the winners beat the
    losers by `lambda`, i.e. only when the gain exceeds the uncertainty.

    That gives a falsifiable prediction rather than a fishing expedition:

        a model with more off-diagonal mass in S needs a LARGER lambda.

    Qwen3-4B needs lambda = 10 and is destroyed by 1.5; Llama-3.1-8B prefers 1.5. If the hypothesis
    holds, Qwen3-4B has visibly more off-diagonal mass. If it does not, this predictor is dead and
    the report should say so -- round 6 already killed one family of calibration-free proxies, and a
    second failure is worth recording rather than hiding.

    WHAT ELSE IS MEASURED, AND WHY
    ------------------------------
    Three cheaper statistics ride along, so that a null on the principled one still leaves something:

      `elect_rate(lambda)`  what fraction of tiles elect E0M3 at each lambda. A model whose election
                            rate collapses between 1.5 and 10 is one where lambda matters; a model
                            where it barely moves cannot care which rule is used.
      `marginality`         the fraction of tiles whose harm ratio sits in [1, 10], i.e. tiles whose
                            decision DEPENDS on lambda. This is the direct "how much is at stake"
                            number and needs no calibration beyond the importance vector.
      `mse_hess_disagree`   fraction of tiles where the unweighted and importance-weighted criteria
                            disagree about electing. High disagreement means the cheap criterion is
                            uninformative on this model, which is its own reason to be strict.

    Sampling: full S is d x d per layer (67 MB at d=4096), so it is built for a stride of layers and
    a subset of projections only. The statistics are ratios and are reported per layer, so a sample
    is enough; `--layer_stride` controls it.

    Usage:
        python analyze_lambda_predictor.py --model_name qwen3-4b --out results/lambda/qwen3-4b.json
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.reorder import scale_block_gain


# ------------------------------------------------------------------ calibration


def get_calib(tokenizer, n_batch, seq_len, device):
    from datasets import load_dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(data["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids
    return [ids[:, i * seq_len:(i + 1) * seq_len].to(device) for i in range(n_batch)]


@torch.no_grad()
def collect_covariance(model, batches, targets):
    """
        E[x x^T] at the input of the named Linear modules, plus its diagonal.

        Accumulated in float64 on the module's own device -- these are sums of squares over
        ~8k tokens and float32 loses the small off-diagonal entries that are the whole point.
    """
    cov, count, handles = {}, {}, []

    def make_hook(name):
        def hook(_m, inputs, _o):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).double()
            c = x.T @ x
            cov[name] = c if name not in cov else cov[name] + c
            count[name] = count.get(name, 0) + x.shape[0]
        return hook

    for name, mod in model.named_modules():
        if name in targets:
            handles.append(mod.register_forward_hook(make_hook(name)))
    for b in batches:
        model(b)
    for h in handles:
        h.remove()
    return {k: v / count[k] for k, v in cov.items()}


# ------------------------------------------------------------------ statistics


def offdiag_stats(S):
    """
        How far is S from its own diagonal?

        `offdiag_spec` is ||S - D||_2 / ||D||_2, the certificate quantity: it bounds the relative
        error of the diagonal surrogate the election actually uses. `coherence` is the mean absolute
        correlation, a scale-free companion that does not depend on one large eigenvalue.
    """
    d = torch.diag(S)
    D = torch.diag(d)
    R = S - D
    # spectral norms via a few power iterations -- these matrices are up to 5120^2
    def specnorm(M, iters=30):
        v = torch.randn(M.shape[0], dtype=M.dtype, device=M.device)
        v /= v.norm()
        for _ in range(iters):
            v = M @ (M.T @ v)
            n = v.norm()
            if n == 0:
                return 0.0
            v /= n
        return float((M.T @ v).norm())

    denom = float(d.abs().max()) or 1.0
    inv = d.clamp(min=1e-12).rsqrt()
    C = R * inv.unsqueeze(0) * inv.unsqueeze(1)
    n = S.shape[0]
    return {
        "offdiag_spec": specnorm(R) / denom,
        "coherence": float(C.abs().sum() / (n * (n - 1))),
        "kappa_diag": float(d.max() / d.mean()),
    }


@torch.no_grad()
def election_stats(w, imp, block_m=8, block_k=64, groupsize=16, lambdas=(1.0, 1.5, 2.0, 5.0, 10.0)):
    """
        Harm ratios per type block, and what each lambda would elect.

        The harm ratio is  sum_{g>0} g_b / sum_{g<0} |g_b|  over the tile's scale blocks, which is
        exactly what `h<lambda>` thresholds. Reporting the DISTRIBUTION of that ratio says how much
        the choice of lambda can possibly matter on this model, before any perplexity is run.
    """
    out = {}
    for tag, importance in (("hess", imp), ("mse", None)):
        g = scale_block_gain(w, groupsize, "mse", "a1", importance=importance)
        M, N = g.shape
        c = block_k // groupsize
        M2, N2 = (M // block_m) * block_m, (N // c) * c
        if M2 == 0 or N2 == 0:
            return None
        t = g[:M2, :N2].reshape(M2 // block_m, block_m, N2 // c, c).permute(0, 2, 1, 3)
        t = t.reshape(-1, block_m * c)
        pos = t.clamp(min=0).sum(dim=1)
        neg = (-t).clamp(min=0).sum(dim=1)
        ratio = pos / neg.clamp(min=1e-30)
        out[tag] = {
            "n_tile": int(t.shape[0]),
            "elect": {str(L): float((ratio > L).float().mean()) for L in lambdas},
            # tiles whose decision actually depends on lambda in the measured range
            "marginality": float(((ratio > 1.0) & (ratio <= 10.0)).float().mean()),
            "median_ratio": float(ratio.median()),
        }
        out[f"_ratio_{tag}"] = ratio
    r_h, r_m = out.pop("_ratio_hess"), out.pop("_ratio_mse")
    out["mse_hess_disagree"] = {
        str(L): float(((r_h > L) != (r_m > L)).float().mean()) for L in (1.0, 1.5, 10.0)
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--layer_stride", type=int, default=8)
    ap.add_argument("--projections", default="q_proj,v_proj,down_proj")
    ap.add_argument("--calib_batches", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = json.load(open("model2path.json"))[args.model_name]
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, device_map="auto")
    model.eval()

    projs = args.projections.split(",")
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or not name.startswith("model.layers."):
            continue
        li = int(name.split(".")[2])
        if li % args.layer_stride == 0 and name.split(".")[-1] in projs:
            targets.append(name)
    print(f"{len(targets)} target layers: {targets[:4]} ...", flush=True)

    dev = next(model.parameters()).device
    batches = get_calib(tok, args.calib_batches, args.seq_len, dev)
    cov = collect_covariance(model, batches, set(targets))

    mods = dict(model.named_modules())
    rows = []
    for name in targets:
        S = cov[name]
        d = torch.diag(S).float()
        st = offdiag_stats(S)
        w = mods[name].weight.data.to(torch.float32)
        es = election_stats(w, d.to(w.device))
        if es is None:
            continue
        rows.append({"layer": name, **st, **es})
        print(f"  {name:<34} offdiag={st['offdiag_spec']:.3f} coh={st['coherence']:.4f} "
              f"kappa={st['kappa_diag']:.1f} elect1.5={es['hess']['elect']['1.5']:.3f} "
              f"elect10={es['hess']['elect']['10.0']:.3f} marg={es['hess']['marginality']:.3f}",
              flush=True)
        del S
        cov[name] = None

    def mean(key, sub=None):
        vals = [(r[key] if sub is None else r[key][sub]) for r in rows]
        return sum(vals) / max(len(vals), 1)

    summary = {
        "model": args.model_name,
        "n_layer_sampled": len(rows),
        "offdiag_spec": mean("offdiag_spec"),
        "coherence": mean("coherence"),
        "kappa_diag": mean("kappa_diag"),
        "elect_1.5": sum(r["hess"]["elect"]["1.5"] for r in rows) / max(len(rows), 1),
        "elect_10": sum(r["hess"]["elect"]["10.0"] for r in rows) / max(len(rows), 1),
        "marginality": sum(r["hess"]["marginality"] for r in rows) / max(len(rows), 1),
        "disagree_1.5": sum(r["mse_hess_disagree"]["1.5"] for r in rows) / max(len(rows), 1),
    }
    print("\nSUMMARY " + json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "layers": rows}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()

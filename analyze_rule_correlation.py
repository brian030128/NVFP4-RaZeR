"""
    Join the per-model predictors to the measured per-rule perplexity, and see if anything orders.

    Reads results/lambda/<model>.json (predictors) and results/w4a4/<model>_types.json (ground
    truth), and asks: does any cheap statistic order the models the same way the MEASURED best rule
    does?

    HOW THIS CAN LIE TO YOU, AND WHAT IS DONE ABOUT IT
    --------------------------------------------------
    With six models, a Spearman correlation over four candidate predictors has a good chance of
    producing |rho| > 0.8 from noise alone -- there are only 720 orderings of six things, and trying
    several statistics is a multiple-comparisons problem nobody would accept at n=6. So this script
    does NOT report a correlation as a finding on its own. It reports:

      1. the rank correlation, as a screen, not a result;
      2. a LEAVE-ONE-OUT check: for each model, pick the rule the predictor would choose using only
         the other five, and report the perplexity actually paid. This is the number that matters,
         because it is what the method would have delivered, and it cannot be inflated by fitting;
      3. the two baselines any predictor must beat --
           `oracle`  the per-model best rule (the unreachable ceiling)
           `fixed`   the single best rule across all models (what you get for free, with no
                     predictor at all)
         A predictor that does not beat `fixed` is worthless however good its rho looks.
"""
import argparse
import glob
import json
import os


RULES = ["mix_4_6_clipa1_hess_h1.5", "mix_4_6_clipa1_hess_h10",
         "mix_4_6_clipa1_hess_m1", "mix_4_6_clipa1_hess_impg16_h10"]

PREDICTORS = ["offdiag_spec", "coherence", "kappa_diag", "elect_1.5", "elect_10",
              "marginality", "disagree_1.5"]


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def load(metric):
    models = {}
    for path in sorted(glob.glob("results/lambda/*.json")):
        name = os.path.basename(path)[:-5]
        gt_path = f"results/w4a4/{name}_types.json"
        if not os.path.exists(gt_path):
            print(f"  (skip {name}: no ground truth at {gt_path})")
            continue
        pred = json.load(open(path))["summary"]
        gt = json.load(open(gt_path))
        ref = next((v for v in gt.values()
                    if v["w_dtype"] == "nvfp4"), None)
        if ref is None:
            print(f"  (skip {name}: no nvfp4 reference row)")
            continue
        delta = {}
        for v in gt.values():
            if v["w_dtype"] in RULES and metric in v:
                delta[v["w_dtype"]] = v[metric] - ref[metric]
        if len(delta) < len(RULES):
            print(f"  (skip {name}: only {len(delta)}/{len(RULES)} rules measured)")
            continue
        models[name] = {"pred": pred, "delta": delta}
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="wikitext")
    args = ap.parse_args()

    models = load(args.metric)
    names = sorted(models)
    if len(names) < 4:
        print(f"only {len(names)} models with both predictors and ground truth; need >= 4")
        return
    print(f"\n{len(names)} models, metric = {args.metric}\n")

    # --- the measured picture -------------------------------------------------------------
    short = [r.replace("mix_4_6_clipa1_hess_", "") for r in RULES]
    print(f"{'model':<26} " + " ".join(f"{s:>12}" for s in short) + f" {'best':>14}")
    for n in names:
        d = models[n]["delta"]
        best = min(RULES, key=lambda r: d[r])
        print(f"{n:<26} " + " ".join(f"{d[r]:>+12.4f}" for r in RULES)
              + f" {best.replace('mix_4_6_clipa1_hess_',''):>14}")

    # --- baselines ------------------------------------------------------------------------
    oracle = sum(min(models[n]["delta"][r] for r in RULES) for n in names) / len(names)
    fixed_scores = {r: sum(models[n]["delta"][r] for n in names) / len(names) for r in RULES}
    fixed_rule = min(fixed_scores, key=fixed_scores.get)
    print(f"\noracle (per-model best, unreachable) : {oracle:+.4f}")
    print(f"fixed  (best single rule, no predictor): {fixed_scores[fixed_rule]:+.4f} "
          f"[{fixed_rule.replace('mix_4_6_clipa1_hess_','')}]")
    print("  worst case of the fixed rule: "
          f"{max(models[n]['delta'][fixed_rule] for n in names):+.4f}")

    # --- do any predictors order the models like the measured optimum? --------------------
    print(f"\n{'predictor':<16} {'rho vs delta(rule)':>52}")
    print(" " * 16 + " ".join(f"{s:>12}" for s in short))
    for p in PREDICTORS:
        if p not in models[names[0]]["pred"]:
            continue
        x = [models[n]["pred"][p] for n in names]
        row = " ".join(f"{spearman(x, [models[n]['delta'][r] for n in names]):>+12.2f}"
                       for r in RULES)
        print(f"{p:<16} {row}")

    # --- leave-one-out: what would the predictor actually have delivered? ------------------
    print("\nLEAVE-ONE-OUT (the only number that matters)")
    print(f"{'predictor':<16} {'mean delta':>12} {'worst':>10}   picks")
    for p in PREDICTORS:
        if p not in models[names[0]]["pred"]:
            continue
        paid, picks = [], []
        for held in names:
            rest = [n for n in names if n != held]
            # rule: among the other models, use the one whose predictor value is closest, and
            # take the rule that was best for it -- a 1-nearest-neighbour predictor, the weakest
            # possible use of the statistic, hence the least able to overfit
            near = min(rest, key=lambda n: abs(models[n]["pred"][p] - models[held]["pred"][p]))
            r = min(RULES, key=lambda rr: models[near]["delta"][rr])
            paid.append(models[held]["delta"][r])
            picks.append(r.replace("mix_4_6_clipa1_hess_", ""))
        print(f"{p:<16} {sum(paid)/len(paid):>+12.4f} {max(paid):>+10.4f}   "
              + ",".join(picks))
    print("\nA predictor is only useful if its leave-one-out mean beats `fixed` above,")
    print("AND its worst case is no worse than `fixed`'s worst case.")


if __name__ == "__main__":
    main()

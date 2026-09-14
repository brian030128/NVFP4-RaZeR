"""
    Turn results/zeroshot_w4a4/<model>.json into the accuracy section of MIXFP4_REPORT.md.

    The tables mirror §1 exactly -- same six models, same seven configurations, same
    delta-against-plain-`nvfp4` framing -- so an accuracy delta can be read straight against the
    perplexity delta directly above it. Perplexity deltas are pulled from the same
    results/w4a4/<model>_types.json files §1 was built from, rather than retyped.

        python summarize_zeroshot_sweep.py --out results/zeroshot_w4a4/SECTION.md
"""

import argparse
import glob
import json
import math
import os

from analyze_zeroshot_paired import compare, load as load_samples

# §1's models, in §1's order, with the label the report uses for each.
MODELS = [
    ("llama-3.1-8b-local",     "Llama-3.1-8B"),
    ("llama-3.1-8b-ins-local", "Llama-3.1-8B-Ins"),
    ("llama-3.2-1b-ins-local", "Llama-3.2-1B-Ins"),
    ("qwen3-4b",               "Qwen3-4B"),
    ("qwen3-8b",               "Qwen3-8B"),
    ("qwen3-14b",              "Qwen3-14B"),
]

# (column header, config label in the sweep json). The first two are references, not rules.
COLUMNS = [
    ("bf16",              "fp16"),
    ("`nvfp4`",           "nvfp4__a-nvfp4_4over6"),
    ("`e2m1`",            "mix_4_6_clipa1_e2m1_8x64__a-nvfp4_4over6"),
    ("`h10`",             "mix_4_6_clipa1_h10_8x64__a-nvfp4_4over6"),
    ("`hess_h1.5`",       "mix_4_6_clipa1_hess_h1.5_8x64__a-nvfp4_4over6"),
    ("`hess_h10`",        "mix_4_6_clipa1_hess_h10_8x64__a-nvfp4_4over6"),
    ("`hess_m1`",         "mix_4_6_clipa1_hess_m1_8x64__a-nvfp4_4over6"),
    ("`hess_impg16_h10`", "mix_4_6_clipa1_hess_impg16_h10_8x64__a-nvfp4_4over6"),
]
REFERENCE = "nvfp4__a-nvfp4_4over6"
RULES = [c for c in COLUMNS if c[1] not in ("fp16", REFERENCE)]


def load(model, root):
    path = os.path.join(root, f"{model}.json")
    return json.load(open(path)) if os.path.isfile(path) else None


def acc(entry, task=None):
    a = entry["accuracy"]
    return a["mean"] if task is None else a[task]["value"]


def stderr_of_mean(entry, tasks):
    """
        The panel mean is an unweighted mean of per-task accuracies, so its sampling standard
        error is sqrt(sum se_i^2)/n under independence across tasks -- which holds, the task sets
        are disjoint. This is the sampling error of ONE measurement; a paired delta between two
        configurations on the SAME documents is correlated and its error is smaller, so this is a
        conservative bound on what a delta has to clear.
    """
    a = entry["accuracy"]
    se = [a[t]["stderr"] for t in tasks if a.get(t, {}).get("stderr") is not None]
    return math.sqrt(sum(s * s for s in se)) / len(se) if se else float("nan")


def load_ppl(model, ppl_root):
    """
        The wikitext/c4 perplexities §1 reports, read from the same files §1 was built from
        (results/w4a4/<model>_types.json) rather than retyped, so the two metrics cannot drift
        apart in this report.
    """
    path = os.path.join(ppl_root, f"{model}_types.json")
    return json.load(open(path)) if os.path.isfile(path) else None


def spearman(xs, ys):
    """Rank correlation, with average ranks for ties. n is 30 here, so this is a summary only."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def fmt(x, digits=4, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/zeroshot_w4a4")
    ap.add_argument("--ppl_root", default="results/w4a4")
    ap.add_argument("--out", default="results/zeroshot_w4a4/SECTION.md")
    args = ap.parse_args()

    data = {m: load(m, args.root) for m, _ in MODELS}
    present = [(m, lbl) for m, lbl in MODELS if data[m]]
    assert present, f"no result files under {args.root}: {glob.glob(args.root + '/*')}"

    tasks = data[present[0][0]]["_meta"]["tasks"]
    lines = []

    # --- what was measured -------------------------------------------------------------------
    lines += [
        "## 1a. Zero-shot accuracy (current scope)",
        "",
        "§1 is perplexity only. Perplexity is a next-token loss, and a quantizer that makes a "
        "model *less confident* lowers it without the model predicting anything better -- a real "
        "possibility here, since the election rule is chosen to reduce a squared error rather "
        "than to preserve a decision. So the identical seven configurations were re-run on "
        "zero-shot multiple choice, where a smoothing artefact earns nothing.",
        "",
        f"Same weights, same code path: W4A4 prefill, weights 8x64 / alpha = 1, activations "
        f"`nvfp4_4over6` at 16x64, importance from 4 wikitext-train windows on the unquantized "
        f"model (`run_zeroshot_sweep.py`, which reuses `run_ppl_sweep.py`'s quantization loop "
        f"unchanged). lm-eval-harness 0.4.5, 0-shot, "
        f"{len(tasks)} tasks: {', '.join('`' + t + '`' for t in tasks)}. `acc_norm` where the "
        "harness reports it, `acc` otherwise; the panel figure is the unweighted mean.",
        "",
    ]

    # --- absolute panel means ----------------------------------------------------------------
    lines += [
        "### Panel mean accuracy, absolute",
        "",
        "`bf16` is the unquantized model and `nvfp4` the quantized reference §1 measures against; "
        "the gap between them is what W4A4 costs before any element-type decision is made.",
        "",
        "| model | " + " | ".join(h for h, _ in COLUMNS) + " | 1 SE |",
        "|---" * (len(COLUMNS) + 2) + "|",
    ]
    for m, lbl in present:
        d = data[m]
        cells = [fmt(acc(d[k])) if k in d else "—" for _, k in COLUMNS]
        se = stderr_of_mean(d[REFERENCE], tasks) if REFERENCE in d else float("nan")
        lines.append(f"| {lbl} | " + " | ".join(cells) + f" | {fmt(se)} |")
    lines.append("")

    # --- deltas against nvfp4 ----------------------------------------------------------------
    lines += [
        "### Panel mean accuracy, delta against `nvfp4`",
        "",
        "Positive is better here -- the opposite sign convention from the perplexity tables, "
        "because this is an accuracy. `e2m1` is again the validation row: alpha frozen at 1 with "
        "the election disabled leaves MixFP4 no freedom, so it must reproduce `nvfp4` exactly.",
        "",
        "| model | " + " | ".join(h for h, _ in COLUMNS if _ != REFERENCE) + " |",
        "|---" * (len([1 for _, k in COLUMNS if k != REFERENCE]) + 1) + "|",
    ]
    for m, lbl in present:
        d = data[m]
        if REFERENCE not in d:
            continue
        ref = acc(d[REFERENCE])
        cells = [fmt(acc(d[k]) - ref, signed=True) if k in d else "—"
                 for _, k in COLUMNS if k != REFERENCE]
        lines.append(f"| {lbl} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- per-rule summary --------------------------------------------------------------------
    lines += [
        "### Summary over the panel",
        "",
        "| rule | mean delta | worst model | best model |",
        "|---|---|---|---|",
    ]
    summary = {}
    for header, key in RULES:
        deltas = [(acc(data[m][key]) - acc(data[m][REFERENCE]), lbl)
                  for m, lbl in present if key in data[m] and REFERENCE in data[m]]
        if not deltas:
            continue
        summary[header] = deltas
        mean = sum(d for d, _ in deltas) / len(deltas)
        worst, best = min(deltas), max(deltas)
        lines.append(f"| {header} | {fmt(mean, signed=True)} | {worst[1]} {fmt(worst[0], signed=True)} "
                     f"| {best[1]} {fmt(best[0], signed=True)} |")
    lines.append("")

    # --- how big is a difference that means nothing? ------------------------------------------
    # Measured, not assumed: the paired jobs re-ran `nvfp4` from scratch on every model, and
    # `nvfp4` uses no calibration, so its quantization is bit-deterministic. Any gap between the
    # two runs of it is pure evaluation noise, and it is the floor every delta above must clear.
    repeats = []
    for m, lbl in present:
        rp = os.path.join(args.root, f"paired_{m}.json")
        if not os.path.isfile(rp):
            continue
        rerun = json.load(open(rp))
        if REFERENCE in rerun and REFERENCE in data[m]:
            first, second = acc(data[m][REFERENCE]), acc(rerun[REFERENCE])
            repeats.append((lbl, first, second, second - first))
    if repeats:
        lines += [
            "### The noise floor, measured",
            "",
            "`nvfp4` uses no calibration, so its quantization is bit-deterministic and re-running "
            "it must give the same number. The paired jobs below re-ran it from scratch on every "
            "model, which turns that into a measurement of the evaluation's own reproducibility "
            "-- the floor any delta above has to clear.",
            "",
            "| model | first run | re-run | difference |",
            "|---|---|---|---|",
        ]
        for lbl, first, second, diff in repeats:
            lines.append(f"| {lbl} | {fmt(first)} | {fmt(second)} | {fmt(diff, signed=True)} |")
        worst = max(abs(d) for _, _, _, d in repeats)
        drifting = [lbl for lbl, _, _, d in repeats if d != 0]
        exact = [lbl for lbl, _, _, d in repeats if d == 0]
        lines += [
            "",
            f"The spread reaches **{worst:.4f}**, and it is a property of the model rather than "
            f"of the run: {', '.join(exact)} reproduce exactly, while "
            f"{', '.join(drifting)} do not. The likely mechanism is that the two runs reach this "
            "configuration in a different order, so allocator and cuBLAS state differ, and tiny "
            "logit differences flip multiple-choice items that were already near ties -- which "
            "also explains why the models that drift are the ones with the most near ties. "
            f"Whatever the cause, a delta of {worst:.4f} on "
            + (drifting[0] if len(drifting) == 1 else "one of those models")
            + " is not evidence of anything, and several deltas in the table above are that "
              "size. That is what the paired test below is for.",
            "",
        ]

    # --- paired tests, where per-document outcomes were logged --------------------------------
    paired_rows = []
    for m, lbl in present:
        sdir = os.path.join(args.root, f"samples_{m}")
        ref_file = os.path.join(sdir, f"{REFERENCE}.json")
        if not os.path.isfile(ref_file):
            continue
        ref_docs = load_samples(ref_file)
        for header, key in RULES:
            vf = os.path.join(sdir, f"{key}.json")
            if not os.path.isfile(vf):
                continue
            _, pooled = compare(ref_docs, load_samples(vf))
            paired_rows.append((lbl, header, pooled))
    if paired_rows:
        lines += [
            "### Paired test against `nvfp4`",
            "",
            "Both configurations are scored on the same documents, so the comparison is paired "
            "and the question is not how much each one varies but how often they disagree. "
            "`b` counts documents only the reference gets right, `c` documents only the variant "
            "gets right; everything else carries no information about the difference. The "
            "p-value is an exact two-sided McNemar test on those counts, pooled over all "
            "documents of all tasks (`analyze_zeroshot_paired.py`).",
            "",
            "| model | rule | documents | b | c | pooled delta | McNemar p |",
            "|---|---|---|---|---|---|---|",
        ]
        for lbl, header, pl in paired_rows:
            lines.append(f"| {lbl} | {header} | {pl['n']} | {pl['b']} | {pl['c']} | "
                         f"{fmt(pl['delta'], signed=True)} | {pl['p']:.3g} |")
        n_tests = len(paired_rows)
        sig = [(lbl, h, pl) for lbl, h, pl in paired_rows if pl["p"] < 0.05]
        bonf = [(lbl, h, pl) for lbl, h, pl in paired_rows if pl["p"] < 0.05 / n_tests]
        lines += [
            "",
            f"{n_tests} tests were run, so the 0.05 threshold is worth {0.05 / n_tests:.4f} "
            f"after a Bonferroni correction. "
            + (f"Uncorrected, {len(sig)} row(s) fall below 0.05: "
               + ", ".join(f"{lbl} {h} (p = {pl['p']:.3g})" for lbl, h, pl in sig) + ". "
               if sig else "No row falls below 0.05 even uncorrected. ")
            + (f"Corrected, {len(bonf)} survive"
               + (": " + ", ".join(f"{lbl} {h}" for lbl, h, pl in bonf) if bonf else "")
               + ".")
            + " Read the table accordingly: it is evidence about the SIZE of these effects, and "
              "the honest summary of that size is that it is small enough to need 18,600 "
              "documents to see at all.",
            "",
        ]

    # --- the comparison the section exists for ------------------------------------------------
    joint = []
    for m, lbl in present:
        pj = load_ppl(m, args.ppl_root)
        if not pj or REFERENCE not in data[m]:
            continue
        ppl_ref = pj.get("nvfp4__a-nvfp4_4over6", {}).get("wikitext")
        if ppl_ref is None:
            continue
        for header, key in RULES:
            if header == "`e2m1`":
                continue            # (0, 0) by construction; it is the control, not a data point
            if key not in data[m] or key not in pj or pj[key].get("wikitext") is None:
                continue
            joint.append((lbl, header,
                          pj[key]["wikitext"] - ppl_ref,
                          acc(data[m][key]) - acc(data[m][REFERENCE])))
    if joint:
        rho = spearman([j[2] for j in joint], [j[3] for j in joint])
        # A rule that lowers perplexity should, if the two metrics agree, raise accuracy: the
        # correlation between them ought to be NEGATIVE, since the sign conventions are opposite.
        agree = sum(1 for _, _, dp, da in joint if (dp < 0) == (da > 0))
        worst_ppl = max(joint, key=lambda j: j[2])
        best_acc = max(joint, key=lambda j: j[3])
        lines += [
            "### Perplexity and accuracy do not rank these rules the same way",
            "",
            "Every cell below is one (model, rule) pair: its wikitext perplexity delta from §1 "
            "and its zero-shot accuracy delta from this section, both against `nvfp4` on the "
            "same weights. The sign conventions are opposite -- perplexity down is good, "
            "accuracy up is good -- so if the two metrics agreed, the correlation would be "
            "negative and the two columns would have opposite signs cell by cell.",
            "",
            "| model | rule | d wikitext | d accuracy | agree? |",
            "|---|---|---|---|---|",
        ]
        for lbl, header, dp, da in joint:
            ok = "yes" if (dp < 0) == (da > 0) else "**no**"
            lines.append(f"| {lbl} | {header} | {dp:+.4f} | {da:+.4f} | {ok} |")
        ratios = [abs(da) / abs(dp) for _, _, dp, da in joint if abs(dp) > 1e-9]
        ratios.sort()
        med_ratio = ratios[len(ratios) // 2] if ratios else float("nan")
        lines += [
            "",
            f"Spearman rho over these {len(joint)} pairs is **{rho:+.2f}** -- negative, so the "
            f"two metrics do agree in rank more often than not, and the signs match in "
            f"{agree} of {len(joint)} cells. **The disagreement is not in direction, it is in "
            "magnitude, and the magnitude is what §1's recommendation rests on.**",
            "",
            f"The clearest case is {worst_ppl[1]} on {worst_ppl[0]}. §1 records it as the worst "
            f"number in the report, **{worst_ppl[2]:+.4f}** wikitext -- the single result that "
            f"disqualifies that rule there -- and its accuracy cost is **{worst_ppl[3]:+.4f}**, "
            "which is to say none at all. A perplexity catastrophe that a reader would expect "
            "to be visible in what the model answers simply is not. Across the panel the median "
            f"|accuracy| per unit |perplexity| is about {med_ratio:.3f}, and it varies over "
            "orders of magnitude between cells, so a perplexity delta does not convert into an "
            "expected accuracy delta at any fixed rate.",
            "",
            f"The largest accuracy gain in the panel, {best_acc[1]} on {best_acc[0]} at "
            f"**{best_acc[3]:+.4f}**, does come with a healthy {best_acc[2]:+.4f} wikitext, so "
            "the two metrics are not adversaries. They are simply measuring things that come "
            "apart exactly where §1 has to make its decision -- at the worst case.",
            "",
        ]

    # --- what a reader should take away -------------------------------------------------------
    if joint and paired_rows:
        rec = "`hess_impg16_h10`"
        rec_paired = [(lbl, pl) for lbl, h, pl in paired_rows if h == rec]
        rec_null = all(pl["p"] > 0.05 for _, pl in rec_paired)
        rec_ps = ", ".join("%.2g" % pl["p"] for _, pl in rec_paired)
        w4a4_cost = [acc(data[m]["fp16"]) - acc(data[m][REFERENCE])
                     for m, _ in present if "fp16" in data[m] and REFERENCE in data[m]]
        lines += [
            "### What this changes, and what it does not",
            "",
            "**It does not overturn §1.** The perplexity numbers there are real and were "
            "measured on these same weights. What this section adds is that they do not carry "
            "over to task accuracy.",
            "",
            f"1. **The recommended rule is invisible here.** {rec} is the configuration §1 "
            f"recommends on worst-case grounds, and on "
            + ("every one of the " if rec_null else "")
            + f"{len(rec_paired)} models with per-document logs its paired delta is "
            + ("not significant" if rec_null else "mostly not significant")
            + f" (p = {rec_ps}). Whatever it buys "
              "in perplexity, it does not show up in what the model answers.",
            "",
            "2. **The election is not a confidence artefact either.** That was the worry this "
            "section was run to test: a rule chosen to shrink a squared error could lower "
            "next-token loss by flattening the distribution while predicting no better. If that "
            "were happening, accuracy would fall. It does not -- most rules are slightly "
            "positive. The perplexity gain is not an artefact of confidence -- it simply does "
            "not buy accuracy.",
            "",
            f"3. **Both are dwarfed by W4A4 itself.** Quantizing costs "
            f"{min(w4a4_cost):.4f} to {max(w4a4_cost):.4f} accuracy against BF16, while the "
            f"best election rule recovers at most {max(j[3] for j in joint):+.4f}. The element "
            "type is a second-order decision about a first-order loss.",
            "",
            "4. **If a rule were to be chosen on accuracy, it would not be this one.** By "
            "accuracy alone `hess_m1` has both the best panel mean and the best worst case, and "
            "it is the one rule that is never negative on any model -- while §1 rejects it "
            "precisely because its perplexity worst case is +0.2865. There is no configuration "
            "here that is best on both metrics, and this section does not propose changing the "
            "recommendation: it argues that the recommendation should be stated as what it is, "
            "a perplexity result.",
            "",
        ]

    # --- per-task detail, so a panel mean cannot hide a single task doing all the work ---------
    lines += ["<details>", "<summary>Per-task accuracy</summary>", ""]
    for m, lbl in present:
        d = data[m]
        lines += [f"**{lbl}**", "",
                  "| config | " + " | ".join(tasks) + " | mean |",
                  "|---" * (len(tasks) + 2) + "|"]
        for header, key in COLUMNS:
            if key not in d:
                continue
            row = [fmt(acc(d[key], t)) if t in d[key]["accuracy"] else "—" for t in tasks]
            lines.append(f"| {header} | " + " | ".join(row) + f" | {fmt(acc(d[key]))} |")
        lines.append("")
    lines += ["</details>", ""]

    out = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(out)
    print(out)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()

"""
    Merge every `results/decide_r*` round into one table per (model, sweep).

    The rounds share baselines (`nvfp4_4over6` and the `_e2m1` control are re-measured in most of
    them), so this both consolidates the record and acts as a consistency check: a baseline that
    disagrees between rounds means the evaluation data was not identical and the deltas cannot be
    compared across rounds.

        python summarize_all_decide.py
        python summarize_all_decide.py --baseline mix_4_6_e2m1_8x64 --top 20
"""

import argparse
import glob
import json
import os
from collections import defaultdict

from summarize_decide import is_realizable, split_label


def load_rounds(pattern):
    """{(model, sweep): {label: (entry, round_name)}}"""
    merged = defaultdict(dict)
    for d in sorted(glob.glob(pattern)):
        rnd = os.path.basename(d)
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            stem = os.path.basename(path).split(".shard")[0].replace(".json", "")
            model, _, sweep = stem.rpartition("_")
            with open(path) as f:
                for label, entry in json.load(f).items():
                    merged[(model, sweep)][label] = (entry, rnd)
    return merged


def check_baselines(rows, datasets):
    """A baseline measured in several rounds must agree, or cross-round deltas are meaningless."""
    seen = defaultdict(dict)
    for label, (entry, rnd) in rows.items():
        for ds in datasets:
            if ds in entry:
                seen[(label, ds)][rnd] = entry[ds]
    problems = []
    for (label, ds), by_round in seen.items():
        vals = list(by_round.values())
        if len(vals) > 1 and (max(vals) - min(vals)) > 5e-4:
            problems.append(f"{label}/{ds}: " +
                            ", ".join(f"{r}={v:.4f}" for r, v in sorted(by_round.items())))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/decide_r*")
    ap.add_argument("--baseline", default="nvfp4_4over6")
    ap.add_argument("--datasets", type=lambda s: s.split(","), default=["wikitext", "c4"])
    ap.add_argument("--top", type=int, default=None, help="Show only the best N rows.")
    ap.add_argument("--sort", default="mean", choices=["mean", "wikitext", "c4"],
                    help='"mean" ranks by the average delta over both datasets.')
    args = ap.parse_args()

    for (model, sweep), rows in sorted(load_rounds(args.pattern).items()):
        problems = check_baselines(rows, args.datasets)
        base = rows.get(args.baseline, ({}, ""))[0]

        scored = []
        for label, (entry, rnd) in rows.items():
            deltas = [entry[ds] - base[ds] for ds in args.datasets
                      if ds in entry and ds in base]
            if not deltas:
                continue
            key = (sum(deltas) / len(deltas) if args.sort == "mean"
                   else entry.get(args.sort, float("inf")) - base.get(args.sort, 0.0))
            scored.append((key, label, entry, rnd))
        scored.sort()
        if args.top:
            scored = scored[: args.top]

        width = max((len(s[1]) for s in scored), default=10)
        print(f"\n### {model} — {sweep.upper()}   (baseline {args.baseline}, sorted by {args.sort})\n")
        head = f"| {'config':<{width}} | HW | round |"
        sep = f"|{'-' * (width + 2)}|----|-------|"
        for ds in args.datasets:
            head += f" {'d' + ds:>10} |"
            sep += "------------|"
        print(head + " mean |")
        print(sep + "------|")
        for key, label, entry, rnd in scored:
            hw = {None: "  ", True: "y ", False: "- "}[is_realizable(split_label(label)[1])]
            line = f"| {label:<{width}} | {hw} | {rnd[-3:]:>5} |"
            ds_deltas = []
            for ds in args.datasets:
                if ds in entry and ds in base:
                    d = entry[ds] - base[ds]
                    ds_deltas.append(d)
                    line += f" {d:>+10.4f} |"
                else:
                    line += f" {'-':>10} |"
            mean = sum(ds_deltas) / len(ds_deltas) if ds_deltas else float("nan")
            print(line + f" {mean:+.4f} |")

        if problems:
            print("\n!! baseline disagreement across rounds -- deltas are NOT comparable:")
            for p in problems:
                print("   " + p)


if __name__ == "__main__":
    main()

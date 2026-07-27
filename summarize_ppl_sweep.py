"""
    Merge the per-shard JSON files written by `run_ppl_sweep.py` into a comparison table.

        python summarize_ppl_sweep.py results/mixfp4_sweep
"""

import argparse
import glob
import json
import os
from collections import defaultdict


# Baseline rows first, then MixFP4 variants ordered by selection metric and type block.
BASELINE_ORDER = ["fp16", "mxfp4", "nvfp4", "nvfp4_4over6", "nvif4", "razer", "razer_e3m3"]
METRIC_ORDER   = ["mix_4_6_hess_m1", "mix_4_6_hess", "mix_4_6_m1", "mix_4_6_cossim",
                  "mix_4_6_sqnr", "mix_4_6"]   # longest prefix first when matching
TB_ORDER       = ["1x16", "16x16", "256x16", "8x64", "16x64", "32x64", "32x128"]
VARIANT_ORDER  = ["mix_4_6", "mix_4_6_hess", "mix_4_6_m1", "mix_4_6_hess_m1",
                  "mix_4_6_sqnr", "mix_4_6_cossim"]


def split_label(label):
    """('mix_4_6_sqnr_8x64') -> ('mix_4_6_sqnr', '8x64'); returns (label, None) for baselines."""
    for prefix in METRIC_ORDER:
        if label.startswith(prefix + "_"):
            return prefix, label[len(prefix) + 1:]
    return label, None


def sort_key(label):
    variant, tb = split_label(label)
    if tb is None:
        idx = BASELINE_ORDER.index(label) if label in BASELINE_ORDER else len(BASELINE_ORDER)
        return (0, idx, label)
    vi = VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else 99
    return (1, vi, TB_ORDER.index(tb) if tb in TB_ORDER else 99)


def is_realizable(label):
    """A type block is expressible by one mxf4nvf4 MMA operand only if its K is at least 64."""
    _, tb = split_label(label)
    if tb is None:
        return None
    return int(tb.split("x")[1]) >= 64


def load(result_dir):
    """{(model, sweep): {label: entry}} merged across shards."""
    merged = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        base = os.path.basename(path)
        stem = base.split(".shard")[0].replace(".json", "")
        model, _, sweep = stem.rpartition("_")
        with open(path) as f:
            merged[(model, sweep)].update(json.load(f))
    return merged


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str)
    parser.add_argument("--markdown", type=str, default=None, help="Also write the report here.")
    parser.add_argument("--baseline", type=str, default="nvfp4",
                        help="Row to measure deltas against, e.g. \"nvfp4\" or \"nvfp4_4over6\".")
    args = parser.parse_args()

    merged = load(args.result_dir)
    lines = []

    for (model, sweep), entries in sorted(merged.items()):
        datasets = [d for d in ("wikitext", "c4") if any(d in e for e in entries.values())]
        labels = sorted(entries, key=sort_key)

        baseline = entries.get(args.baseline, {})
        lines.append(f"\n### {model} — {sweep.upper()} (group size 16, seq len 2048)\n")
        header = "| format | type block | " + " | ".join(f"{d} ppl" for d in datasets)
        header += " | " + " | ".join(f"d{d} vs {args.baseline}" for d in datasets) + " | HW |"
        lines.append(header)
        lines.append("|" + "---|" * (2 + 2 * len(datasets) + 1))

        for label in labels:
            e = entries[label]
            tb = split_label(label)[1] or "-"
            cells = [fmt(e.get(d)) for d in datasets]
            deltas = []
            for d in datasets:
                if label in ("fp16", args.baseline) or d not in e or d not in baseline:
                    deltas.append("-")
                else:
                    deltas.append(f"{e[d] - baseline[d]:+.4f}")
            r = is_realizable(label)
            hw = "" if r is None else ("y" if r else "-")
            lines.append(f"| {label} | {tb} | " + " | ".join(cells) + " | " +
                         " | ".join(deltas) + f" | {hw} |")

        expected = set()
        missing = sorted(expected - set(entries), key=sort_key)
        if missing:
            lines.append(f"\n_missing ({len(missing)}): {', '.join(missing)}_")

    report = "\n".join(lines)
    report += ("\n\n`HW` = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand. "
               "`-` marks MixFP4 type blocks with K < 64, which are accuracy upper bounds only.\n")
    print(report)

    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(report)
        print(f"\nwritten to {args.markdown}")


if __name__ == "__main__":
    main()

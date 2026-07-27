"""
    Merge the per-shard JSON files written by `run_ppl_sweep.py` into a comparison table.

        python summarize_ppl_sweep.py results/mixfp4_sweep
"""

import argparse
import glob
import json
import os
from collections import defaultdict


# Row order for the report. Anything not listed is appended in file order.
ROW_ORDER = [
    "fp16", "mxfp4", "nvfp4", "nvfp4_4over6", "nvif4", "razer", "razer_e3m3",
    "mixfp4_1x16", "mixfp4_16x16", "mixfp4_256x16",
    "mixfp4_8x64", "mixfp4_16x64", "mixfp4_32x64", "mixfp4_32x128",
    "mix_4_6_1x16", "mix_4_6_16x16", "mix_4_6_256x16",
    "mix_4_6_8x64", "mix_4_6_16x64", "mix_4_6_32x64", "mix_4_6_32x128",
]

# MixFP4 type blocks with K < 64 are not expressible by a single mxf4nvf4 MMA operand
NOT_REALIZABLE = {"mixfp4_1x16", "mixfp4_16x16", "mixfp4_256x16",
                  "mix_4_6_1x16", "mix_4_6_16x16", "mix_4_6_256x16"}


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
    args = parser.parse_args()

    merged = load(args.result_dir)
    lines = []

    for (model, sweep), entries in sorted(merged.items()):
        datasets = [d for d in ("wikitext", "c4") if any(d in e for e in entries.values())]
        labels = [l for l in ROW_ORDER if l in entries]
        labels += [l for l in entries if l not in ROW_ORDER]

        baseline = entries.get("nvfp4", {})
        lines.append(f"\n### {model} — {sweep.upper()} (group size 16, seq len 2048)\n")
        header = "| format | type block | " + " | ".join(f"{d} ppl" for d in datasets)
        header += " | " + " | ".join(f"d{d} vs nvfp4" for d in datasets) + " | HW |"
        lines.append(header)
        lines.append("|" + "---|" * (2 + 2 * len(datasets) + 1))

        for label in labels:
            e = entries[label]
            tb = e.get("w_type_block", "-") if e.get("w_dtype") in ("mixfp4", "mix_4_6") else "-"
            cells = [fmt(e.get(d)) for d in datasets]
            deltas = []
            for d in datasets:
                if label == "fp16" or d not in e or d not in baseline:
                    deltas.append("-")
                else:
                    deltas.append(f"{e[d] - baseline[d]:+.4f}")
            hw = "-" if label in NOT_REALIZABLE else ("y" if label.startswith(("mixfp4","mix_4_6")) else "")
            lines.append(f"| {label} | {tb} | " + " | ".join(cells) + " | " +
                         " | ".join(deltas) + f" | {hw} |")

        missing = [l for l in ROW_ORDER if l not in entries and l.startswith(('mixfp4','mix_4_6'))]
        if missing:
            lines.append(f"\n_missing: {', '.join(missing)}_")

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

"""
    Summarize a `run_ppl_sweep.py --configs ...` sweep into a table sorted by perplexity.

    Unlike `summarize_ppl_sweep.py`, this does not carry a hardcoded list of variant names: labels
    are split on the trailing "<M>x<K>" if there is one, and everything else is a baseline row.
    That is what makes it usable while the set of selection rules is still changing.

        python summarize_decide.py results/decide_r1 --baseline nvfp4_4over6
        python summarize_decide.py results/decide_r1 results/mix_4_6_sweep --sort wikitext
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict


TB_RE = re.compile(r"^(?P<variant>.+)_(?P<tb>\d+x\d+)$")


def split_label(label):
    """'mix_4_6_clipe0_m2_8x64' -> ('mix_4_6_clipe0_m2', '8x64'); baselines get tb=None."""
    m = TB_RE.match(label)
    return (m.group("variant"), m.group("tb")) if m else (label, None)


def is_realizable(tb):
    """One mxf4nvf4 MMA operand covers 64 contiguous K, so a type block needs K >= 64."""
    return None if tb is None else int(tb.split("x")[1]) >= 64


def load(result_dirs):
    """{(model, sweep): {label: entry}} merged across shards and directories."""
    merged = defaultdict(dict)
    for d in result_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            stem = os.path.basename(path).split(".shard")[0].replace(".json", "")
            model, _, sweep = stem.rpartition("_")
            with open(path) as f:
                merged[(model, sweep)].update(json.load(f))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+")
    ap.add_argument("--baseline", default="nvfp4_4over6",
                    help="Row every delta is measured against.")
    ap.add_argument("--sort", default="wikitext", choices=["wikitext", "c4", "label"])
    ap.add_argument("--datasets", type=lambda s: s.split(","), default=["wikitext", "c4"])
    ap.add_argument("--filter", default=None, help="Only rows whose label contains this substring.")
    args = ap.parse_args()

    for (model, sweep), entries in sorted(load(args.result_dirs).items()):
        base = entries.get(args.baseline, {})
        rows = []
        for label, e in entries.items():
            if args.filter and args.filter not in label:
                continue
            variant, tb = split_label(label)
            rows.append((label, variant, tb, e))

        if args.sort == "label":
            rows.sort(key=lambda r: r[0])
        else:
            # missing metric sorts last rather than crashing a partially finished sweep
            rows.sort(key=lambda r: r[3].get(args.sort, float("inf")))

        width = max((len(r[0]) for r in rows), default=10)
        print(f"\n### {model} — {sweep.upper()}   (baseline: {args.baseline}, "
              f"sorted by {args.sort})\n")
        head = f"| {'config':<{width}} | HW |"
        sep  = f"|{'-' * (width + 2)}|----|"
        for ds in args.datasets:
            head += f" {ds:>9} | {'d' + ds:>9} |"
            sep  += "-----------|-----------|"
        print(head)
        print(sep)

        for label, _variant, tb, e in rows:
            hw = {None: "  ", True: "y ", False: "- "}[is_realizable(tb)]
            line = f"| {label:<{width}} | {hw} |"
            for ds in args.datasets:
                v = e.get(ds)
                b = base.get(ds)
                line += f" {v:>9.4f} |" if v is not None else f" {'-':>9} |"
                if v is not None and b is not None:
                    line += f" {v - b:>+9.4f} |"
                else:
                    line += f" {'-':>9} |"
            print(line)

    print("\nHW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); "
          "- = accuracy upper bound only.")


if __name__ == "__main__":
    main()

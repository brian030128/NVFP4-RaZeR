"""
    Test a zero-shot accuracy delta as the paired comparison it actually is.

    Every configuration in the sweep is scored on the SAME documents, so comparing two of them
    is paired. The standard error lm-eval prints is the error of one measurement -- on a 6-task
    panel it is around 0.005, which is larger than most of the deltas the election rules produce
    and would make every one of them "not significant". That bound is the wrong one: what
    matters is not how much each configuration's accuracy varies over resampled documents, but
    how often the two configurations DISAGREE on the same document.

    With per-document outcomes (`run_zeroshot_sweep.py --samples_dir`) that is directly
    measurable. For each task,

        b = documents the reference got right and the variant got wrong
        c = documents the variant got right and the reference got wrong

    the delta is (c - b)/n and McNemar's exact test on (b, c) gives its p-value. Documents both
    get right, or both get wrong, carry no information about the difference and are exactly what
    the unpaired bound wastes its variance on.

        python analyze_zeroshot_paired.py --samples_dir results/zeroshot_w4a4/samples_qwen3-8b \
            --reference nvfp4__a-nvfp4_4over6
"""

import argparse
import glob
import json
import math
import os


def metric_of(record):
    """lm-eval logs acc_norm where a task defines it; fall back to acc."""
    for key in ("acc_norm", "acc", "exact_match"):
        if key in record and record[key] is not None:
            return key, float(record[key])
    return None, None


def load(path):
    """{task: {doc_id: correct}} for one configuration."""
    raw = json.load(open(path))
    out = {}
    for task, records in raw.items():
        per_doc = {}
        for r in records:
            _, v = metric_of(r)
            if v is not None and r.get("doc_id") is not None:
                per_doc[r["doc_id"]] = v
        if per_doc:
            out[task] = per_doc
    return out


def mcnemar_exact(b, c):
    """
        Two-sided exact McNemar: under the null the b+c discordant documents split 50/50, so
        the p-value is the binomial tail. Exact rather than chi-square because b+c is often
        small once two 4-bit configurations are compared.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(ref, var):
    rows, tb, tc, tn, tdelta_n = [], 0, 0, 0, 0.0
    for task in sorted(set(ref) & set(var)):
        docs = sorted(set(ref[task]) & set(var[task]))
        # Fractional scores (a task may score partial credit) are thresholded at the midpoint,
        # which is exact for the 0/1 accuracies these tasks report.
        b = sum(1 for d in docs if ref[task][d] > 0.5 >= var[task][d])
        c = sum(1 for d in docs if var[task][d] > 0.5 >= ref[task][d])
        n = len(docs)
        rows.append({"task": task, "n": n, "b": b, "c": c,
                     "delta": (c - b) / n if n else float("nan"),
                     "p": mcnemar_exact(b, c)})
        tb += b; tc += c; tn += n; tdelta_n += (c - b)
    pooled = {"task": "POOLED", "n": tn, "b": tb, "c": tc,
              "delta": tdelta_n / tn if tn else float("nan"),
              "p": mcnemar_exact(tb, tc)}
    return rows, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--reference", default="nvfp4__a-nvfp4_4over6")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref_path = os.path.join(args.samples_dir, f"{args.reference}.json")
    assert os.path.isfile(ref_path), f"no reference samples at {ref_path}"
    ref = load(ref_path)

    lines = ["| variant | task | n | ref-only right (b) | var-only right (c) | delta | McNemar p |",
             "|---|---|---|---|---|---|---|"]
    for path in sorted(glob.glob(os.path.join(args.samples_dir, "*.json"))):
        label = os.path.splitext(os.path.basename(path))[0]
        if label == args.reference:
            continue
        rows, pooled = compare(ref, load(path))
        for r in rows + [pooled]:
            lines.append(f"| `{label}` | {r['task']} | {r['n']} | {r['b']} | {r['c']} | "
                         f"{r['delta']:+.4f} | {r['p']:.3g} |")
    out = "\n".join(lines) + "\n"
    print(out)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(out)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()

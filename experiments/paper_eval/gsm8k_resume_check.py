"""Part M, the pause (amendment 15): may a partial configuration resume with the same batches as an uninterrupted run?

evaluation.accuracy writes a batch's records only after the whole batch is generated, and on a rerun it skips the
problems already in <out>.jsonl and sorts the rest by prompt length, as it sorts all of them in a fresh run (a stable
sort, ties in dataset order). If the records on disk are exactly the first m x 64 problems of that order, the resumed
run's batches are the uninterrupted run's batches m, m + 1, ...: the same problems, the same left padding. Greedy
decoding does not read the per-batch seed. This script checks that condition from the records alone (CPU only):

    python gsm8k_resume_check.py <model> <config> [--batch 64]   ->  prints and writes <out>.resume_check.json

Every line must parse, no problem may appear twice, and the set of recorded problems must be the order's first
m x batch entries for some m >= 0.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/dev/n16k64_campaign/paper_eval/ptq_gsm8k")
FQ2 = Path("/home/dev/n16k64_campaign/sm120runs/wt")
sys.path.insert(0, str(FQ2))


def harness_order():
    """The problems in the order evaluation.accuracy batches them (load_task, then sort by len(user_message))."""
    from evaluation.accuracy import load_task, user_message
    rows = load_task("gsm8k")
    order = [r["id"] for r in sorted(rows, key=lambda r: len(user_message(r)))]
    assert len(order) == len(set(order)) == 1319
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("config")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    js = ROOT / a.model / f"{a.config}.jsonl"
    res = dict(model=a.model, config=a.config, batch=a.batch, jsonl=str(js), exists=js.exists())
    ids, bad = [], []
    if js.exists():
        for i, line in enumerate(js.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                ids.append(json.loads(line)["id"])
            except (ValueError, KeyError):
                bad.append(i + 1)
    order = harness_order()
    m, rem = divmod(len(ids), a.batch)
    res.update(records=len(ids), unparsable_lines=bad, duplicates=len(ids) - len(set(ids)), whole_batches=m,
               partial_batch_records=rem, prefix_equal=set(ids) == set(order[:len(ids)]),
               written_in_batch_order=ids == order[:len(ids)])
    res["resume_ok"] = (not bad and res["duplicates"] == 0 and rem == 0 and res["prefix_equal"])
    out = js.with_suffix(".resume_check.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""No-GPU independent consistency reconstruction for an extracted P81 bundle."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def unescape(part):
    return part.replace("~1", "/").replace("~0", "~")


def pointer(value, raw):
    if raw in ("", "/"):
        return value
    out = value
    for part in raw.lstrip("/").split("/"):
        key = unescape(part)
        out = out[int(key)] if isinstance(out, list) else out[key]
    return out


def resolve_source(root, logical):
    if logical.startswith("parent:"):
        return root / "parent" / logical.removeprefix("parent:")
    if logical.startswith("external:"):
        return None
    return root / "campaign" / logical


def close(a, b):
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)


def find_final(root):
    paths = sorted((root / "campaign/runs").glob("P80_final_analysis_attempt*/final/FINAL_PPL_TABLE.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one packaged P80 final table, found {len(paths)}")
    return paths[0].parent


def render_csv(rows):
    fields = sorted({key for row in rows for key in row})
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def check_json_csv_pair(final, stem):
    data = load(final / f"{stem}.json")
    csv_path = final / f"{stem}.csv"
    with csv_path.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    key = "raw_rows" if stem == "FINAL_PPL_TABLE" else "flat_rows"
    expected = len(data.get(key, []))
    if len(csv_rows) != expected and stem == "FINAL_PPL_TABLE":
        expected += len(data.get("paired_dlogppl_rows", []))
    if len(csv_rows) != expected:
        raise RuntimeError(f"{stem} JSON/CSV row mismatch: {expected} != {len(csv_rows)}")
    rows = list(data.get(key, []))
    if stem == "FINAL_PPL_TABLE":
        rows += list(data.get("paired_dlogppl_rows", []))
    reconstructed = render_csv(rows)
    if csv_path.read_bytes() != reconstructed.encode("utf-8"):
        raise RuntimeError(f"{stem} CSV is not a byte-exact reconstruction of its JSON rows")
    return len(csv_rows)


def verify_pointer_rows(root, rows):
    checked = skipped = 0
    for row in rows:
        src = resolve_source(root, row.get("source_path", ""))
        if src is None or not src.is_file() or not row.get("json_pointer"):
            skipped += 1
            continue
        node = pointer(load(src), row["json_pointer"])
        if "estimate" in node:
            effect = node["estimate"]
        elif "diff" in node:
            effect = node["diff"]
        elif "estimate_draw_mean" in node:
            effect = node["estimate_draw_mean"]
        else:
            raise RuntimeError(f"pointer no longer resolves to an effect: {src}:{row['json_pointer']}")
        if not close(effect, row.get("effect", row.get("dlogppl"))):
            raise RuntimeError(f"effect mismatch: {src}:{row['json_pointer']}")
        ci = node.get("ci95", node.get("hierarchical_ci95"))
        if not (isinstance(ci, list) and len(ci) == 2 and close(ci[0], row["ci95_low"])
                and close(ci[1], row["ci95_high"])):
            raise RuntimeError(f"CI mismatch: {src}:{row['json_pointer']}")
        checked += 1
    return checked, skipped


def verify_raw_ppl(root, rows):
    checked = skipped = 0
    cache = {}
    for row in rows:
        if row.get("record_type") != "raw_ppl" or row.get("raw_ppl") is None:
            skipped += 1
            continue
        src = resolve_source(root, row.get("source_path", ""))
        if src is None or not src.is_file() or src.name != "ppl_report.json":
            skipped += 1
            continue
        report = cache.setdefault(src, load(src))
        value = report["evaluation"][row["policy"]][row["corpus"]]
        if not close(value["ppl"], row["raw_ppl"]) or not close(value["mean_nll"], row["mean_nll"]):
            raise RuntimeError(f"raw PPL mismatch: {src} {row['policy']} {row['corpus']}")
        checked += 1
    return checked, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.bundle_root).resolve()
    final = find_final(root)
    table_counts = {stem: check_json_csv_pair(final, stem) for stem in (
        "FINAL_PPL_TABLE", "CALIBRATION_ROBUSTNESS", "MATCHED_BUDGET_CONTROLS", "DOWNSTREAM_RESULTS")}
    ppl = load(final / "FINAL_PPL_TABLE.json")
    raw_checked, raw_skipped = verify_raw_ppl(root, ppl["raw_rows"])
    pointer_checked, pointer_skipped = verify_pointer_rows(root, ppl["paired_dlogppl_rows"])
    other_checked = other_skipped = 0
    for stem in ("CALIBRATION_ROBUSTNESS", "MATCHED_BUDGET_CONTROLS", "DOWNSTREAM_RESULTS"):
        checked, skipped = verify_pointer_rows(root, load(final / f"{stem}.json").get("flat_rows", []))
        other_checked += checked; other_skipped += skipped
    report = {"status": "pass", "no_gpu": True, "tables": table_counts,
              "raw_ppl_rows_checked_against_packaged_reports": raw_checked,
              "raw_ppl_rows_not_direct_report_sources": raw_skipped,
              "paired_rows_checked_by_source_json_pointer": pointer_checked + other_checked,
              "paired_rows_without_direct_pointer_source": pointer_skipped + other_skipped}
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

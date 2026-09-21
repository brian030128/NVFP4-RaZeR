"""Append and verify the immutable PROTOCOL_EXTENSION amendment chain."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


CR = Path(os.environ["CAMPAIGN_ROOT"])
SOURCE = Path(os.environ.get("CAMPAIGN_SOURCE_ROOT", Path(__file__).resolve().parents[1])).resolve()
LOG = CR / "provenance/PROTOCOL_EXTENSION_AMENDMENTS.jsonl"
FREEZE = CR / "freeze/PROTOCOL_EXTENSION.json"


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows():
    return [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]


def verify():
    previous = None
    for i, row in enumerate(rows()):
        if row.get("previous_entry_sha256") != previous:
            raise RuntimeError(f"extension amendment link mismatch at line {i + 1}")
        body = {k: v for k, v in row.items() if k != "entry_sha256"}
        if hashlib.sha256(canon(body)).hexdigest() != row.get("entry_sha256"):
            raise RuntimeError(f"extension amendment digest mismatch at line {i + 1}")
        if row.get("protocol_sha256") != sha(FREEZE):
            raise RuntimeError(f"extension amendment protocol mismatch at line {i + 1}")
        previous = row["entry_sha256"]
    return previous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, choices=("implementation_fix", "operational_resolution", "analysis_addition", "winner_freeze", "scope_note"))
    ap.add_argument("--title", required=True)
    ap.add_argument("--detail", required=True)
    ap.add_argument("--result-exposure-before", required=True)
    ap.add_argument("--files", nargs="*", default=[])
    a = ap.parse_args()
    previous = verify()
    source_files = {}
    for raw in a.files:
        p = Path(raw)
        if not p.is_absolute():
            p = SOURCE / p
        p = p.resolve()
        try:
            logical = "source:" + p.relative_to(SOURCE).as_posix()
        except ValueError:
            logical = "campaign:" + p.relative_to(CR).as_posix()
        source_files[logical] = {"sha256": sha(p), "bytes": p.stat().st_size}
    body = {
        "schema_version": "1.0", "event": a.event,
        "logged_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "previous_entry_sha256": previous, "protocol_sha256": sha(FREEZE),
        "title": a.title, "detail": a.detail,
        "result_exposure_before": a.result_exposure_before, "source_files": source_files,
    }
    body["entry_sha256"] = hashlib.sha256(canon(body)).hexdigest()
    with LOG.open("a") as f:
        f.write(json.dumps(body, sort_keys=True, ensure_ascii=True) + "\n")
        f.flush(); os.fsync(f.fileno())
    verify()
    print(json.dumps({"entry_sha256": body["entry_sha256"], "line": len(rows())}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent no-GPU verification for an extracted P81 reviewer bundle."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import struct
from pathlib import Path, PurePosixPath


# Hex encoding keeps the verifier itself free of every string it is meant to
# detect, so including this script in the deliverable cannot self-trigger.
FORBIDDEN_BYTES = tuple(bytes.fromhex(value) for value in (
    "4a6161616161415f6c", "4a414141414141", "2f686f6d652f",
    "2f7368617265332f", "6770757365727634", "627261696e5f6c",
    "626f736f6e5f6c", "70616e676368756e5f6c", "6368616f7975616e5f6c"))
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}
MAP_MAGIC = b"MIXFP4MAP/1\n"
MAP_REQUIRED = ("format", "protocol_id", "policy", "model", "type_block",
                "scale_block", "baseline", "alternative", "modules", "totals",
                "source_manifest_sha256", "calibration_manifest_sha256")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def windows_safe(name):
    if "\\" in name or ":" in name or name.startswith("/"):
        return False
    p = PurePosixPath(name)
    if any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in p.parts):
        return False
    return all(part.split(".", 1)[0].upper() not in WINDOWS_RESERVED for part in p.parts)


def jsonl_bytes(data, label):
    rows = 0
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"JSONL parse failure {label}:{line_no}: {exc}")
            rows += 1
    return rows


def amendment_chain_errors(rows, protocol_sha256):
    errors = []
    previous = None
    for position, row in enumerate(rows, 1):
        if row.get("previous_entry_sha256") != previous:
            errors.append(f"amendment link mismatch at line {position}")
        body = {key: value for key, value in row.items() if key != "entry_sha256"}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True).encode("ascii")
        digest = hashlib.sha256(encoded).hexdigest()
        if row.get("entry_sha256") != digest:
            errors.append(f"amendment entry digest mismatch at line {position}")
        if row.get("protocol_sha256") != protocol_sha256:
            errors.append(f"amendment protocol digest mismatch at line {position}")
        previous = row.get("entry_sha256")
    if not rows:
        errors.append("amendment chain is empty")
    return errors


def verify_map_bytes(data, label):
    """Parse the canonical map format without requiring PyTorch or a GPU."""
    if not data.startswith(MAP_MAGIC) or len(data) < len(MAP_MAGIC) + 8:
        raise RuntimeError(f"map magic/header prefix failure: {label}")
    (header_length,) = struct.unpack(
        "<Q", data[len(MAP_MAGIC):len(MAP_MAGIC) + 8])
    start = len(MAP_MAGIC) + 8
    stop = start + header_length
    if stop > len(data):
        raise RuntimeError(f"truncated map header: {label}")
    raw_header = data[start:stop]
    header = json.loads(raw_header.decode("ascii"))
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode("ascii")
    if canonical != raw_header:
        raise RuntimeError(f"noncanonical map header: {label}")
    missing = [key for key in MAP_REQUIRED if key not in header]
    if missing:
        raise RuntimeError(f"map header missing {missing}: {label}")
    offset = stop
    selected = 0
    for module in header["modules"]:
        rows, columns = module["grid_shape"]
        bits = int(rows) * int(columns)
        byte_count = (bits + 7) // 8
        chunk = data[offset:offset + byte_count]
        if len(chunk) != byte_count:
            raise RuntimeError(f"truncated map payload: {label}")
        if bits % 8 and chunk and chunk[-1] & ~((1 << (bits % 8)) - 1):
            raise RuntimeError(f"nonzero map padding bits: {label}/{module.get('name')}")
        count = sum(byte.bit_count() for byte in chunk)
        if count != int(module["selected"]):
            raise RuntimeError(f"map selected count mismatch: {label}/{module.get('name')}")
        selected += count
        offset += byte_count
    if offset != len(data):
        raise RuntimeError(f"trailing map payload bytes: {label}")
    if selected != int(header["totals"]["selected_tiles"]):
        raise RuntimeError(f"map total selected count mismatch: {label}")
    return header


def sample_paths(node):
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "path" and isinstance(value, str) and value.endswith(".jsonl.gz"):
                found.append(value)
            else:
                found.extend(sample_paths(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(sample_paths(value))
    return found


EMBEDDED_OWNER_RE = re.compile(
    r"(?i)[\"']?(?:owner|user|username)[\"']?\s*[:=]\s*[\"'][^\"']+[\"']"
)
EMBEDDED_UID_RE = re.compile(
    r"(?i)[\"']?(?:owner_uid|uid)[\"']?\s*[:=]\s*[\"']?\d+"
)


def embedded_identity_paths(node, pointer=""):
    """Locate unredacted owner metadata duplicated inside JSON strings."""
    hits = []
    if isinstance(node, dict):
        for key, value in node.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            hits.extend(embedded_identity_paths(value, pointer + "/" + escaped))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            hits.extend(embedded_identity_paths(value, pointer + f"/{index}"))
    elif isinstance(node, str):
        owner_match = EMBEDDED_OWNER_RE.search(node)
        if owner_match and "redacted_owner" not in owner_match.group(0).lower():
            hits.append({"pointer": pointer or "/", "kind": "embedded_owner"})
        if EMBEDDED_UID_RE.search(node):
            hits.append({"pointer": pointer or "/", "kind": "embedded_numeric_owner_uid"})
    return hits


def bundle_census_errors(actual_files, manifest_files, checksum_files):
    """Require a one-to-one census across archive, manifest, and checksums.

    The checksum file cannot checksum itself, and the evidence manifest cannot
    recursively describe itself.  Those are the only two permitted exceptions.
    """
    actual_files = set(actual_files)
    manifest_files = set(manifest_files)
    checksum_files = set(checksum_files)
    expected_checksums = manifest_files | {"EVIDENCE_BUNDLE_MANIFEST.json"}
    expected_archive = expected_checksums | {"SHA256SUMS.txt"}
    errors = []
    if checksum_files != expected_checksums:
        errors.append(
            "manifest/checksum census mismatch: "
            f"missing={sorted(expected_checksums - checksum_files)[:20]}, "
            f"extra={sorted(checksum_files - expected_checksums)[:20]}"
        )
    if actual_files != expected_archive:
        errors.append(
            "archive/checksum census mismatch: "
            f"missing={sorted(expected_archive - actual_files)[:20]}, "
            f"extra={sorted(actual_files - expected_archive)[:20]}"
        )
    return errors


def verify_run_manifest_source_links(root, by_path):
    """Bridge original run checksums to redacted packaged artifacts.

    SHA256SUMS_run.txt commits to the source bytes.  The evidence manifest
    carries both source and packaged hashes, so a reviewer can validate the
    original chain without pretending redacted JSON has its original digest.
    """
    root = Path(root)
    errors = []
    checked = 0
    for manifest_rel, manifest_row in sorted(by_path.items()):
        if not manifest_rel.endswith("/SHA256SUMS_run.txt"):
            continue
        run_prefix = PurePosixPath(manifest_rel).parent
        path = root / manifest_rel
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            parts = line.split("  ", 1)
            if (len(parts) != 2 or len(parts[0]) != 64
                    or any(char not in "0123456789abcdef" for char in parts[0])
                    or not windows_safe(parts[1])):
                errors.append(f"malformed run manifest line: {manifest_rel}:{line_no}")
                continue
            digest, child = parts
            target_rel = (run_prefix / PurePosixPath(child)).as_posix()
            target_row = by_path.get(target_rel)
            if target_row is None:  # explicitly excluded artifact or post-manifest metadata
                continue
            checked += 1
            if target_row.get("source_sha256") != digest:
                errors.append(f"run manifest/source hash mismatch: {target_rel}")

        record_rel = (run_prefix / "run_record.json").as_posix()
        record_row = by_path.get(record_rel)
        if record_row is None:
            continue
        record = json.loads((root / record_rel).read_text())
        artifacts = record.get("artifacts") or {}
        if artifacts.get("sha256_manifest") != manifest_row.get("source_sha256"):
            errors.append(f"run_record manifest reference mismatch: {record_rel}")
        result_rel = (run_prefix / "job_result.json").as_posix()
        result_row = by_path.get(result_rel)
        if (result_row is not None and artifacts.get("result_sha256")
                != result_row.get("source_sha256")):
            errors.append(f"run_record result reference mismatch: {record_rel}")
    return errors, checked


def verify_packaged_checksum_sidecars(root, by_path):
    """Verify conventional sidecars and their original-to-redacted hash bridge."""
    root = Path(root)
    errors = []
    checked = 0
    for rel, row in sorted(by_path.items()):
        if not rel.endswith(".sha256"):
            continue
        lines = [line for line in (root / rel).read_text().splitlines() if line.strip()]
        if len(lines) != 1:
            continue
        import re
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", lines[0])
        if not match:
            continue
        digest, target_name = match.groups()
        target_rel = (PurePosixPath(rel).parent / target_name).as_posix()
        target_row = by_path.get(target_rel)
        if target_row is None:
            continue
        checked += 1
        if digest != target_row.get("sha256") or sha(root / target_rel) != digest:
            errors.append(f"packaged checksum sidecar mismatch: {rel} -> {target_rel}")
        if row.get("sidecar_target") != target_rel:
            errors.append(f"checksum sidecar target metadata mismatch: {rel}")
        if row.get("sidecar_source_target_sha256") != target_row.get("source_sha256"):
            errors.append(f"checksum sidecar source bridge mismatch: {rel} -> {target_rel}")
        if row.get("sidecar_packaged_target_sha256") != target_row.get("sha256"):
            errors.append(f"checksum sidecar packaged bridge mismatch: {rel} -> {target_rel}")
    return errors, checked


def verify(root):
    root = Path(root).resolve()
    manifest_path = root / "EVIDENCE_BUNDLE_MANIFEST.json"
    sums_path = root / "SHA256SUMS.txt"
    manifest = json.loads(manifest_path.read_text())
    errors, identifier_hits, json_files, jsonl_rows, gzip_members = [], [], 0, 0, 0
    binary_identity, map_files, npz_files, lmeval_refs = 0, 0, 0, 0
    raw_entries = manifest["entries"]
    by_path = {row["bundle_relative_path"]: row for row in raw_entries}
    if len(by_path) != len(raw_entries):
        errors.append("duplicate paths in evidence manifest")
    if manifest.get("entry_count") != len(raw_entries):
        errors.append("evidence manifest entry_count mismatch")
    for rel, row in by_path.items():
        if not windows_safe(rel):
            errors.append(f"Windows-unsafe path: {rel}")
        path = root / rel
        if not path.is_file():
            errors.append(f"missing manifest entry: {rel}")
            continue
        digest = sha(path)
        if digest != row["sha256"] or path.stat().st_size != row["byte_size"]:
            errors.append(f"manifest hash/size mismatch: {rel}")
        if row.get("transform") == "byte_identical":
            binary_identity += 1
            if row.get("source_sha256") != digest:
                errors.append(f"binary source hash changed: {rel}")
        data = path.read_bytes()
        for token in FORBIDDEN_BYTES:
            if token.lower() in data.lower():
                identifier_hits.append({"path": rel, "token": token.decode("ascii", "replace")})
        if rel.endswith(".json"):
            try:
                value = json.loads(data)
                json_files += 1
                for hit in embedded_identity_paths(value):
                    identifier_hits.append({"path": rel, **hit})
                if rel.endswith("/lmeval_report.json"):
                    for sample in sample_paths(value):
                        lmeval_refs += 1
                        target = root / sample
                        if (not windows_safe(sample) or Path(sample).is_absolute()
                                or not target.resolve().is_relative_to(root)
                                or not target.is_file()):
                            errors.append(f"rebased lmeval sample does not resolve: {rel} -> {sample}")
            except Exception as exc:
                errors.append(f"JSON parse failure {rel}: {exc}")
        elif rel.endswith(".jsonl"):
            try:
                jsonl_rows += jsonl_bytes(data, rel)
                for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
                    if line.strip():
                        for hit in embedded_identity_paths(json.loads(line)):
                            identifier_hits.append({"path": rel, "line": line_no, **hit})
            except Exception as exc:
                errors.append(str(exc))
        elif rel.endswith(".jsonl.gz"):
            try:
                unpacked = gzip.decompress(data)
                gzip_members += 1
                jsonl_rows += jsonl_bytes(unpacked, rel)
                for token in FORBIDDEN_BYTES:
                    if token.lower() in unpacked.lower():
                        identifier_hits.append({"path": rel + " (decompressed)",
                                                "token": token.decode("ascii", "replace")})
            except Exception as exc:
                errors.append(f"gzip/JSONL failure {rel}: {exc}")
        elif rel.endswith(".mixfp4map"):
            map_files += 1
            try:
                verify_map_bytes(data, rel)
            except Exception as exc:
                errors.append(f"map parse failure {rel}: {exc}")
        elif rel.endswith(".npz"):
            npz_files += 1
            try:
                import numpy as np
                with np.load(path, allow_pickle=False) as archive:
                    for key in archive.files:
                        _ = archive[key].shape
            except Exception as exc:
                errors.append(f"NPZ parse failure {rel}: {exc}")

    run_link_errors, run_manifest_links = verify_run_manifest_source_links(root, by_path)
    errors.extend(run_link_errors)
    sidecar_errors, checksum_sidecars = verify_packaged_checksum_sidecars(root, by_path)
    errors.extend(sidecar_errors)

    expected_sums = {}
    for line_no, line in enumerate(sums_path.read_text().splitlines(), 1):
        parts = line.split("  ", 1)
        if (len(parts) != 2 or len(parts[0]) != 64
                or any(char not in "0123456789abcdef" for char in parts[0])):
            errors.append(f"malformed SHA256SUMS line: {line_no}")
            continue
        digest, rel = parts
        if not windows_safe(rel):
            errors.append(f"unsafe SHA256SUMS path: {rel}")
            continue
        if rel in expected_sums:
            errors.append(f"duplicate SHA256SUMS path: {rel}")
        expected_sums[rel] = digest
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    expected_files = set(expected_sums)
    errors.extend(bundle_census_errors(actual_files, by_path, expected_files))
    for rel, digest in expected_sums.items():
        path = root / rel
        if not path.is_file() or sha(path) != digest:
            errors.append(f"SHA256SUMS mismatch: {rel}")
    # Independently scan the self-describing manifest and checksum file too;
    # they cannot recursively list themselves in the evidence manifest.
    for rel in sorted(actual_files - set(by_path)):
        data = (root / rel).read_bytes()
        for token in FORBIDDEN_BYTES:
            if token.lower() in data.lower():
                identifier_hits.append({"path": rel, "token": token.decode("ascii", "replace")})

    coverage_path = root / "bundle_metadata/MATRIX_COVERAGE.json"
    try:
        coverage = json.loads(coverage_path.read_text())
        rows = coverage["rows"]
        allowed = {"complete", "stopped_by_gate", "unsupported", "blocked"}
        if (coverage.get("status") != "complete" or coverage.get("nonterminal_rows")
                or len(rows) != 28 or any(row.get("status") not in allowed for row in rows)):
            errors.append("final matrix coverage is nonterminal or malformed")
        for row in rows:
            for evidence in row.get("evidence") or []:
                if evidence.startswith("parent:"):
                    target = root / "parent" / evidence.removeprefix("parent:")
                elif "/" in evidence or "." in Path(evidence).name:
                    target = root / evidence if evidence.startswith(("campaign/", "parent/", "tools/")) \
                        or evidence in ("SHA256SUMS.txt", "EVIDENCE_BUNDLE_MANIFEST.json") \
                        else root / "campaign" / evidence
                else:
                    target = root / "campaign/runs" / evidence
                if not target.exists():
                    errors.append(f"matrix evidence does not resolve: {row.get('matrix_id')} -> {evidence}")
        with (root / "bundle_metadata/EXPERIMENT_MATRIX.csv").open(newline="") as handle:
            matrix_csv = list(csv.DictReader(handle))
        json_status = {row["matrix_id"]: row["status"] for row in rows}
        csv_status = {row["matrix_id"]: row["status"] for row in matrix_csv}
        if (len(matrix_csv) != len(rows) or len(csv_status) != len(matrix_csv)
                or csv_status != json_status):
            errors.append("EXPERIMENT_MATRIX.csv does not match MATRIX_COVERAGE.json")
    except Exception as exc:
        errors.append(f"matrix coverage validation failure: {exc!r}")
    try:
        exclusions = json.loads((root / "BUNDLE_EXCLUSIONS.json").read_text())
        if exclusions.get("status") != "complete" or exclusions.get("silent_required_claim_evidence_absent") is not False:
            errors.append("bundle exclusions do not certify complete claim evidence")
    except Exception as exc:
        errors.append(f"bundle exclusion validation failure: {exc!r}")
    try:
        amendment_path = root / "campaign/provenance/PROTOCOL_EXTENSION_AMENDMENTS.jsonl"
        amendment_rows = [json.loads(line) for line in amendment_path.read_text().splitlines()
                          if line.strip()]
        freeze_sha = sha(root / "campaign/freeze/PROTOCOL_EXTENSION.json")
        errors.extend(amendment_chain_errors(amendment_rows, freeze_sha))
    except Exception as exc:
        errors.append(f"amendment chain validation failure: {exc!r}")
    errors.extend(f"identifier exposure: {row}" for row in identifier_hits)
    return {"schema_version": "1.0", "status": "pass" if not errors else "fail",
            "root": str(root), "manifest_entries": len(by_path),
            "checksum_entries": len(expected_sums), "json_files_parsed": json_files,
            "jsonl_rows_parsed": jsonl_rows, "gzip_members_checked": gzip_members,
            "maps_parsed": map_files, "npz_archives_checked": npz_files,
            "byte_identical_entries": binary_identity,
            "run_manifest_source_links_checked": run_manifest_links,
            "checksum_sidecars_checked": checksum_sidecars,
            "lmeval_rebased_sample_references_resolved": lmeval_refs,
            "identifier_hits": identifier_hits, "errors": errors,
            "passed": not errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", nargs="?", default=".")
    args = parser.parse_args()
    report = verify(args.bundle_root)
    print(json.dumps(report, indent=1, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

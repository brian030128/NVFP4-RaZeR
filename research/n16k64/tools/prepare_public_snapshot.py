#!/usr/bin/env python3
"""Redact machine identity from copied N16K64 evidence and write source lineage.

This utility intentionally changes only path/host/account/device identifiers in text
evidence.  It never rewrites binary figures or scientific arrays.  Source roots are
provided at invocation time and are not persisted in the public index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".sha256", ".txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def origin_for(public_rel: Path, roots: dict[str, Path]) -> tuple[str, Path]:
    parts = public_rel.parts
    if parts[0] == "software":
        snapshot = parts[1]
        group = parts[2]
        remainder = Path(*parts[3:])
        source_tree = {
            "primary": "NVFP4-RaZeR-main",
            "ppl_extension": "NVFP4-RaZeR-extension-v2",
            "boundary": "NVFP4-RaZeR-main",
        }[snapshot]
        campaign = "ppl_improvement" if snapshot == "ppl_extension" else snapshot
        if group == "campaign":
            source = roots[campaign] / "source" / source_tree / "campaign" / remainder
        elif group == "support":
            source = roots[campaign] / "source" / source_tree / remainder
        else:
            raise ValueError(f"unexpected software group: {public_rel}")
        return f"campaign-source://{campaign}/{source_tree}/{group}/{remainder.as_posix()}", source

    if parts[0] != "campaigns":
        raise ValueError(f"unexpected snapshot path: {public_rel}")
    campaign = parts[1]
    remainder = Path(*parts[2:])
    root = roots[campaign]
    if campaign == "primary":
        if remainder.name in {"PROTOCOL_FREEZE.json", "PROTOCOL_FREEZE.sha256"}:
            source = root / "freeze" / remainder.name
        elif remainder.parts[0] == "analysis_ppl":
            source = root / "runs/V90_analyze_ppl_attempt5" / remainder
        elif remainder.parts[0] == "analysis_accuracy":
            source = root / "runs/V90_analyze_accuracy_attempt11" / remainder
        elif remainder.name in {
            "ARTIFACT_VALIDATION.json", "ARTIFACT_VALIDATION.md", "FAILED_OR_SKIPPED_RUNS.json"
        }:
            source = root / "runs/V83_validate_artifacts_attempt7/artifact_validation" / remainder.name
        else:
            source = root / "runs/V90_final_reports_attempt7/final" / remainder.name
    elif campaign == "ppl_improvement":
        if remainder.name == "PROTOCOL_EXTENSION.json":
            source = root / "freeze" / remainder.name
        else:
            source = root / "runs/P80_final_analysis_attempt1/final" / remainder.name
    else:
        source = root / remainder
    return f"campaign-artifact://{campaign}/{source.relative_to(root).as_posix()}", source


def discover_private_tokens(files: list[Path]) -> set[str]:
    tokens: set[str] = set()
    patterns = (
        re.compile(r"/home/([^/\s\"']+)"),
        re.compile(r"/work/([^/\s\"']+)"),
        re.compile(r"/share[123]/saves/([^/\s\"']+)"),
        re.compile(r"[\"']owner[\"']\s*:\s*[\"']([^\"']+)[\"']"),
    )
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in patterns:
            tokens.update(match.group(1) for match in pattern.finditer(text))
    return {token for token in tokens if len(token) >= 3}


def redact_text(text: str, private_tokens: set[str], device_ids: dict[str, str]) -> tuple[str, Counter]:
    counts: Counter = Counter()

    def sub(pattern: str | re.Pattern, replacement, value: str, label: str) -> str:
        updated, count = re.subn(pattern, replacement, value)
        counts[label] += count
        return updated

    text = sub(r"/home/[^/\s\"']+", "<USER_HOME>", text, "home_prefix")
    text = sub(r"/work/[^/\s\"']+", "<WORK_ROOT>", text, "work_prefix")
    text = sub(r"/share[123]/saves/[^/\s\"']+", "<ARCHIVE_ROOT>", text, "archive_prefix")
    text = sub(r"\bgpuserv[0-9]+\b", "<RESEARCH_HOST>", text, "host")
    for token in sorted(private_tokens, key=lambda item: (-len(item), item)):
        escaped = re.escape(token)
        text = sub(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", "<REDACTED_USER>", text, "account")

    def device_replacement(match: re.Match) -> str:
        counts["device_uuid"] += 1
        return device_ids[match.group(0)]

    text = re.sub(
        r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        device_replacement,
        text,
    )
    return text, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--ppl-improvement-root", type=Path, required=True)
    parser.add_argument("--mechanism-root", type=Path, required=True)
    parser.add_argument("--followup-root", type=Path, required=True)
    parser.add_argument("--boundary-root", type=Path, required=True)
    args = parser.parse_args()

    snapshot = Path(__file__).resolve().parents[1]
    roots = {
        "primary": args.primary_root.resolve(),
        "ppl_improvement": args.ppl_improvement_root.resolve(),
        "mechanism": args.mechanism_root.resolve(),
        "followup": args.followup_root.resolve(),
        "boundary": args.boundary_root.resolve(),
    }
    copied_roots = (
        snapshot / "campaigns",
        snapshot / "software/primary",
        snapshot / "software/ppl_extension",
        snapshot / "software/boundary",
    )
    copied = sorted(
        path for top in copied_roots for path in top.rglob("*") if path.is_file()
    )
    private_tokens = discover_private_tokens(copied)
    all_text = "\n".join(
        path.read_bytes().decode("utf-8", errors="ignore")
        for path in copied if path.suffix.lower() in TEXT_SUFFIXES
    )
    raw_devices = sorted(set(re.findall(
        r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        all_text,
    )))
    device_ids = {value: f"GPU-DEVICE-{index:02d}" for index, value in enumerate(raw_devices, 1)}

    original: dict[Path, tuple[str, Path, str]] = {}
    for public in copied:
        relative = public.relative_to(snapshot)
        logical, source = origin_for(relative, roots)
        if not source.is_file():
            raise FileNotFoundError(f"source missing for {relative}: {source}")
        original[public] = (logical, source, sha256(source))

    changed_counts: dict[str, dict[str, int]] = {}
    for public in copied:
        relative = public.relative_to(snapshot)
        if relative.parts[0] != "campaigns" or public.suffix.lower() not in TEXT_SUFFIXES:
            continue
        raw = public.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        redacted, counts = redact_text(text, private_tokens, device_ids)
        if redacted != text:
            public.write_bytes(redacted.encode("utf-8"))
            changed_counts[relative.as_posix()] = dict(sorted(counts.items()))

    protocol = snapshot / "campaigns/primary/PROTOCOL_FREEZE.json"
    protocol_sidecar = snapshot / "campaigns/primary/PROTOCOL_FREEZE.sha256"
    protocol_sidecar.write_text(f"{sha256(protocol)}  PROTOCOL_FREEZE.json\n", encoding="utf-8")

    entries = []
    for public in copied:
        relative = public.relative_to(snapshot)
        logical, source, source_hash = original[public]
        public_hash = sha256(public)
        if relative.as_posix() == "campaigns/primary/PROTOCOL_FREEZE.sha256":
            transform = "checksum_rebased_to_public_copy"
        elif relative.parts[0] == "software" and public_hash != source_hash:
            transform = "portable_path_plumbing_patch"
        elif public_hash != source_hash:
            transform = "text_identity_and_path_redaction"
        else:
            transform = "byte_identical"
        entries.append({
            "public_path": relative.as_posix(),
            "logical_source": logical,
            "source_sha256": source_hash,
            "public_sha256": public_hash,
            "public_bytes": public.stat().st_size,
            "transform": transform,
        })

    index = {
        "schema_version": 1,
        "scope": "copied campaign artifacts and historical software snapshots",
        "source_roots_persisted": False,
        "scientific_value_transform": "none; redaction is restricted to text path/identity fields",
        "entries": entries,
    }
    (snapshot / "SNAPSHOT_SOURCE_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    log = {
        "schema_version": 1,
        "redacted_file_count": len(changed_counts),
        "private_account_tokens_discovered": len(private_tokens),
        "device_ids_pseudonymized": len(device_ids),
        "per_file_replacement_counts": changed_counts,
    }
    (snapshot / "REDACTION_LOG.json").write_text(
        json.dumps(log, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

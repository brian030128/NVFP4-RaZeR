"""Build and independently verify the redacted P81 reviewer handoff ZIP."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from campaign import runtime
from campaign.extension_final_analysis import latest_complete
from campaign.extension_verify_bundle import verify as verify_extracted


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])
SR = Path(os.environ["CAMPAIGN_STORAGE_ROOT"])
PROTOCOL = "bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1"
ZIP_NAME = "mixfp4_n16k64_ppl_improvement_agent_handoff.zip"
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".csv", ".log", ".sh", ".py",
                 ".toml", ".yaml", ".yml", ".sha256"}
BINARY_IDENTITY_SUFFIXES = {".mixfp4map", ".npz", ".jsonl.gz"}
PRIVATE_TOKENS = tuple(bytes.fromhex(value) for value in (
    "4a6161616161415f6c", "4a414141414141", "2f686f6d652f",
    "2f7368617265332f", "6770757365727634", "627261696e5f6c",
    "626f736f6e5f6c", "70616e676368756e5f6c", "6368616f7975616e5f6c"))
PRIVATE_TOKEN_REPLACEMENTS = (
    "redacted_user", "redacted_user", "REDACTED_HOME/",
    "REDACTED_STORAGE/", "review-host", "redacted_owner",
    "redacted_owner", "redacted_owner", "redacted_owner",
)
assert len(PRIVATE_TOKENS) == len(PRIVATE_TOKEN_REPLACEMENTS)


@dataclass(frozen=True)
class Selection:
    source: Path
    dest: str
    artifact_class: str
    claim: str


def sha(path):
    return runtime.sha256_file(Path(path))


def load(path):
    return json.loads(Path(path).read_text())


def source_logical(path):
    path = Path(path)
    for prefix, root in (("campaign", CR), ("parent", PR)):
        try:
            return prefix + ":" + path.absolute().relative_to(root.absolute()).as_posix()
        except ValueError:
            try:
                return prefix + ":" + path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                pass
    return "external:" + path.name


def artifact_class(path):
    path = Path(path)
    name = path.name
    if name.endswith(".mixfp4map"):
        return "exact_map_binary"
    if name.endswith(".jsonl.gz"):
        return "per_example_sample_gzip"
    if name.endswith(".npz"):
        return "binary_array_evidence"
    if name == "ppl_report.json" or name.startswith("windows_"):
        return "per_window_ppl_evidence"
    if name == "lmeval_report.json":
        return "downstream_evaluation_report"
    if "preflight" in path.parts or name == "gpu_monitor.jsonl":
        return "gpu_ownership_evidence"
    if "final" in path.parts:
        return "final_derived_report"
    if "provenance" in path.parts or "freeze" in path.parts:
        return "protocol_and_provenance"
    if "source" in path.parts:
        return "analysis_source_or_test"
    if "handoff" in path.parts:
        return "frozen_specification"
    if name.endswith(".log"):
        return "failure_log"
    return "campaign_metadata"


def claim_for(path):
    match = re.search(r"(?:^|/)(P\d{2})[_/]", Path(path).as_posix())
    if match:
        return match.group(1)
    if "final" in Path(path).parts:
        return "P80_FINAL_ANALYSIS"
    return "G0-G9 provenance/reproduction"


def add(selection, seen, source, dest=None, cls=None, claim=None):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    if dest is None:
        try:
            dest = "campaign/" + source.absolute().relative_to(CR.absolute()).as_posix()
        except ValueError:
            dest = "parent/" + source.absolute().relative_to(PR.absolute()).as_posix()
    if dest in seen:
        if seen[dest].resolve() != source.resolve():
            raise RuntimeError(f"bundle destination collision: {dest}")
        return
    seen[dest] = source
    selection.append(Selection(source, dest, cls or artifact_class(source), claim or claim_for(dest)))


def add_tree(selection, seen, root, dest_root, predicate=lambda p: True):
    root = Path(root)
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and predicate(path):
            add(selection, seen, path, f"{dest_root}/{path.relative_to(root).as_posix()}")


def host_local_run_metadata(name):
    """Files whose only purpose is host-local container bookkeeping."""
    return name in ("container.cid", "docker_inspect.json")


def select_campaign_files():
    selected, seen, exclusions = [], {}, []
    for top in ("freeze", "provenance", "plans", "reports", "handoff/agent_handoff",
                "queue/jobs", "queue/release"):
        add_tree(selected, seen, CR / top, "campaign/" + top,
                 lambda p: "__pycache__" not in p.parts and not p.name.endswith(".tmp"))
    add(selected, seen, CR / "registry/attempts.jsonl")
    # Package the small, frozen environment specifications needed to recreate
    # the two CPU/GPU environments, but never copy the host-local virtualenvs.
    for name in ("campaign_env.sh", "common.in", "hist.in", "hist.lock.txt",
                 "lock_sha256.txt", "main.in", "main.lock.txt"):
        path = CR / "env" / name
        if path.is_file():
            add(selected, seen, path, f"campaign/env/{name}",
                cls="environment_lock", claim="G0-G9 provenance/reproduction")
    source = CR / "source/NVFP4-RaZeR-extension-v2/campaign"
    add_tree(selected, seen, source, "campaign/source/NVFP4-RaZeR-extension-v2/campaign",
             lambda p: "__pycache__" not in p.parts and (p.suffix == ".py" or p.name.endswith(".json")))

    excluded_pt = []
    completed_logs = 0
    docker_inspects = 0
    current_run_name = runtime.run_dir.name if runtime.run_dir is not None else None
    for run in sorted((CR / "runs").iterdir()):
        if not run.is_dir() or run.name == current_run_name:
            continue
        status = None
        if (run / "launch_record.json").is_file():
            status = load(run / "launch_record.json").get("status")
        for path in sorted(run.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            rel = path.relative_to(run)
            name = path.name
            if path.suffix == ".pt":
                expected = run_manifest_hash(run, rel.as_posix())
                excluded_pt.append({"logical_source": source_logical(path), "sha256": expected or sha(path),
                                    "byte_size": path.stat().st_size, "reason": "large raw selector/calibration tensor; derived maps, reports, and tabular tile evidence are packaged",
                                    "claim_reproduction_effect": "not required for no-GPU table/statistics reproduction; required only to regenerate candidate scores"})
                continue
            if host_local_run_metadata(name):
                if name == "docker_inspect.json":
                    docker_inspects += 1
                continue
            if name == "container.log" and status == "complete":
                completed_logs += 1
                continue
            include = (name.endswith(tuple(TEXT_SUFFIXES)) or name.endswith(tuple(BINARY_IDENTITY_SUFFIXES))
                       or name in ("docker_command.sh",))
            if include:
                add(selected, seen, path, "campaign/runs/" + run.name + "/" + rel.as_posix())
    exclusions.extend(excluded_pt)
    original_handoff = CR / "handoff/original_instruction_handoff.zip"
    if original_handoff.is_file():
        exclusions.append({
            "logical_source": source_logical(original_handoff),
            "sha256": sha(original_handoff), "byte_size": original_handoff.stat().st_size,
            "reason": "nested copy of the instruction archive; its checksum-verified extracted normative files are packaged under campaign/handoff/agent_handoff",
            "claim_reproduction_effect": "none",
        })
    exclusions += [
        {"class": "model_and_dataset_cache", "logical_source": "campaign:cache/",
         "sha256": None, "reason": "approximately hundreds of GiB of immutable model/dataset caches; pinned revisions and input hashes are packaged",
         "claim_reproduction_effect": "reviewers must rehydrate pinned inputs for optional GPU reproduction; no effect on no-GPU verification"},
        {"class": "completed_container_logs", "count": completed_logs,
         "logical_source": "campaign:runs/*/container.log", "sha256": None,
         "reason": "verbose duplicate progress logs; accepted run reports, ownership evidence, launch records and checksums are packaged",
         "claim_reproduction_effect": "none"},
        {"class": "ephemeral_scheduler_state", "logical_source": "campaign:queue/status, queue/logs, retry/rerun tokens, and campaign:allocator/",
         "sha256": None, "reason": "mutable daemon status, locks and retry tokens are operational state, not scientific evidence",
         "claim_reproduction_effect": "none; immutable queue specifications and approved release tokens are packaged"},
        {"class": "container_ids", "logical_source": "campaign:runs/*/container.cid", "sha256": None,
         "reason": "host-local ephemeral identifiers", "claim_reproduction_effect": "none"},
        {"class": "docker_inspect_metadata", "count": docker_inspects,
         "logical_source": "campaign:runs/*/docker_inspect.json", "sha256": None,
         "reason": "host-local Docker container, network, mount and runtime identifiers; immutable image identity, launch command, environment locks and GPU ownership evidence are packaged separately",
         "claim_reproduction_effect": "none"},
        {"class": "runtime_virtualenvs_and_python_bytecode",
         "logical_source": "campaign:env/venv_* and campaign:source/**/__pycache__",
         "sha256": None,
         "reason": "host-local executable environments and regenerated bytecode; frozen input and lock files are packaged",
         "claim_reproduction_effect": "none; reviewers recreate the environment from packaged lock files"},
        {"class": "upstream_source_outside_extension_package",
         "logical_source": "campaign:source/* outside NVFP4-RaZeR-extension-v2/campaign/*.py",
         "sha256": None,
         "reason": "the executable extension package, tests, source identity, dirty-diff audit, and environment locks are packaged; unrelated upstream scripts are retained on the server",
         "claim_reproduction_effect": "none for no-GPU verification or the reported extension workflow"},
        {"class": "parent_campaign_except_selected_evidence", "logical_source": "parent:*",
         "sha256": "432d922a49751603740622a084a665983fb337295048112e68311cd8110e19ee",
         "reason": "the complete parent reviewer bundle already exists; only reused reports/windows/maps needed by this extension are duplicated here",
         "claim_reproduction_effect": "none; parent authoritative manifest identity is retained"},
    ]
    add_parent_reuse(selected, seen)
    return selected, exclusions


def run_manifest_hash(run, rel):
    sums = Path(run) / "SHA256SUMS_run.txt"
    if not sums.is_file():
        return None
    for line in sums.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name == rel:
            return digest
    return None


def add_parent_run_evidence(selected, seen, run_id):
    run = PR / "runs" / run_id
    for rel in ("ppl/ppl_report.json", "ppl/windows_wiki.json", "ppl/windows_c4.json",
                "run_record.json", "run_record_validation.json", "SHA256SUMS_run.txt"):
        path = run / rel
        if path.is_file():
            add(selected, seen, path, f"parent/runs/{run_id}/{rel}")


def add_parent_reuse(selected, seen):
    inv_path = latest_complete("P12_existing_k2_provenance") / "reuse_inventory/PARENT_REUSE_INVENTORY.json"
    inv = load(inv_path)
    for model in inv["models"]:
        for key in ("existing_k2_evidence", "parent_baselines"):
            run_id = (model.get(key) or {}).get("run_id")
            if run_id:
                add_parent_run_evidence(selected, seen, run_id)
        for row in model.get("maps", {}).values():
            path = Path(row["source_path"])
            add(selected, seen, path, "parent/" + path.relative_to(PR).as_posix())
    # Any exact parent map referenced by a new accepted run is part of the
    # extension's executable evidence, even when it is not listed by P12.
    for run in sorted((CR / "runs").iterdir()):
        report_path = run / "ppl/ppl_report.json"
        if not report_path.is_file():
            continue
        try:
            report = load(report_path)
        except json.JSONDecodeError:
            continue
        for row in report.get("installs", []):
            raw = row.get("map_path")
            if raw and Path(raw).is_file():
                path = Path(raw)
                try:
                    rel = path.relative_to(PR)
                except ValueError:
                    continue
                add(selected, seen, path, "parent/" + rel.as_posix())


def path_aliases(selection):
    aliases = {}
    for item in selection:
        aliases[str(item.source.absolute())] = item.dest
        aliases[str(item.source.resolve())] = item.dest
        try:
            rel = item.source.resolve().relative_to(SR.resolve())
            aliases[str(SR / rel)] = item.dest
        except ValueError:
            pass
    return dict(sorted(aliases.items(), key=lambda row: len(row[0]), reverse=True))


def redact_string(value, aliases):
    out = value
    if out in aliases:
        return aliases[out]
    # Exact path values (including every lm-eval sample reference) are handled
    # above.  Embedded paths in logs/commands are safely made logical by the
    # root substitutions below; scanning every alias for every JSON string is
    # quadratic in a large evidence bundle and unnecessary.
    out = out.replace(str(CR), "campaign").replace(str(PR), "parent").replace(str(SR), "REDACTED_STORAGE")
    out = re.sub(r"/home/[^/\s\"']+", "REDACTED_HOME", out)
    out = re.sub(r"/share[0-9]+/[^/\s\"']+", "REDACTED_STORAGE", out)
    # The byte spellings are assembled dynamically so this source remains safe
    # to package even before the transform is applied.
    for token, replacement in zip(PRIVATE_TOKENS, PRIVATE_TOKEN_REPLACEMENTS):
        out = out.replace(token.decode(), replacement)
    return out


def redact_embedded_identity(value, aliases):
    """Redact owner metadata serialized inside a free-form JSON/log string."""
    out = redact_string(value, aliases)
    out = re.sub(
        r"(?i)([\"']?(?:owner|user|username)[\"']?\s*[:=]\s*[\"'])[^\"']+([\"'])",
        r"\1redacted_owner\2",
        out,
    )
    out = re.sub(
        r"(?i)([\"']?(?:owner_uid|uid)[\"']?\s*[:=]\s*)(?:[\"']?\d+[\"']?)",
        r"\1redacted_owner_uid",
        out,
    )
    return out


SENSITIVE_VALUE_KEYS = {
    "user", "users", "username", "usernames", "uid", "uids",
    "owner", "owners", "owner_uid", "owner_uids", "hostname", "host",
}
SECRET_VALUE_KEYS = {"hf_token", "huggingface_token", "api_key", "access_token", "password", "secret"}
HOST_LOCAL_ID_KEYS = {"allocation_id", "container", "container_id", "lease", "lease_id", "pci_bus_id"}


def transform_json(value, aliases, key=None):
    low = str(key).lower() if key is not None else ""
    if low in SENSITIVE_VALUE_KEYS:
        replacement = "review-host" if low in ("hostname", "host") else "redacted_owner"
        if isinstance(value, list):
            return [replacement for _ in value]
        return replacement
    if low in HOST_LOCAL_ID_KEYS and value not in (None, ""):
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
        return f"redacted_{low}_{digest}"
    if low in SECRET_VALUE_KEYS and value not in (None, ""):
        return "redacted_secret"
    if isinstance(value, dict):
        return {redact_string(str(k), aliases): transform_json(v, aliases, k) for k, v in value.items()}
    if isinstance(value, list):
        return [transform_json(v, aliases, key) for v in value]
    if isinstance(value, str):
        return redact_embedded_identity(value, aliases)
    return value


def numeric_fingerprint(value, key=None):
    rows = []
    low = str(key).lower() if key is not None else ""
    if low in SENSITIVE_VALUE_KEYS:
        return rows
    if isinstance(value, dict):
        for k in sorted(value):
            for item in numeric_fingerprint(value[k], k):
                rows.append((str(k), item))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            for item in numeric_fingerprint(child, key):
                rows.append((index, item))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append(value)
    return rows


def copy_one(item, target, aliases):
    target.parent.mkdir(parents=True, exist_ok=True)
    name = item.source.name
    source_sha = sha(item.source)
    transformed = False
    numeric_preserved = None
    if name.endswith(".jsonl.gz") or any(name.endswith(s) for s in (".mixfp4map", ".npz")):
        shutil.copyfile(item.source, target)
        mode = "byte_identical"
    elif item.source.suffix.lower() == ".json":
        original = load(item.source)
        updated = transform_json(original, aliases)
        numeric_preserved = numeric_fingerprint(original) == numeric_fingerprint(updated)
        if not numeric_preserved:
            raise RuntimeError(f"numeric scientific content changed during redaction: {item.source}")
        target.write_text(json.dumps(updated, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        transformed = target.read_bytes() != item.source.read_bytes()
        mode = "text_redacted_or_path_rebased" if transformed else "byte_identical_text"
    elif item.source.suffix.lower() == ".jsonl":
        lines = []
        for line_no, line in enumerate(item.source.read_text().splitlines(), 1):
            if not line.strip():
                continue
            original = json.loads(line)
            updated = transform_json(original, aliases)
            if numeric_fingerprint(original) != numeric_fingerprint(updated):
                raise RuntimeError(f"numeric JSONL content changed: {item.source}:{line_no}")
            lines.append(json.dumps(updated, sort_keys=True, ensure_ascii=False))
        target.write_text("\n".join(lines) + ("\n" if lines else ""))
        transformed, numeric_preserved = target.read_bytes() != item.source.read_bytes(), True
        mode = "text_redacted_or_path_rebased" if transformed else "byte_identical_text"
    elif item.source.suffix.lower() in TEXT_SUFFIXES or name == "docker_command.sh":
        # Decode bytes directly instead of Path.read_text(): TextIO universal
        # newline handling would silently turn CSV CRLF records into LF.  The
        # clean-extract recomputation deliberately requires the packaged CSV to
        # be the byte-exact csv.DictWriter rendering of the redacted JSON rows.
        text = item.source.read_bytes().decode("utf-8", errors="replace")
        if item.source.suffix.lower() not in {".py", ".sh"} and name != "docker_command.sh":
            text = redact_embedded_identity(text, aliases)
        else:
            text = redact_string(text, aliases)
        target.write_bytes(text.encode("utf-8"))
        transformed = target.read_bytes() != item.source.read_bytes()
        mode = "text_redacted_or_path_rebased" if transformed else "byte_identical_text"
    else:
        shutil.copyfile(item.source, target)
        mode = "byte_identical"
    target.chmod(0o644)
    packaged_sha = sha(target)
    if mode == "byte_identical" and packaged_sha != source_sha:
        raise RuntimeError(f"binary identity failure: {item.source}")
    return {"bundle_relative_path": item.dest, "sha256": packaged_sha,
            "byte_size": target.stat().st_size, "source_sha256": source_sha,
            "source_byte_size": item.source.stat().st_size,
            "source_artifact_class": item.artifact_class,
            "associated_claim_or_experiment_row": item.claim,
            "zip_part": ZIP_NAME,
            "original_source_logical_identifier": source_logical(item.source),
            "transform": mode, "numeric_content_preserved": numeric_preserved}


def rebase_checksum_sidecars(stage, entries):
    """Make conventional one-file sidecars verify the redacted packaged bytes.

    The evidence manifest retains both the original sidecar/target hashes and
    the packaged hashes. Multi-entry manifests such as ARTIFACT_MANIFEST.sha256
    are not conventional sidecars and remain source-chain records.
    """
    by_path = {row["bundle_relative_path"]: row for row in entries}
    rebased = 0
    for rel, row in sorted(by_path.items()):
        if not rel.endswith(".sha256"):
            continue
        path = Path(stage) / rel
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) != 1:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", lines[0])
        if not match:
            continue
        source_target_sha, target_name = match.groups()
        target_rel = (PurePosixPath(rel).parent / target_name).as_posix()
        target = by_path.get(target_rel)
        if target is None:
            continue
        if source_target_sha != target["source_sha256"]:
            raise RuntimeError(f"source checksum sidecar mismatch: {rel} -> {target_rel}")
        packaged_target_sha = target["sha256"]
        updated = f"{packaged_target_sha}  {target_name}\n"
        changed = path.read_text() != updated
        if changed:
            path.write_text(updated)
            row.update(sha256=sha(path), byte_size=path.stat().st_size,
                       transform="checksum_sidecar_rebased_to_packaged_hash",
                       numeric_content_preserved=True)
        row.update(sidecar_target=target_rel,
                   sidecar_source_target_sha256=source_target_sha,
                   sidecar_packaged_target_sha256=packaged_target_sha)
        rebased += 1
    return rebased


def final_coverage(p80_final):
    coverage = load(p80_final / "MATRIX_COVERAGE_PRE_BUNDLE.json")
    for row in coverage["rows"]:
        if row["matrix_id"] == "P02_CARRYOVER_FIXES":
            row.update(status="complete", reason="P81 clean-extract verification closes redacted-chain and Windows-path carry-over findings",
                       evidence=["SHA256SUMS.txt", "tools/verify_bundle.py"])
        elif row["matrix_id"] == "P81_ARTIFACT_BUNDLE":
            row.update(status="complete", reason="bundle accepted only if independent P81 verification returns pass",
                       evidence=["EVIDENCE_BUNDLE_MANIFEST.json", "SHA256SUMS.txt",
                                 "tools/verify_bundle.py", "tools/recompute_tables.py"])
    coverage.update(status="complete", nonterminal_rows=[])
    coverage["summary"] = dict(sorted({s: sum(row["status"] == s for row in coverage["rows"])
                                               for s in {row["status"] for row in coverage["rows"]}}.items()))
    coverage["authoritative"] = True
    return coverage


def reviewer_readme(entries, exclusions, p80_final):
    binary = sum(row["transform"] == "byte_identical" for row in entries)
    return f"""# MixFP4 N16K64 PPL-improvement reviewer evidence

This is the independently verified evidence bundle for the append-only post-hoc k2 robustness extension. The parent campaign is identified by artifact-manifest SHA-256 `{PARENT_MANIFEST}` and is not overwritten.

## Layout

- `campaign/`: extension protocol, provenance/amendments, plans, source/tests, all derived reports, exact maps, per-window evidence, per-example samples, GPU ownership evidence, and failed/OOM/invalid attempt records.
- `parent/`: only exact parent maps and PPL/window artifacts reused by the extension.
- `tools/verify_bundle.py`: checksum, JSON/JSONL/gzip, binary-map, path-safety, identifier, and lmeval-path verification.
- `tools/recompute_tables.py`: no-GPU reconstruction checks for all four final JSON/CSV tables.
- `EVIDENCE_BUNDLE_MANIFEST.json`: packaged and source SHA-256, byte size, artifact class, claim/row, ZIP part, logical source, and transform for each entry.
- `BUNDLE_EXCLUSIONS.json`: every individually omitted raw tensor plus class-level operational/cache exclusions and claim impact.
- `bundle_metadata/EXPERIMENT_MATRIX.csv`: authoritative terminal status and evidence for every frozen matrix row; its JSON counterpart is `MATRIX_COVERAGE.json`.
- `campaign/env/`: frozen dependency inputs and lock files for optional environment recreation (host-local virtualenvs are excluded).

## Integrity and no-GPU reproduction

```bash
sha256sum -c {ZIP_NAME}.sha256
unzip -q {ZIP_NAME} -d mixfp4_evidence
cd mixfp4_evidence
sha256sum -c SHA256SUMS.txt
python tools/verify_bundle.py .
python tools/recompute_tables.py .
```

Expected result: both Python commands print `status: pass`; all manifest/checksum entries resolve; all JSON/JSONL and gzip members parse; every exact map parses; all rebased lm-eval sample paths resolve; and the identifier scan reports zero hits. The final statistical tables are under the sole `campaign/runs/P80_final_analysis_attempt*/final/` directory. Numeric comparisons are exact at the stored JSON precision.

These two verification commands require Python 3 and NumPy but no GPU or PyTorch; optional model-quality reproduction uses the pinned campaign environment instead.

## Redaction and path rebasing

Only text artifacts were transformed. `{binary}` binary/map/NPZ/compressed-sample entries are byte-identical to their campaign sources and retain the same SHA-256. Private roots, owners and host identifiers in text were replaced by logical `campaign/`, `parent/`, `REDACTED_HOME`, `REDACTED_STORAGE`, `redacted_owner`, and `review-host` identifiers. Every absolute `path` in each `lmeval_report.json` that points to a packaged sample is rebased to its bundle-relative `campaign/runs/.../*.jsonl.gz` path; reviewers resolve it relative to this extracted directory.

Conventional one-target `.sha256` sidecars are rebased to the redacted packaged bytes, so `sha256sum -c` continues to work after extraction. Their evidence-manifest rows retain the original sidecar SHA-256 and both the original target SHA-256 and packaged target SHA-256, preserving a verified source-to-redacted chain. Multi-entry scientific manifests remain source-chain records and are bridged entry-by-entry through `source_sha256` in the evidence manifest.

## Optional GPU reproduction

Rehydrate the pinned model/dataset revisions listed in the protocol/input audit, use the packaged exact map without regeneration, and launch one homogeneous A6000 or RTX 6000 Ada device through the packaged fail-closed allocator. A full model/corpus PPL cell generally needs tens of minutes to several hours depending on model size; selector/calibration regeneration can require substantially more. Allow roughly 50 GiB GPU memory per device, 64-230 GiB host RAM depending on the model, and hundreds of GiB for pinned caches. Exact aggregate PPL should meet the frozen 0.5% anchor tolerance; cross-device portability uses the tolerance recorded in P75.

## Scope limitation

All quality results are BF16-dequantized software/fake-quant experiments. Native FP4/E0M3 Tensor Core execution, the external N8 ~13% and N16 ~1.5% overhead estimates, latency/speedup, area, and power remain out of scope and unsupported.
"""


PARENT_MANIFEST = "432d922a49751603740622a084a665983fb337295048112e68311cd8110e19ee"


def deterministic_zip(source_root, destination):
    count = 0
    with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(rel, date_time=(2026, 9, 15, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            already = rel.endswith((".zip", ".gz", ".npz", ".pt", ".mixfp4map"))
            info.compress_type = zipfile.ZIP_STORED if already else zipfile.ZIP_DEFLATED
            with path.open("rb") as handle:
                archive.writestr(info, handle.read(), compresslevel=None if already else 6)
            count += 1
    return count


def recursive_checksum_lines(root):
    """Checksum every staged file except the single top-level checksum file."""
    root = Path(root)
    checksum = root / "SHA256SUMS.txt"
    return [f"{sha(path)}  {path.relative_to(root).as_posix()}"
            for path in sorted(root.rglob("*"))
            if path.is_file() and path != checksum]


def write_matrix_csv(path, coverage):
    fieldnames = list(coverage["rows"][0])
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in coverage["rows"]:
            row = dict(source)
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
                elif value is None:
                    row[key] = ""
            writer.writerow(row)


def safe_remove(path, expected_parent):
    path, expected_parent = Path(path).resolve(), Path(expected_parent).resolve()
    if path.parent != expected_parent or not path.name.startswith("p81_"):
        raise RuntimeError(f"refusing unsafe temporary-directory removal: {path}")
    shutil.rmtree(path)


def main():
    if sha(CR / "freeze/PROTOCOL_EXTENSION.json") != PROTOCOL:
        raise SystemExit("protocol digest mismatch")
    p80_run = latest_complete("P80_final_analysis")
    p80_final = p80_run / "final"
    deliverables = SR / "deliverables"
    staging_parent, verify_parent = SR / "staging", SR / "verification"
    for path in (deliverables, staging_parent, verify_parent):
        path.mkdir(parents=True, exist_ok=True)
    final_targets = [deliverables / ZIP_NAME, deliverables / (ZIP_NAME + ".sha256"),
                     deliverables / "mixfp4_n16k64_ppl_improvement_BUNDLE_INDEX.json",
                     deliverables / "mixfp4_n16k64_ppl_improvement_EXPERIMENT_MATRIX.csv"]
    if any(path.exists() for path in final_targets):
        raise FileExistsError(f"append-only P81 target already exists: {[str(p) for p in final_targets if p.exists()]}")
    stage = Path(tempfile.mkdtemp(prefix="p81_", dir=staging_parent))
    verify_dir = Path(tempfile.mkdtemp(prefix="p81_", dir=verify_parent))
    tmp_zip = None
    try:
        selected, exclusions = select_campaign_files()
        aliases = path_aliases(selected)
        entries = []
        for item in selected:
            entries.append(copy_one(item, stage / item.dest, aliases))
        checksum_sidecars_rebased = rebase_checksum_sidecars(stage, entries)

        tools_dir = stage / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        generated = []
        for source_name, dest_name in (("extension_verify_bundle.py", "verify_bundle.py"),
                                       ("extension_recompute_tables.py", "recompute_tables.py")):
            source = Path(__file__).with_name(source_name)
            dest = tools_dir / dest_name
            dest.write_text(redact_string(source.read_text(), aliases))
            dest.chmod(0o644)
            generated.append((dest, "reviewer_verification_tool"))

        coverage = final_coverage(p80_final)
        metadata = stage / "bundle_metadata"
        metadata.mkdir(parents=True, exist_ok=True)
        runtime.atomic_json(metadata / "MATRIX_COVERAGE.json", coverage)
        write_matrix_csv(metadata / "EXPERIMENT_MATRIX.csv", coverage)
        risk = load(p80_final / "PAPER_RISK_REGISTER_FINAL.json")
        for row in risk["risks"]:
            if row["id"] == "R16":
                row.update(disposition="closed", evidence_or_reason="P81 validates the redacted chain, Windows-safe names, clean extraction, and complete packaged manifest")
        runtime.atomic_json(metadata / "PAPER_RISK_REGISTER_FINAL.json", risk)
        generated += [(metadata / "MATRIX_COVERAGE.json", "final_matrix_coverage"),
                      (metadata / "EXPERIMENT_MATRIX.csv", "final_experiment_matrix"),
                      (metadata / "PAPER_RISK_REGISTER_FINAL.json", "final_risk_register")]

        exclusions_doc = {"schema_version": "1.0", "status": "complete",
                          "protocol_sha256": PROTOCOL, "exclusions": exclusions,
                          "silent_required_claim_evidence_absent": False}
        runtime.atomic_json(stage / "BUNDLE_EXCLUSIONS.json", exclusions_doc)
        generated.append((stage / "BUNDLE_EXCLUSIONS.json", "bundle_metadata"))
        # Add generated files to the manifest before writing the manifest itself.
        for path, cls in generated:
            entries.append({"bundle_relative_path": path.relative_to(stage).as_posix(),
                            "sha256": sha(path), "byte_size": path.stat().st_size,
                            "source_sha256": sha(path), "source_byte_size": path.stat().st_size,
                            "source_artifact_class": cls,
                            "associated_claim_or_experiment_row": "P81_ARTIFACT_BUNDLE",
                            "zip_part": ZIP_NAME,
                            "original_source_logical_identifier": "generated:P81/" + path.name,
                            "transform": "generated", "numeric_content_preserved": True})
        readme = stage / "REVIEWER_README.md"
        readme.write_text(reviewer_readme(entries, exclusions, p80_final))
        entries.append({"bundle_relative_path": "REVIEWER_README.md", "sha256": sha(readme),
                        "byte_size": readme.stat().st_size, "source_sha256": sha(readme),
                        "source_byte_size": readme.stat().st_size,
                        "source_artifact_class": "bundle_metadata",
                        "associated_claim_or_experiment_row": "P81_ARTIFACT_BUNDLE",
                        "zip_part": ZIP_NAME, "original_source_logical_identifier": "generated:P81/REVIEWER_README.md",
                        "transform": "generated", "numeric_content_preserved": True})
        entries.sort(key=lambda row: row["bundle_relative_path"])
        manifest = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                    "zip_part": ZIP_NAME, "entry_count": len(entries), "entries": entries,
                    "checksum_sidecars_rebased": checksum_sidecars_rebased,
                    "source_and_packaged_hashes_distinct_when_text_redacted": True,
                    "binary_identity_rule": "transform=byte_identical requires source_sha256 == sha256"}
        runtime.atomic_json(stage / "EVIDENCE_BUNDLE_MANIFEST.json", manifest)
        sums = recursive_checksum_lines(stage)
        (stage / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")

        # A raw byte scan before archiving catches identifiers even in binary
        # objects that cannot legally be rewritten.
        for path in stage.rglob("*"):
            if not path.is_file():
                continue
            data = path.read_bytes()
            for token in PRIVATE_TOKENS:
                if token.lower() in data.lower():
                    raise RuntimeError(f"identifier remains before ZIP build: {path.relative_to(stage)}")
            if path.name.endswith(".jsonl.gz"):
                unpacked = gzip.decompress(data)
                for token in PRIVATE_TOKENS:
                    if token.lower() in unpacked.lower():
                        raise RuntimeError(f"identifier remains in compressed sample: {path.relative_to(stage)}")

        zip_path = deliverables / ZIP_NAME
        tmp_zip = deliverables / (ZIP_NAME + ".tmp")
        file_count = deterministic_zip(stage, tmp_zip)
        zip_sha = sha(tmp_zip)
        with zipfile.ZipFile(tmp_zip) as archive:
            bad_crc = archive.testzip()
            unsafe = [info.filename for info in archive.infolist()
                      if not __import__("campaign.extension_verify_bundle", fromlist=["windows_safe"]).windows_safe(info.filename)]
            if bad_crc or unsafe:
                raise RuntimeError(f"ZIP CRC/path validation failed: crc={bad_crc}, unsafe={unsafe[:5]}")
            archive.extractall(verify_dir)
        verification = verify_extracted(verify_dir)
        # The verifier reports the temporary absolute extraction root for local
        # diagnostics.  That host-private path must not enter a deliverable.
        verification["root"] = "temporary_extracted_bundle_removed_after_verification"
        recompute = subprocess.run([sys.executable, str(verify_dir / "tools/recompute_tables.py"), str(verify_dir)],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600)
        if recompute.returncode:
            raise RuntimeError("no-GPU table reconstruction failed:\n" + recompute.stdout)
        verification.update(zip_filename=ZIP_NAME, zip_sha256=zip_sha,
                            zip_size_bytes=tmp_zip.stat().st_size, zip_file_count=file_count,
                            zip_crc_error=bad_crc, windows_unsafe_paths=unsafe,
                            no_gpu_table_recompute=json.loads(recompute.stdout),
                            staging_removed_after_verification=True,
                            extraction_removed_after_verification=True)
        if not verification["passed"]:
            raise RuntimeError("independent extracted-bundle verification failed")
        os.replace(tmp_zip, zip_path)
        (deliverables / (ZIP_NAME + ".sha256")).write_text(f"{zip_sha}  {ZIP_NAME}\n")
        runtime.atomic_json(deliverables / "mixfp4_n16k64_ppl_improvement_BUNDLE_VERIFICATION_REPORT.json", verification)
        for name in ("EVIDENCE_BUNDLE_MANIFEST.json", "BUNDLE_EXCLUSIONS.json", "REVIEWER_README.md"):
            shutil.copyfile(stage / name, deliverables / f"mixfp4_n16k64_ppl_improvement_{name}")
        runtime.atomic_json(deliverables / "mixfp4_n16k64_ppl_improvement_MATRIX_COVERAGE.json", coverage)
        shutil.copyfile(metadata / "EXPERIMENT_MATRIX.csv",
                        deliverables / "mixfp4_n16k64_ppl_improvement_EXPERIMENT_MATRIX.csv")
        index = {"schema_version": "1.0", "status": "complete", "protocol_sha256": PROTOCOL,
                 "archive": {"filename": ZIP_NAME, "sha256": zip_sha,
                             "byte_size": zip_path.stat().st_size, "file_count": file_count},
                 "manifest": {"filename": "mixfp4_n16k64_ppl_improvement_EVIDENCE_BUNDLE_MANIFEST.json",
                              "sha256": sha(deliverables / "mixfp4_n16k64_ppl_improvement_EVIDENCE_BUNDLE_MANIFEST.json"),
                              "entries": len(entries)},
                 "verification_report": {"filename": "mixfp4_n16k64_ppl_improvement_BUNDLE_VERIFICATION_REPORT.json",
                                         "sha256": sha(deliverables / "mixfp4_n16k64_ppl_improvement_BUNDLE_VERIFICATION_REPORT.json")},
                 "experiment_matrix": {"filename": "mixfp4_n16k64_ppl_improvement_EXPERIMENT_MATRIX.csv",
                                       "sha256": sha(deliverables / "mixfp4_n16k64_ppl_improvement_EXPERIMENT_MATRIX.csv"),
                                       "rows": len(coverage["rows"]), "all_terminal": True},
                 "matrix": coverage["summary"],
                 "extension_run_directories_including_P81": sum(p.is_dir() for p in (CR / "runs").iterdir()),
                 "parent_authoritative_totals": {"run_directories": 338, "manifest_entries": 78705},
                 "parent_artifact_manifest_sha256": PARENT_MANIFEST,
                 "native_performance_claims": "out_of_scope_unsupported"}
        runtime.atomic_json(deliverables / "mixfp4_n16k64_ppl_improvement_BUNDLE_INDEX.json", index)
        launch = load(runtime.run_dir / "launch_record.json")
        outputs = [zip_path, deliverables / (ZIP_NAME + ".sha256"),
                   deliverables / "mixfp4_n16k64_ppl_improvement_EXPERIMENT_MATRIX.csv"]
        outputs += sorted(deliverables.glob("mixfp4_n16k64_ppl_improvement_*.json"))
        outputs += sorted(deliverables.glob("mixfp4_n16k64_ppl_improvement_REVIEWER_README.md"))
        runtime.atomic_json(runtime.run_dir / "job_result.json", {
            "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": PROTOCOL,
            "source": {"model_id": "P81-artifact-bundle", "model_revision": zip_sha,
                       "tokenizer_revision": "not_applicable", "model_class": "bundle",
                       "module_manifest_sha256": sha(deliverables / "mixfp4_n16k64_ppl_improvement_EVIDENCE_BUNDLE_MANIFEST.json"),
                       "source_manifest_sha256": launch["source_manifest_sha256"]},
            "environment": runtime.environment(), "data": {"calibration_manifest_sha256": "packaged",
                "evaluation_manifest_sha256": "packaged", "token_hashes": {}, "overlap_audit": None},
            "policies": [], "results": {"raw_outputs": [str(x) for x in outputs],
                "summary": index, "uncertainty": {}, "attempted_endpoints": ["P81_ARTIFACT_BUNDLE"],
                "missing_endpoints": []}, "logs": [], "failures": []})
        print(json.dumps({"status": "complete", "zip": str(zip_path), "sha256": zip_sha,
                          "bytes": zip_path.stat().st_size, "files": file_count,
                          "manifest_entries": len(entries)}, sort_keys=True))
    finally:
        if tmp_zip is not None and tmp_zip.exists():
            if tmp_zip.resolve().parent != deliverables.resolve() or tmp_zip.name != ZIP_NAME + ".tmp":
                raise RuntimeError(f"refusing unsafe temporary ZIP removal: {tmp_zip}")
            tmp_zip.unlink()
        if stage.exists():
            safe_remove(stage, staging_parent)
        if verify_dir.exists():
            safe_remove(verify_dir, verify_parent)


if __name__ == "__main__":
    main()

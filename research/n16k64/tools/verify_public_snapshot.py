#!/usr/bin/env python3
"""CPU-only integrity and consistency checks for the public N16K64 snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_exclusions(snapshot: Path) -> set[Path]:
    """Files that cannot be self-sealed without creating a hash cycle."""
    return {
        snapshot / "ARTIFACT_MANIFEST.sha256",
        snapshot / "validation/LOCAL_VALIDATION_RESULTS.json",
    }


def strict_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite {value}")),
    )


def iter_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def write_manifest(snapshot: Path) -> None:
    manifest = snapshot / "ARTIFACT_MANIFEST.sha256"
    excluded = manifest_exclusions(snapshot)
    files = sorted(
        path for path in snapshot.rglob("*")
        if path.is_file() and path not in excluded
    )
    manifest.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(snapshot).as_posix()}\n" for path in files),
        encoding="utf-8",
    )


def verify_manifest(snapshot: Path, errors: list[str], metrics: dict) -> None:
    manifest = snapshot / "ARTIFACT_MANIFEST.sha256"
    if not manifest.is_file():
        errors.append("ARTIFACT_MANIFEST.sha256 is missing")
        return
    listed = set()
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            errors.append(f"malformed manifest line {line_no}")
            continue
        expected, rel = match.groups()
        path = snapshot / rel
        listed.add(rel)
        if not path.is_file():
            errors.append(f"manifest target missing: {rel}")
        elif sha256(path) != expected:
            errors.append(f"manifest hash mismatch: {rel}")
    excluded = manifest_exclusions(snapshot)
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and path not in excluded
    }
    if listed != actual:
        errors.append(
            f"manifest membership mismatch: missing={sorted(actual-listed)}, extra={sorted(listed-actual)}"
        )
    metrics["manifest_entries"] = len(listed)
    metrics["manifest_sha256"] = sha256(manifest)


def verify_source_index(snapshot: Path, errors: list[str], metrics: dict) -> None:
    index = strict_json(snapshot / "SNAPSHOT_SOURCE_INDEX.json")
    transforms: dict[str, int] = {}
    for row in index["entries"]:
        path = snapshot / row["public_path"]
        if not path.is_file():
            errors.append(f"source-index target missing: {row['public_path']}")
            continue
        actual = sha256(path)
        if actual != row["public_sha256"]:
            errors.append(f"source-index public hash mismatch: {row['public_path']}")
        if path.stat().st_size != row["public_bytes"]:
            errors.append(f"source-index size mismatch: {row['public_path']}")
        transform = row["transform"]
        transforms[transform] = transforms.get(transform, 0) + 1
        if transform == "byte_identical" and row["source_sha256"] != row["public_sha256"]:
            errors.append(f"byte-identical entry differs: {row['public_path']}")
        if transform != "byte_identical" and row["source_sha256"] == row["public_sha256"]:
            errors.append(f"declared transform did not alter file: {row['public_path']}")
    metrics["source_index_entries"] = len(index["entries"])
    metrics["source_index_transforms"] = transforms


def parse_formats(snapshot: Path, errors: list[str], metrics: dict) -> None:
    counts = {"json": 0, "jsonl": 0, "csv": 0, "svg": 0, "png": 0, "npz": 0}
    for path in snapshot.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.suffix == ".json":
                strict_json(path)
                counts["json"] += 1
            elif path.suffix == ".jsonl":
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if line.strip():
                        json.loads(
                            line,
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                ValueError(f"non-finite {value} at line {line_no}")
                            ),
                        )
                counts["jsonl"] += 1
            elif path.suffix == ".csv":
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.reader(handle))
                if rows and any(len(row) != len(rows[0]) for row in rows[1:]):
                    errors.append(f"ragged CSV: {path.relative_to(snapshot)}")
                counts["csv"] += 1
            elif path.suffix == ".svg":
                ET.parse(path)
                counts["svg"] += 1
            elif path.suffix == ".npz":
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        errors.append(f"NPZ CRC failure: {path.relative_to(snapshot)}")
                    for name in archive.namelist():
                        raw = archive.read(name)
                        if not name.endswith('.npy') or not raw.startswith(b'\x93NUMPY'):
                            errors.append(f"unexpected NPZ member: {path.relative_to(snapshot)}")
                        if re.search(rb'/home/|/work/|/share[123]/|gpuserv[0-9]+', raw):
                            errors.append(f"private identifier in NPZ: {path.relative_to(snapshot)}")
                counts['npz'] += 1
            elif path.suffix == ".png":
                if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                    errors.append(f"invalid PNG signature: {path.relative_to(snapshot)}")
                counts["png"] += 1
        except Exception as exc:  # report every malformed artifact in one pass
            errors.append(f"parse failure {path.relative_to(snapshot)}: {exc}")
    metrics["parsed_files"] = counts


def verify_file_modes(snapshot: Path, errors: list[str], metrics: dict) -> None:
    expected_executable = {
        "software/ppl_extension/campaign/extension_recompute_tables.py",
        "software/ppl_extension/campaign/extension_verify_bundle.py",
        "tools/prepare_public_snapshot.py",
        "tools/verify_public_snapshot.py",
    }
    expected_executable = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob('*')
        if path.is_file() and path.suffix in {'.py', '.sh'} and path.read_bytes().startswith(b'#!')
    }
    actual_executable = set()
    world_writable = []
    for path in snapshot.rglob("*"):
        if not path.is_file():
            continue
        mode = path.stat().st_mode
        relative = path.relative_to(snapshot).as_posix()
        if mode & 0o111:
            actual_executable.add(relative)
        if mode & 0o002:
            world_writable.append(relative)
    if actual_executable != expected_executable:
        errors.append(
            f"executable-mode mismatch: missing={sorted(expected_executable-actual_executable)}, "
            f"extra={sorted(actual_executable-expected_executable)}"
        )
    if world_writable:
        errors.append(f"world-writable candidate files: {world_writable}")
    metrics["executable_files"] = sorted(actual_executable)
    metrics["world_writable_candidate_files"] = len(world_writable)


def verify_links(snapshot: Path, repo: Path, errors: list[str], metrics: dict) -> None:
    authored = [
        repo / "README.md",
        snapshot / "README.md",
        snapshot / "CLAIM_EVIDENCE_MATRIX.md",
        snapshot / "ARTIFACT_INVENTORY.md",
        snapshot / "COMPATIBILITY_AND_VALIDATION.md",
        snapshot / "software/README.md",
        snapshot / "validation/VALIDATION_COMMANDS.md",
        snapshot / "INTEGRATION_AUDIT.md",
        snapshot / "TODO_EXPERIMENTS.md",
    ]
    checked = 0
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for doc in authored:
        if not doc.is_file():
            errors.append(f"authored document missing: {doc.relative_to(repo)}")
            continue
        for target in pattern.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            checked += 1
            if not (doc.parent / path_part).resolve().exists():
                errors.append(f"broken link in {doc.relative_to(repo)}: {target}")
    metrics["internal_links_checked"] = checked


def verify_privacy_and_secrets(snapshot: Path, errors: list[str], metrics: dict) -> None:
    all_text_files = [
        path for path in snapshot.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in {".csv", ".json", ".jsonl", ".md", ".ps1", ".py", ".sh", ".sha256", ".toml", ".txt", ".yaml", ".yml"}
    ]
    evidence_files = [
        path for path in all_text_files
        if "software" not in path.relative_to(snapshot).parts
        and "tools" not in path.relative_to(snapshot).parts
    ]
    forbidden = {
        "absolute_home": re.compile(r"/home/[^\s\"']+"),
        "absolute_work": re.compile(r"/work/[^\s\"']+"),
        "absolute_share": re.compile(r"/share[123]/[^\s\"']+"),
        "private_host": re.compile(r"\bgpuserv[0-9]+\b", re.I),
        "github_token": re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]+"),
        "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY"),
    }
    scanned = 0
    for path in evidence_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        scanned += 1
        for label, pattern in forbidden.items():
            if pattern.search(text):
                errors.append(f"{label} found in {path.relative_to(snapshot)}")

    credential_patterns = {
        "github_token": re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
        "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY"),
        "credential_url": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
    }
    for path in all_text_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in credential_patterns.items():
            if pattern.search(text):
                errors.append(f"{label} found in {path.relative_to(snapshot)}")

    real_uuid = re.compile(
        r"GPU-([0-9a-fA-F]{8})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-([0-9a-fA-F]{4})-([0-9a-fA-F]{12})"
    )
    software_absolute_path_examples = set()
    for path in (snapshot / "software").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\bgpuserv[0-9]+\b", text, re.I):
            errors.append(f"private host literal in software: {path.relative_to(snapshot)}")
        for match in real_uuid.finditer(text):
            compact = "".join(match.groups()).lower()
            if len(set(compact)) != 1:
                errors.append(f"non-synthetic device UUID in software: {path.relative_to(snapshot)}")
        for line in text.splitlines():
            if any(prefix in line for prefix in ("/home/", "/work/", "/share1/", "/share2/", "/share3/")):
                rel = path.relative_to(snapshot).as_posix()
                if rel.endswith("software/ppl_extension/campaign/extension_bundle.py") and "re.sub" in line:
                    software_absolute_path_examples.add("redaction_regex")
                elif rel.endswith(
                    "software/ppl_extension/campaign/tests/test_extension_final_workflows.py"
                ) and "/home/private_user/result.json" in line:
                    software_absolute_path_examples.add("synthetic_test_fixture")
                else:
                    errors.append(f"unexpected absolute path in software: {rel}")
    if software_absolute_path_examples != {"redaction_regex", "synthetic_test_fixture"}:
        errors.append("expected synthetic absolute-path tests/redaction regex changed")
    metrics["privacy_secret_files_scanned"] = scanned
    metrics["credential_files_scanned"] = len(all_text_files)


def verify_scientific_invariants(snapshot: Path, errors: list[str], metrics: dict) -> None:
    checks = 0

    def require(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    primary = strict_json(snapshot / "campaigns/primary/ARTIFACT_VALIDATION.json")
    if (primary["runs"], primary["entries"]) != (338, 78705):
        errors.append("primary authoritative run/manifest totals changed")
    if primary["current_runs_with_problems"]:
        errors.append("primary current_runs_with_problems is not empty")
    if primary["artifact_manifest_sha256"] != "432d922a49751603740622a084a665983fb337295048112e68311cd8110e19ee":
        errors.append("primary artifact manifest hash changed")
    matrix = strict_json(snapshot / "campaigns/primary/MATRIX_COVERAGE.json")["summary"]
    if matrix != {"complete": 39, "missing": 0, "partial": 0}:
        errors.append("primary matrix coverage changed")
    gate = strict_json(snapshot / "campaigns/primary/decision_gate.json")
    if gate["label"] != "strong pass (quality component only; overhead out of scope)":
        errors.append("primary decision label changed")
    if not all(gate["details"]["holm_noninferior"]):
        errors.append("primary Holm gate no longer all true")

    protocol = strict_json(snapshot / "campaigns/primary/PROTOCOL_FREEZE.json")
    policy = protocol["primary_policy"]
    require(policy["type_block"] == [16, 64], "primary type-tile definition changed")
    require(policy["scale_block"] == 16, "primary scale block changed")
    require(policy["k"] == 3, "primary k changed")
    require("mean_CE + 3 SE_CE" in policy["rule"], "primary CE election rule changed")
    require("mean_KL + 3 SE_KL" in policy["rule"], "primary KL election rule changed")
    require(
        policy["calibration"] == "seed0: 64 OpenWebMath + 64 CodeParrot sequences, 512 tokens",
        "primary calibration definition changed",
    )
    require("BF16 dequantized" in protocol["scope"]["claim"], "fake-quant scope changed")

    confirmatory = strict_json(snapshot / "campaigns/primary/analysis_ppl/CONFIRMATORY_PPL.json")
    require(len(confirmatory["endpoints"]) == 6, "confirmatory PPL endpoint count changed")
    require(
        all(row["estimate"] < 0 and row["noninferior_holm"] for row in confirmatory["endpoints"]),
        "confirmatory PPL direction/Holm decision changed",
    )
    n16_minus_n8 = [
        confirmatory["models"][model]["contrasts"]["n16_k3-n8_k3"][corpus]["estimate"]
        for model in ("mistral7b", "olmo2_13b", "phi4")
        for corpus in ("wiki", "c4")
    ]
    require(all(value > 0 for value in n16_minus_n8), "N8 point-estimate comparison changed")
    require(
        all(row["macro_ci95"][0] <= 0 <= row["macro_ci95"][1] for row in gate["details"]["accuracy"].values()),
        "confirmatory macro-accuracy interval interpretation changed",
    )

    extension = strict_json(snapshot / "campaigns/ppl_improvement/ARTIFACT_VALIDATION.json")
    if not extension["passed"] or extension["attempt_count"] != 169:
        errors.append("PPL-extension audit/attempt total changed")
    extension_claims = {
        row["id"]: row["classification"]
        for row in strict_json(snapshot / "campaigns/ppl_improvement/CLAIM_REGISTER.json")["claims"]
    }
    require(extension_claims.get("C01") == "supported_post_hoc", "k=2 post-hoc classification changed")
    require(extension_claims.get("C03") == "supported", "matched-density claim changed")
    require(
        all(
            extension_claims.get(claim, "").startswith(("unsupported", "out_of_scope"))
            for claim in ("C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12")
        ),
        "unsupported extension claim was promoted",
    )
    extension_gpu = strict_json(snapshot / "campaigns/ppl_improvement/GPU_POLICY_AUDIT.json")
    require(math.isclose(extension_gpu["gpu_hours"]["total"], 89.02212256034214), "extension GPU-hours changed")
    require(extension_gpu["gpu_concurrency"]["max_concurrent_campaign_gpus"] == 3, "extension GPU concurrency changed")

    mechanism = strict_json(snapshot / "campaigns/mechanism/ARTIFACT_AUDIT.json")
    followup = strict_json(snapshot / "campaigns/followup/ARTIFACT_AUDIT.json")
    boundary = strict_json(snapshot / "campaigns/boundary/ARTIFACT_AUDIT.json")
    for name, audit in (("mechanism", mechanism), ("followup", followup), ("boundary", boundary)):
        if not audit.get("passed"):
            errors.append(f"{name} artifact audit is not passed")

    mechanism_summary = strict_json(snapshot / "campaigns/mechanism/CAMPAIGN_SUMMARY.json")
    mechanism_classes = mechanism_summary["hypothesis_classification"]
    require(mechanism_classes["ranking_not_calibration"]["passed"], "mechanism ranking gate changed")
    require(not mechanism_classes["ce_kl_veto"]["passed"], "mechanism veto gate changed")
    require(not mechanism_classes["attention_mlp_interaction"]["passed"], "mechanism interaction gate changed")
    require(
        math.isclose(mechanism_summary["gpu_hours"]["valid_complete_attempts"], 7.920919125477473),
        "mechanism valid GPU-hours changed",
    )

    followup_summary = strict_json(snapshot / "campaigns/followup/CAMPAIGN_SUMMARY.json")
    followup_classes = followup_summary["classifications"]
    require(followup_classes["dose_response"]["passed"], "follow-up dose-response gate changed")
    require(not followup_classes["objective_ablation"]["passed"], "objective-ablation conclusion changed")
    require(not followup_classes["mistral_veto_completion"]["passed"], "Mistral veto conclusion changed")
    require(
        followup_summary["attempt_accounting"]["status_counts"] == {"complete": 5, "failed": 4, "invalid": 1},
        "follow-up attempt accounting changed",
    )
    require(
        math.isclose(followup_summary["gpu_hours"]["valid_complete_attempts"], 9.07108963833915),
        "follow-up valid GPU-hours changed",
    )

    coverage = strict_json(snapshot / "campaigns/boundary/COVERAGE_GATE.json")
    if coverage["selected_B"] != 4:
        errors.append("boundary selected B changed")
    uniqueness = strict_json(snapshot / "campaigns/boundary/CORRUPTION_UNIQUENESS_GATE.json")
    if not uniqueness["passed"]:
        errors.append("corruption uniqueness gate is not passed")
    for model, row in uniqueness["models"].items():
        if len(set(row["p1_selected_set_sha256"])) != 4:
            errors.append(f"p=1 hashes are not distinct for {model}")
    summary = strict_json(snapshot / "campaigns/boundary/CAMPAIGN_SUMMARY.json")
    for family in ("boundary", "corruption"):
        if summary["classifications"][family]["classification"] != "power_limited_support":
            errors.append(f"boundary/corruption classification drift: {family}")
    if summary["time_target_met"]:
        errors.append("boundary time-target deviation was lost")
    require(summary["required_gpu_policy_evaluations"] == 168, "boundary evaluation total changed")
    require(
        summary["attempt_accounting"]["status_counts"]
        == {"complete": 7, "docker_failed": 1, "failed": 4, "invalid": 5},
        "boundary attempt accounting changed",
    )
    require(summary["wall_seconds"] > 24 * 3600, "boundary time-target disclosure changed")
    require(math.isclose(summary["gpu_hours"]["valid_complete_attempts"], 23.262747789091534), "boundary valid GPU-hours changed")
    require(math.isclose(summary["gpu_hours"]["all_attempts"], 45.03183795968691), "boundary total GPU-hours changed")

    expected_coverage = {
        "llama8b": (1480, 0.8309938236945537, 370),
        "qwen4b": (3724, 0.9134167274957071, 931),
        "mistral7b": (3912, 0.9361091170136396, 978),
    }
    for model, (retained, fraction, per_band) in expected_coverage.items():
        row = coverage["candidates"]["4"][model]
        require(row["retained_selected_tiles"] == retained, f"boundary retained tiles changed: {model}")
        require(math.isclose(row["selected_common_support_coverage"], fraction), f"boundary coverage changed: {model}")
        require(retained // 4 == per_band, f"boundary per-band tiles changed: {model}")

    boundary_gpu = strict_json(snapshot / "campaigns/boundary/GPU_POLICY_AUDIT.json")
    require(boundary_gpu["co_tenancy_attempts"] == 2, "boundary co-tenancy accounting changed")
    require(boundary_gpu["foreign_processes_signalled"] == 0, "foreign-process safety record changed")
    require(boundary_gpu["maximum_observed_concurrent_gpus"] == 3, "boundary GPU concurrency changed")

    boundary_results = strict_json(snapshot / "campaigns/boundary/BOUNDARY_RESULTS.json")
    corruption_results = strict_json(snapshot / "campaigns/boundary/CORRUPTION_RESULTS.json")
    qwen_power = {
        row["power_status"]
        for result in (boundary_results, corruption_results)
        for rows in result["holm_families"].values()
        for row in rows
        if row.get("model") == "qwen4b"
    }
    require(qwen_power == {"descriptive", "limited_inference"}, "Qwen endpoint-specific power status changed")
    p100 = {
        f"{model}/{corpus}": corruption_results["models"][model][corpus]["primary"]["p100_minus_p0"]["estimate"]
        for model in ("llama8b", "qwen4b", "mistral7b")
        for corpus in ("c4", "wiki")
    }
    require(all(value > 0 for value in p100.values()), "full-corruption direction changed")
    require(
        all(
            result["pooled_standardized"][panel]["standardized_within_model_corpus"]
            for result in (boundary_results, corruption_results)
            for panel in ("all_models", "leave_qwen4b_out")
        ),
        "pooled standardized-sensitivity label changed",
    )

    authored_text = "\n".join(
        (snapshot / name).read_text(encoding="utf-8")
        for name in ("README.md", "CLAIM_EVIDENCE_MATRIX.md", "COMPATIBILITY_AND_VALIDATION.md")
    )
    for required in (
        "power_limited_support",
        "individual-tile causal",
        "imposed stress level",
        "one-shot analysis validation",
        "hour-20 launch cutoff",
        "simultaneous-confidence",
        "Tensor Core",
    ):
        require(required in authored_text, f"required claim boundary missing from authored review: {required}")

    access = strict_json(snapshot / "validation/GITHUB_ACCESS_AUDIT.json")
    require(access["gate"].get("status") == "FAIL", "GitHub access blocker status changed")
    require(not access["candidate_branch"]["actual_push_performed"], "access audit incorrectly reports a push")
    safety = strict_json(snapshot / "validation/SNAPSHOT_SAFETY_AUDIT.json")
    require(safety["source_stability"]["source_hash_mismatches"] == 0, "source stability mismatch recorded")
    require(safety["conclusion"]["stable_snapshot_gate"] == "PASS", "snapshot stability gate changed")
    require(
        safety["conclusion"]["exclusive_global_process_visibility"] == "UNAVAILABLE",
        "host process-visibility limitation was lost",
    )

    conversion_checks = 0
    for path in (snapshot / "campaigns").rglob("*.json"):
        value = strict_json(path)
        for row in iter_dicts(value):
            if isinstance(row.get("estimate"), (int, float)) and isinstance(
                row.get("relative_ppl_change"), (int, float)
            ):
                conversion_checks += 1
                expected = math.expm1(row["estimate"])
                if not math.isclose(expected, row["relative_ppl_change"], rel_tol=1e-10, abs_tol=1e-12):
                    errors.append(f"delta-NLL conversion mismatch in {path.relative_to(snapshot)}")
                    break
    if conversion_checks == 0:
        errors.append("no delta-NLL conversion pairs were checked")
    metrics["delta_nll_relative_ppl_checks"] = conversion_checks
    metrics["scientific_claim_checks"] = checks
    metrics["boundary_p100_delta_nll"] = p100
    metrics["campaign_gpu_hours"] = {
        "ppl_improvement_all_attempts": extension_gpu["gpu_hours"]["total"],
        "mechanism_valid": mechanism_summary["gpu_hours"]["valid_complete_attempts"],
        "followup_valid": followup_summary["gpu_hours"]["valid_complete_attempts"],
        "boundary_valid": summary["gpu_hours"]["valid_complete_attempts"],
        "boundary_all_attempts": summary["gpu_hours"]["all_attempts"],
    }
    metrics["primary_runs"] = primary["runs"]
    metrics["primary_manifest_entries"] = primary["entries"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    snapshot = Path(__file__).resolve().parents[1]
    repo = snapshot.parents[1]
    if args.write_manifest:
        write_manifest(snapshot)

    errors: list[str] = []
    metrics: dict = {}
    parse_formats(snapshot, errors, metrics)
    verify_file_modes(snapshot, errors, metrics)
    verify_source_index(snapshot, errors, metrics)
    verify_links(snapshot, repo, errors, metrics)
    verify_privacy_and_secrets(snapshot, errors, metrics)
    verify_scientific_invariants(snapshot, errors, metrics)
    verify_manifest(snapshot, errors, metrics)
    report = {
        "schema_version": 1,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "metrics": metrics,
    }
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

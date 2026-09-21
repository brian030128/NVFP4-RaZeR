"""Read-only input gate for the boundary/corruption campaign."""
from __future__ import annotations

import argparse
import json
import os
import pwd
from pathlib import Path

import numpy as np

from campaign import gpu_preflight as gp
from campaign import mapio
from campaign.boundary_common import (
    EXPECTED_INPUTS,
    EXPECTED_PAIRED_ARRAYS,
    FOLLOWUP_ROOT,
    HANDOFF_ROOT,
    MECHANISM_ROOT,
    MODELS,
    OUTER_ZIP,
    PRIMARY_ROOT,
    REVISIONS,
    atomic_json,
    primary_paths,
    sha256_file,
)

EXPECTED_OUTER = "c15c66091e705dd22d29c3ba52b89bbedc557b8db0b68988041576cb2c4b69c4"
EXPECTED_SOURCE = "367667a12be8e7f91f5e748980b454f65417f2ae3dcaa49534c360072d70238a"
EXPECTED_FOLLOWUP_MANIFEST = "b658a422d6dd53ab2f2442cf8ffcf11c3a33d9beea0d9bf6047f228c0677ccef"


def verify_manifest(root: Path, manifest: Path) -> dict:
    failures = []
    count = 0
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed line {line_number}")
            continue
        path = root / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        actual = sha256_file(path)
        count += 1
        if actual != expected:
            failures.append(f"hash mismatch {relative}")
    return {"path": str(manifest), "sha256": sha256_file(manifest), "entries_checked": count,
            "failures": failures, "passed": not failures}


def gpu_inventory() -> list[dict]:
    apps_by_uuid: dict[str, list[dict]] = {}
    for app in gp.smi_compute_apps():
        try:
            uid, owner, start_ticks, cgroup = gp.proc_owner(app["pid"])
            row = dict(app, owner_uid=uid, owner=owner, start_ticks=start_ticks, cgroup=cgroup)
        except Exception as exc:
            row = dict(app, owner_uid=None, owner=None, ownership_error=repr(exc))
        apps_by_uuid.setdefault(app["gpu_uuid"], []).append(row)
    rows = []
    for gpu in gp.smi_gpus():
        processes = apps_by_uuid.get(gpu["uuid"], [])
        unresolved = any(row.get("owner_uid") is None for row in processes)
        eligible = (gpu["name"] == "NVIDIA RTX A6000" and not processes and
                    gpu["memory_used_mib"] < 1024 and not unresolved)
        rows.append(dict(gpu, compute_processes=processes, unresolved_ownership=unresolved,
                         eligible_now=eligible))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    campaign = Path(args.campaign_root).resolve()
    failures: list[str] = []

    outer_sha = sha256_file(OUTER_ZIP)
    sidecar = Path(str(OUTER_ZIP) + ".sha256")
    sidecar_digest = sidecar.read_text().split()[0] if sidecar.is_file() else None
    if outer_sha != EXPECTED_OUTER or sidecar_digest != outer_sha:
        failures.append("outer package or sidecar checksum mismatch")
    package = verify_manifest(HANDOFF_ROOT, HANDOFF_ROOT / "SHA256SUMS.txt")
    if not package["passed"]:
        failures.extend(f"handoff: {item}" for item in package["failures"])
    source_zip = HANDOFF_ROOT / "source/NVFP4-RaZeR-f692459.zip"
    source_sha = sha256_file(source_zip)
    if source_sha != EXPECTED_SOURCE:
        failures.append("clean source snapshot mismatch")

    followup_manifest = verify_manifest(FOLLOWUP_ROOT, FOLLOWUP_ROOT / "ARTIFACT_MANIFEST.sha256")
    if followup_manifest["sha256"] != EXPECTED_FOLLOWUP_MANIFEST:
        failures.append("follow-up manifest digest mismatch")
    if not followup_manifest["passed"]:
        failures.extend(f"follow-up: {item}" for item in followup_manifest["failures"])

    supplied_evidence = {}
    evidence_root = HANDOFF_ROOT / "evidence/followup_campaign"
    for path in sorted(evidence_root.iterdir()):
        if not path.is_file():
            continue
        server = FOLLOWUP_ROOT / path.name
        identical = server.is_file() and sha256_file(server) == sha256_file(path)
        supplied_evidence[path.name] = {"package_sha256": sha256_file(path),
                                        "server_path": str(server), "byte_identical": identical}
        if not identical:
            failures.append(f"packaged/server follow-up evidence mismatch: {path.name}")

    score_inputs = {}
    for model in MODELS:
        map_path, moments_path = primary_paths(model)
        map_sha, moments_sha = sha256_file(map_path), sha256_file(moments_path)
        if map_sha != EXPECTED_INPUTS[model]["map"]:
            failures.append(f"{model}: primary map digest mismatch")
        if moments_sha != EXPECTED_INPUTS[model]["moments"]:
            failures.append(f"{model}: moment digest mismatch")
        header, masks, parsed = mapio.read_map(map_path, EXPECTED_INPUTS[model]["map"])
        score_inputs[model] = {
            "primary_map": {"path": str(map_path), "sha256": map_sha, "bytes": map_path.stat().st_size,
                            "selected_tiles": header["totals"]["selected_tiles"],
                            "total_tiles": header["totals"]["total_tiles"], "modules": len(masks),
                            "parsed_sha256": parsed},
            "moments": {"path": str(moments_path), "sha256": moments_sha,
                        "bytes": moments_path.stat().st_size},
        }

    paired = {}
    expected_clusters = {"llama8b": {"c4": 231, "wiki": 53},
                         "qwen4b": {"c4": 231, "wiki": 53},
                         "mistral7b": {"c4": 235, "wiki": 57}}
    for model in MODELS:
        paired[model] = {}
        for domain in ("c4", "wiki"):
            path = FOLLOWUP_ROOT / f"arrays/{model}_{domain}_paired_cluster_nll.npz"
            digest = sha256_file(path)
            if digest != EXPECTED_PAIRED_ARRAYS[f"{model}_{domain}"]:
                failures.append(f"{model}/{domain}: paired array digest mismatch")
            with np.load(path) as payload:
                clusters = len(payload["cluster_ids"])
                tokens = int(payload["cluster_tokens"].sum())
                policies = [str(v) for v in payload["policies"]]
                manifest_sha = str(payload["evaluation_manifest_sha256"])
            if clusters != expected_clusters[model][domain]:
                failures.append(f"{model}/{domain}: natural cluster count mismatch")
            paired[model][domain] = {"path": str(path), "sha256": digest, "clusters": clusters,
                                     "tokens": tokens, "policies": policies,
                                     "evaluation_manifest_sha256": manifest_sha}

    caches = {}
    for model, (repo, revision) in REVISIONS.items():
        path = campaign / "cache/hf/hub" / f"models--{repo}" / "snapshots" / revision
        present = path.is_dir()
        caches[model] = {"path": str(path), "revision": revision, "present": present}
        if not present:
            failures.append(f"missing pinned model cache: {model}")
    for label, repo, revision, relative in (
        ("wiki", "Salesforce--wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-2-raw-v1/test-00000-of-00001.parquet"),
        ("c4", "allenai--c4", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2", "en/c4-validation.00000-of-00008.json.gz"),
    ):
        path = campaign / "cache/hf/hub" / f"datasets--{repo}" / "snapshots" / revision / relative
        caches[label] = {"path": str(path), "revision": revision, "present": path.is_file()}
        if not path.is_file():
            failures.append(f"missing pinned dataset cache: {label}")

    inventory = gpu_inventory()
    eligible = [row["uuid"] for row in inventory if row["eligible_now"]]
    if not eligible:
        failures.append("no uncontended RTX A6000 at input gate")

    process_rows = []
    uid = os.getuid()
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            status = item.stat()
            if status.st_uid != uid:
                continue
            cmd = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(token in cmd.lower() for token in ("mixfp4", "boundary", "corruption", "campaign")):
            process_rows.append({"pid": int(item.name), "owner": pwd.getpwuid(uid).pw_name, "command": cmd[:2000]})

    result = {
        "schema": "mixfp4-n16k64-boundary-corruption-input-provenance/v1",
        "campaign_root": str(campaign),
        "logical_root": str(Path(args.campaign_root)),
        "append_only": True,
        "handoff": {"outer_zip": str(OUTER_ZIP), "outer_sha256": outer_sha,
                    "expected_outer_sha256": EXPECTED_OUTER, "sidecar_matches": sidecar_digest == outer_sha,
                    "read_only_root": str(HANDOFF_ROOT), "internal_verification": package,
                    "powershell_runtime_available": False,
                    "verification_note": "The supplied PowerShell verifier was reviewed. Because pwsh is unavailable, its exact splitlines/SHA-256 logic was executed on Linux without changing the CRLF manifest."},
        "source": {"clean_snapshot": str(source_zip), "sha256": source_sha,
                   "declared_commit": "f692459195beb437a7093ab878472bbebcbe63c3",
                   "working_source": str(campaign / "source/NVFP4-RaZeR-main"),
                   "lineage": "byte copy of the independently verified completed follow-up source scaffold, followed only by campaign-local boundary/corruption additions"},
        "followup_campaign": {"logical_root": str(FOLLOWUP_ROOT), "physical_root": str(FOLLOWUP_ROOT.resolve()),
                              "manifest": followup_manifest, "supplied_evidence": supplied_evidence},
        "read_only_parents": {"primary": str(PRIMARY_ROOT), "mechanism": str(MECHANISM_ROOT),
                              "followup": str(FOLLOWUP_ROOT)},
        "score_and_anchor_inputs": score_inputs,
        "score_regeneration_required": False,
        "paired_power_inputs": paired,
        "caches": caches,
        "gpu_inventory": inventory,
        "eligible_a6000_uuids": eligible,
        "maximum_immediately_eligible_a6000": min(3, len(eligible)),
        "ownership_scan": {"matching_same_user_processes": process_rows,
                           "decision": "no prior boundary/corruption writer detected; current Codex/diagnostic processes are not campaign writers"},
        "gate_passed": not failures,
        "failures": failures,
    }
    atomic_json(args.out, result)
    gate = {"schema": "mixfp4-boundary-corruption-input-gate/v1", "passed": not failures,
            "input_provenance": {"path": str(Path(args.out)), "sha256": sha256_file(args.out)},
            "checks": {"outer_and_internal_package": package["passed"] and outer_sha == EXPECTED_OUTER,
                       "followup_manifest_entries": followup_manifest["entries_checked"],
                       "followup_manifest_passed": followup_manifest["passed"],
                       "score_moments_and_maps": not any("map digest" in f or "moment digest" in f for f in failures),
                       "paired_arrays": not any("paired array" in f or "cluster count" in f for f in failures),
                       "pinned_caches": all(row["present"] for row in caches.values()),
                       "uncontended_a6000_available": bool(eligible)},
            "failures": failures}
    atomic_json(campaign / "INPUT_GATE.json", gate)
    print(json.dumps({"passed": not failures, "failures": failures, "eligible_a6000": eligible,
                      "input_provenance_sha256": sha256_file(args.out)}, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

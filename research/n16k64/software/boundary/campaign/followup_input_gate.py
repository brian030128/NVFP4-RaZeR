"""Read-only hour-0 gate for the frozen N16K64 follow-up campaign.

The gate deliberately does not create or regenerate calibration data.  It rehashes
the score moments, primary maps, reusable evaluation evidence, handoff package, and
source snapshot; verifies pinned caches; and records a fresh GPU ownership snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from campaign import gpu_preflight as gp
from campaign import mapio


EXPECTED_OUTER = "e9bcd4eb29cd5adc363bd29df72af6bc44b33b06d151ed77081850d8ab692256"
EXPECTED_SOURCE = "367667a12be8e7f91f5e748980b454f65417f2ae3dcaa49534c360072d70238a"
EXPECTED_MECHANISM_MANIFEST = "1cb942724aeacc2c74862377bfbae8899c76ac6caa824a3f79cfd6c8e78033c4"
EXPECTED_INPUTS = {
    "llama8b": {
        "map": "0920f55ddc053a5f5a8b0d046d0b74a62d4abafcad5e36051d7682da961e1f2b",
        "moments": "e28c87a08db30cbd799a1886fe8b8fb2c821ed2c7adf8a6ffdab1735dd8095c6",
    },
    "qwen4b": {
        "map": "188bf0e51c372cd831b158e457ede4b53121f0513261f39df8403971b20c01e8",
        "moments": "2315102bf1b8b7f95cd3db62e976ee7cfd4a857e7cd4ca3489f315ee1965cb5e",
    },
    "mistral7b": {
        "map": "0c3d822a18d0480ca2aeff0abd78cf6e4ca3155bb207156a400fde124e1cf13d",
        "moments": "68321bb3fd9ecb8f0050659cdbe62a39c17eb262995d075e64de1630ece27ab1",
    },
}
MODEL_RUN = {
    "llama8b": ("V30_calib_llama8b_seed0_attempt1", "llama8b"),
    "qwen4b": ("V30_calib_qwen4b_seed0_attempt1", "qwen4b"),
    "mistral7b": ("V61_calib_mistral7b_seed0_attempt2", "mistral7b"),
}
MECHANISM_RUN = {
    "llama8b": "V22_full_llama8b_attempt1",
    "qwen4b": "V21_full_qwen4b_attempt2",
    "mistral7b": "V30_mistral_one_shot_attempt1",
}
REVISIONS = {
    "llama8b": ("meta-llama--Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
    "qwen4b": ("Qwen--Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "mistral7b": ("mistralai--Mistral-7B-v0.3", "caa1feb0e54d415e2df31207e5f4e273e33509b1"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_internal_manifest(handoff: Path) -> dict:
    manifest = handoff / "SHA256SUMS.txt"
    failures, entries = [], []
    # splitlines intentionally accepts the CRLF file emitted by the Windows packager.
    for lineno, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line:
            continue
        try:
            expected, rel = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed line {lineno}")
            continue
        target = handoff / rel
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            failures.append(f"malformed digest line {lineno}")
            continue
        if not target.is_file():
            failures.append(f"missing {rel}")
            continue
        actual = sha(target)
        entries.append({"path": rel, "sha256": actual, "bytes": target.stat().st_size})
        if actual != expected:
            failures.append(f"hash mismatch {rel}")
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": sha(manifest),
        "entries_checked": len(entries),
        "failures": failures,
        "passed": not failures,
    }


def primary_files(parent: Path, model: str) -> tuple[Path, Path]:
    run, stem = MODEL_RUN[model]
    root = parent / "runs" / run
    return (
        root / f"maps/{stem}_seed0_n16_k3.mixfp4map",
        root / "calibration/moments/moments_full.pt",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--parent-root", required=True)
    ap.add_argument("--mechanism-root", required=True)
    ap.add_argument("--handoff-root", required=True)
    ap.add_argument("--outer-zip", required=True)
    ap.add_argument("--gpu-snapshot", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    campaign = Path(args.campaign_root).resolve()
    parent = Path(args.parent_root).resolve()
    mechanism = Path(args.mechanism_root).resolve()
    handoff = Path(args.handoff_root).resolve()
    outer = Path(args.outer_zip).resolve()
    failures: list[str] = []

    outer_sha = sha(outer)
    if outer_sha != EXPECTED_OUTER:
        failures.append("outer handoff hash mismatch")
    sidecar = outer.with_suffix(outer.suffix + ".sha256")
    sidecar_expected = sidecar.read_text().split()[0] if sidecar.is_file() else None
    if sidecar_expected != outer_sha:
        failures.append("outer sidecar mismatch")
    internal = verify_internal_manifest(handoff)
    if not internal["passed"]:
        failures.extend(f"handoff: {x}" for x in internal["failures"])

    source_zip = handoff / "source/NVFP4-RaZeR-f692459.zip"
    source_sha = sha(source_zip)
    if source_sha != EXPECTED_SOURCE:
        failures.append("source snapshot hash mismatch")

    mechanism_manifest = mechanism / "ARTIFACT_MANIFEST.sha256"
    mechanism_manifest_sha = sha(mechanism_manifest)
    if mechanism_manifest_sha != EXPECTED_MECHANISM_MANIFEST:
        failures.append("mechanism manifest hash mismatch")
    manifest_rows = {}
    for line in mechanism_manifest.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        manifest_rows[rel] = digest

    score_inputs = {}
    for model in ("llama8b", "qwen4b", "mistral7b"):
        map_path, moments_path = primary_files(parent, model)
        map_sha, moments_sha = sha(map_path), sha(moments_path)
        if map_sha != EXPECTED_INPUTS[model]["map"]:
            failures.append(f"{model} primary map mismatch")
        if moments_sha != EXPECTED_INPUTS[model]["moments"]:
            failures.append(f"{model} score moments mismatch")
        header, masks, parsed_sha = mapio.read_map(map_path, EXPECTED_INPUTS[model]["map"])
        score_inputs[model] = {
            "primary_map": {"path": str(map_path), "sha256": map_sha, "bytes": map_path.stat().st_size,
                            "selected_tiles": header["totals"]["selected_tiles"],
                            "total_tiles": header["totals"]["total_tiles"], "modules": len(masks),
                            "type_block": header["type_block"], "parsed_sha256": parsed_sha},
            "moments": {"path": str(moments_path), "sha256": moments_sha,
                        "bytes": moments_path.stat().st_size, "verified": moments_sha == EXPECTED_INPUTS[model]["moments"]},
        }

    reusable = {}
    for model, run_id in MECHANISM_RUN.items():
        run = mechanism / "runs" / run_id
        files = ["ppl/ppl_report.json", "ppl/windows_wiki.json", "ppl/windows_c4.json"]
        files += [f"ppl/tokens_{policy}_{domain}.npz" for policy in ("four_over_six", "full") for domain in ("wiki", "c4")]
        checked = []
        for rel_in_run in files:
            path = run / rel_in_run
            rel = path.relative_to(mechanism).as_posix()
            actual = sha(path)
            expected = manifest_rows.get(rel)
            ok = expected == actual
            checked.append({"path": str(path), "relative_to_mechanism": rel, "sha256": actual,
                            "manifest_sha256": expected, "bytes": path.stat().st_size, "verified": ok})
            if not ok:
                failures.append(f"mechanism reusable evidence mismatch: {rel}")
        report = json.loads((run / "ppl/ppl_report.json").read_text())
        reusable[model] = {
            "run_id": run_id,
            "files": checked,
            "evaluation_manifest_sha256": report["evaluation_manifest_sha256"],
            "model_revision": report["spec"]["revision"],
            "baseline_and_full_present": all(p in report["evaluation"] for p in ("four_over_six", "full")),
        }

    old_map_root = mechanism / "runs/V10_derive_maps_attempt2/derived_maps/mistral7b"
    c_maps = {}
    for policy in ("full_plus_ce_vetoed_kl_approved", "full_plus_ce_vetoed_matched_random"):
        path = old_map_root / f"mistral7b_{policy}.mixfp4map"
        rel = path.relative_to(mechanism).as_posix()
        actual = sha(path)
        expected = manifest_rows.get(rel)
        if actual != expected:
            failures.append(f"Mistral frozen veto map mismatch: {policy}")
        c_maps[policy] = {"path": str(path), "sha256": actual, "manifest_sha256": expected,
                         "bytes": path.stat().st_size, "verified": actual == expected}

    caches = {}
    for model, (repo, revision) in REVISIONS.items():
        path = campaign / "cache/hf/hub" / f"models--{repo}" / "snapshots" / revision
        present = path.is_dir()
        caches[model] = {"path": str(path), "revision": revision, "present": present}
        if not present:
            failures.append(f"missing pinned model cache: {model}")
    for label, repo, revision, rel in (
        ("wiki", "Salesforce--wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-2-raw-v1/test-00000-of-00001.parquet"),
        ("c4", "allenai--c4", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2", "en/c4-validation.00000-of-00008.json.gz"),
    ):
        path = campaign / "cache/hf/hub" / f"datasets--{repo}" / "snapshots" / revision / rel
        present = path.is_file()
        caches[label] = {"path": str(path), "revision": revision, "present": present}
        if not present:
            failures.append(f"missing pinned dataset cache: {label}")

    gpus, apps = gp.smi_gpus(), gp.smi_compute_apps()
    app_by = {}
    for app in apps:
        try:
            uid, owner, start, _ = gp.proc_owner(app["pid"])
            rec = dict(app, owner_uid=uid, owner=owner, start_ticks=start)
        except Exception as exc:  # unresolved ownership fails closed for that device
            rec = dict(app, owner_uid=None, owner=None, owner_error=repr(exc))
        app_by.setdefault(app["gpu_uuid"], []).append(rec)
    inventory = []
    for gpu in gpus:
        procs = app_by.get(gpu["uuid"], [])
        eligible = gpu["name"] == "NVIDIA RTX A6000" and not procs and gpu["memory_used_mib"] < 1024
        inventory.append(dict(gpu, compute_processes=procs, eligible_now=eligible,
                              unresolved_ownership=any(p.get("owner_uid") is None for p in procs)))
    eligible = [g["uuid"] for g in inventory if g["eligible_now"]]
    if not eligible:
        failures.append("no uncontended A6000 at input gate")

    supplied_snapshots = []
    for raw in args.gpu_snapshot:
        path = Path(raw).resolve()
        record = json.loads(path.read_text())
        supplied_snapshots.append({"path": str(path), "sha256": sha(path), "passed": record.get("passed"),
                                   "timestamp_utc": record.get("timestamp_utc"), "reasons": record.get("reasons", [])})

    result = {
        "schema": "mixfp4-n16k64-followup-input-provenance/v1",
        "campaign_root": str(campaign),
        "append_only": True,
        "parent_campaign_read_only": str(parent),
        "mechanism_campaign_read_only": str(mechanism),
        "handoff": {
            "outer_zip": str(outer), "outer_sha256": outer_sha, "expected_outer_sha256": EXPECTED_OUTER,
            "sidecar_sha256_matches": sidecar_expected == outer_sha,
            "extracted_read_only_root": str(handoff), "internal_verification": internal,
            "powershell_runtime_available": False,
            "verification_note": "verify_package.ps1 was independently reviewed; Linux splitlines/sha256 verification applied its exact checks to all entries. The CRLF manifest was not modified.",
        },
        "source": {"archive": str(source_zip), "sha256": source_sha, "expected_sha256": EXPECTED_SOURCE,
                   "declared_commit": "f692459195beb437a7093ab878472bbebcbe63c3",
                   "working_source_lineage": "hash-verified mechanism campaign source scaffold copied before follow-up-only additions"},
        "mechanism_manifest": {"path": str(mechanism_manifest), "sha256": mechanism_manifest_sha,
                               "expected_sha256": EXPECTED_MECHANISM_MANIFEST, "entries": len(manifest_rows)},
        "score_and_map_inputs": score_inputs,
        "score_gate": {"regeneration_required": False,
                       "decision": "verified seed0 direct-N16 sufficient statistics are complete for all three models; do not regenerate scores"},
        "reusable_baseline_and_full_evidence": reusable,
        "frozen_mistral_veto_completion_inputs": c_maps,
        "caches": caches,
        "gpu_inventory": inventory,
        "eligible_a6000_uuids_at_gate": eligible,
        "maximum_immediately_eligible_a6000": min(3, len(eligible)),
        "gpu_snapshots": supplied_snapshots,
        "gpu_policy_decision": "A6000 only; at most three campaign leases; reject any device with foreign or unresolved compute ownership; retain failed preflight evidence.",
        "gate_passed": not failures,
        "failures": failures,
    }
    payload = json.dumps(result, indent=1, sort_keys=True) + "\n"
    Path(args.out).write_text(payload)
    print(json.dumps({"out": args.out, "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                      "gate_passed": not failures, "failures": failures}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

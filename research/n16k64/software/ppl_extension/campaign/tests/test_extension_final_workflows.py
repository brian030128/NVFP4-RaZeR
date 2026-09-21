import json
import inspect
from pathlib import Path

import pytest

from campaign.extension_bundle import (
    Selection,
    copy_one,
    host_local_run_metadata,
    numeric_fingerprint,
    recursive_checksum_lines,
    rebase_checksum_sidecars,
    redact_embedded_identity,
    redact_string,
    transform_json,
    write_matrix_csv,
)
from campaign.extension_final_analysis import (
    concurrency_audit,
    g2_material_claim_classification,
    g5_continuation_gate,
    matrix_coverage_prebundle,
    proven_post_exit_processes,
    walk_contrasts,
)
from campaign.extension_recompute_tables import check_json_csv_pair, render_csv
from campaign.extension_verify_bundle import (
    amendment_chain_errors,
    bundle_census_errors,
    embedded_identity_paths,
    verify_packaged_checksum_sidecars,
    verify_run_manifest_source_links,
    windows_safe,
)


def test_concurrency_limit_and_uuid_overlap_are_independently_checked():
    events = [
        {"event": "started", "logged_utc": "1", "run_id": "a", "uuids": ["u0", "u1"]},
        {"event": "started", "logged_utc": "2", "run_id": "b", "uuids": ["u2"]},
        {"event": "finished", "logged_utc": "3", "run_id": "a"},
        {"event": "finished", "logged_utc": "4", "run_id": "b"},
    ]
    report = concurrency_audit(events)
    assert report["passed"] and report["max_concurrent_campaign_gpus"] == 3
    events.insert(2, {"event": "started", "logged_utc": "2.5", "run_id": "c", "uuids": ["u2"]})
    events.append({"event": "finished", "logged_utc": "3.5", "run_id": "c"})
    assert not concurrency_audit(events)["passed"]


def test_redaction_rebases_paths_without_changing_scientific_numbers():
    original = {"path": "/private/campaign/sample.jsonl.gz", "raw_ppl": 6.25,
                "nested": [{"estimate": -0.001, "ci95": [-0.002, 0.0]}],
                "hostname": "private-host", "uid": 1011}
    aliases = {"/private/campaign/sample.jsonl.gz": "campaign/sample.jsonl.gz"}
    transformed = transform_json(original, aliases)
    assert transformed["path"] == "campaign/sample.jsonl.gz"
    assert transformed["hostname"] == "review-host"
    assert transformed["uid"] == "redacted_owner"
    assert numeric_fingerprint(original) == numeric_fingerprint(transformed)


def test_redaction_removes_aggregated_owner_and_uid_lists():
    original = {"owners": ["private-user"], "owner_uids": [1011],
                "observed_process_pids": [12345], "estimate": -0.001}
    transformed = transform_json(original, {})
    assert transformed["owners"] == ["redacted_owner"]
    assert transformed["owner_uids"] == ["redacted_owner"]
    assert transformed["observed_process_pids"] == [12345]
    assert numeric_fingerprint(original) == numeric_fingerprint(transformed)


def test_redaction_removes_owner_and_uid_embedded_in_failure_text():
    text = (
        "foreign process: [{'pid': 123, 'owner': 'thirdparty_l', 'uid': 1034}]"
    )
    transformed = redact_embedded_identity(text, {})
    assert "thirdparty_l" not in transformed
    assert "1034" not in transformed
    assert "'owner': 'redacted_owner'" in transformed
    assert "'uid': redacted_owner_uid" in transformed


def test_every_observed_private_owner_token_has_a_redaction_replacement():
    for encoded in (
        "627261696e5f6c",
        "626f736f6e5f6c",
        "70616e676368756e5f6c",
        "6368616f7975616e5f6c",
    ):
        owner = bytes.fromhex(encoded).decode()
        assert owner not in redact_string(owner, {})


def test_verifier_detects_embedded_owner_metadata_but_accepts_redacted_text():
    raw = {"reason": "{'owner': 'thirdparty_l', 'uid': 1034}"}
    kinds = {row["kind"] for row in embedded_identity_paths(raw)}
    assert kinds == {"embedded_owner", "embedded_numeric_owner_uid"}
    redacted = {"reason": "{'owner': 'redacted_owner', 'uid': redacted_owner_uid}"}
    assert embedded_identity_paths(redacted) == []


def test_redaction_pseudonymizes_host_local_execution_identifiers_consistently():
    original = {"container": "host-container-id", "lease_id": "host-lease-id",
                "allocation_id": "host-allocation-id", "pci_bus_id": "0000:01:00.0",
                "repeat": {"container": "host-container-id"}}
    transformed = transform_json(original, {})
    assert transformed["container"].startswith("redacted_container_")
    assert transformed["container"] == transformed["repeat"]["container"]
    assert transformed["lease_id"].startswith("redacted_lease_id_")
    assert transformed["allocation_id"].startswith("redacted_allocation_id_")
    assert transformed["pci_bus_id"].startswith("redacted_pci_bus_id_")


def test_host_local_docker_inspect_metadata_is_excluded_from_bundle():
    assert host_local_run_metadata("container.cid")
    assert host_local_run_metadata("docker_inspect.json")
    assert not host_local_run_metadata("launch_record.json")
    assert not host_local_run_metadata("gpu_monitor.jsonl")


def test_flat_contrast_rows_retain_source_pointer(tmp_path):
    path = tmp_path / "analysis.json"
    value = {"models": {"m": {"wiki": {"estimate": -0.2, "ci95": [-0.3, -0.1],
                                           "model": "m", "corpus": "wiki"}}}}
    path.write_text(json.dumps(value))
    rows = walk_contrasts(value, path)
    assert len(rows) == 1
    assert rows[0]["dlogppl"] == -0.2
    assert rows[0]["json_pointer"] == "/models/m/wiki"


def test_flat_contrast_rows_preserve_two_sided_and_holm_p_values(tmp_path):
    path = tmp_path / "analysis.json"
    value = {"estimate": -0.2, "ci95": [-0.3, -0.1],
             "p_two_sided": 0.0125, "p_holm": 0.0375}
    path.write_text(json.dumps(value))
    row = walk_contrasts(value, path)[0]
    assert row["p_raw"] == 0.0125
    assert row["p_adjusted"] == 0.0375


def test_flat_accuracy_rows_preserve_mcnemar_and_holm_p_values(tmp_path):
    path = tmp_path / "analysis.json"
    value = {"diff": -0.01, "ci95": [-0.02, 0.0],
             "mcnemar_p": 0.025, "mcnemar_p_holm": 0.1}
    path.write_text(json.dumps(value))
    row = walk_contrasts(value, path)[0]
    assert row["record_type"] == "paired_accuracy"
    assert row["p_raw"] == 0.025
    assert row["p_adjusted"] == 0.1


def test_windows_safe_archive_names():
    assert windows_safe("campaign/runs/P80/final/table.json")
    assert not windows_safe("campaign\\bad.json")
    assert not windows_safe("campaign/CON.txt")
    assert not windows_safe("../escape")


def test_bundle_census_allows_only_the_two_recursive_metadata_exceptions():
    manifest = {"campaign/result.json", "REVIEWER_README.md"}
    checksums = manifest | {"EVIDENCE_BUNDLE_MANIFEST.json"}
    actual = checksums | {"SHA256SUMS.txt"}
    assert bundle_census_errors(actual, manifest, checksums) == []
    assert bundle_census_errors(actual | {"silent.bin"}, manifest, checksums)
    assert bundle_census_errors(actual, manifest, checksums | {"silent.bin"})


def test_recursive_checksums_exclude_only_top_level_checksum(tmp_path):
    nested = tmp_path / "handoff/SHA256SUMS.txt"
    nested.parent.mkdir()
    nested.write_text("normative inner checksums\n")
    (tmp_path / "SHA256SUMS.txt").write_text("self\n")
    lines = recursive_checksum_lines(tmp_path)
    assert len(lines) == 1
    assert lines[0].endswith("  handoff/SHA256SUMS.txt")


def test_terminal_matrix_csv_round_trip(tmp_path):
    coverage = {"rows": [
        {"matrix_id": "P00", "status": "complete", "evidence": ["run0"], "reason": None},
        {"matrix_id": "P21", "status": "stopped_by_gate", "evidence": ["gate.json"],
         "reason": "frozen gate did not authorize breadth"},
    ]}
    path = tmp_path / "EXPERIMENT_MATRIX.csv"
    write_matrix_csv(path, coverage)
    import csv
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["complete", "stopped_by_gate"]
    assert rows[0]["evidence"] == '["run0"]'


def test_amendment_chain_verifies_links_entries_and_protocol():
    import hashlib
    protocol = "a" * 64
    rows = []
    previous = None
    for title in ("initial", "clarification"):
        body = {"title": title, "protocol_sha256": protocol,
                "previous_entry_sha256": previous}
        digest = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")).hexdigest()
        rows.append({**body, "entry_sha256": digest})
        previous = digest
    assert amendment_chain_errors(rows, protocol) == []
    rows[1]["previous_entry_sha256"] = "0" * 64
    assert amendment_chain_errors(rows, protocol)


def test_redacted_run_manifest_chain_uses_source_not_packaged_hash(tmp_path):
    prefix = tmp_path / "campaign/runs/example_attempt1"
    prefix.mkdir(parents=True)
    (prefix / "result.json").write_text('{"owner":"redacted_owner"}\n')
    source_digest = "a" * 64
    (prefix / "SHA256SUMS_run.txt").write_text(f"{source_digest}  result.json\n")
    by_path = {
        "campaign/runs/example_attempt1/result.json": {"source_sha256": source_digest,
                                                        "sha256": "b" * 64},
        "campaign/runs/example_attempt1/SHA256SUMS_run.txt": {"source_sha256": "c" * 64},
    }
    errors, checked = verify_run_manifest_source_links(tmp_path, by_path)
    assert errors == [] and checked == 1
    by_path["campaign/runs/example_attempt1/result.json"]["source_sha256"] = "d" * 64
    assert verify_run_manifest_source_links(tmp_path, by_path)[0]


def test_conventional_checksum_sidecar_is_rebased_with_source_bridge(tmp_path):
    target = tmp_path / "freeze/WINNER.json"
    target.parent.mkdir()
    target.write_text('{"path":"campaign/map"}\n')
    sidecar = tmp_path / "freeze/WINNER.sha256"
    source_target_sha = "a" * 64
    sidecar.write_text(f"{source_target_sha}  WINNER.json\n")
    target_sha = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
    entries = [
        {"bundle_relative_path": "freeze/WINNER.json", "sha256": target_sha,
         "byte_size": target.stat().st_size, "source_sha256": source_target_sha},
        {"bundle_relative_path": "freeze/WINNER.sha256",
         "sha256": __import__("hashlib").sha256(sidecar.read_bytes()).hexdigest(),
         "byte_size": sidecar.stat().st_size, "source_sha256": "b" * 64,
         "transform": "byte_identical_text", "numeric_content_preserved": None},
    ]
    assert rebase_checksum_sidecars(tmp_path, entries) == 1
    by_path = {row["bundle_relative_path"]: row for row in entries}
    assert verify_packaged_checksum_sidecars(tmp_path, by_path) == ([], 1)
    assert sidecar.read_text() == f"{target_sha}  WINNER.json\n"
    assert entries[1]["sidecar_source_target_sha256"] == source_target_sha


def test_no_gpu_csv_renderer_matches_final_analysis_field_order():
    rows = [{"z": 2, "a": 1}, {"a": 3, "extra": None}]
    assert render_csv(rows) == "a,extra,z\r\n1,,2\r\n3,,\r\n"


def test_text_redaction_preserves_csv_crlf_for_exact_reconstruction(tmp_path):
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    source.write_bytes(b"path,value\r\n/home/private_user/result.json,1\r\n")
    item = Selection(source, "campaign/result.csv", "final_derived_report",
                     "P80_FINAL_ANALYSIS")
    copy_one(item, target, {})
    assert target.read_bytes() == b"path,value\r\nREDACTED_HOME/result.json,1\r\n"


def test_no_gpu_csv_check_detects_value_drift(tmp_path):
    rows = [{"effect": -0.1, "ci95_low": -0.2, "ci95_high": 0.0}]
    (tmp_path / "CALIBRATION_ROBUSTNESS.json").write_text(json.dumps({"flat_rows": rows}))
    path = tmp_path / "CALIBRATION_ROBUSTNESS.csv"
    path.write_bytes(render_csv(rows).encode("utf-8"))
    assert check_json_csv_pair(tmp_path, "CALIBRATION_ROBUSTNESS") == 1
    path.write_bytes(path.read_bytes().replace(b"-0.1", b"-0.3"))
    with pytest.raises(RuntimeError, match="byte-exact reconstruction"):
        check_json_csv_pair(tmp_path, "CALIBRATION_ROBUSTNESS")


def test_g2_gate_does_not_turn_a_practically_equivalent_effect_into_material_claim():
    assert g2_material_claim_classification(
        {"passed": True, "detail": {"practically_equivalent": True}}
    ) == "unsupported_practically_equivalent"
    assert g2_material_claim_classification(
        {"passed": True, "detail": {"practically_equivalent": False}}
    ) == "supported"


def test_g5_reports_the_scientific_continuation_outcome_not_only_compliance():
    p31 = {"G5": {"cross_model_rule_met": False}}
    stopped = g5_continuation_gate(p31, {"continue_to_p32": False}, "gate.json")
    assert stopped == {
        "passed": False,
        "continued_to_256": False,
        "decision_rule_applied": True,
        "evidence": "gate.json",
        "detail": p31["G5"],
    }
    continued = g5_continuation_gate(p31, {"continue_to_p32": True}, "gate.json")
    assert continued["passed"] and continued["continued_to_256"]


def test_only_unambiguous_post_exit_processes_are_exempted_from_after_check():
    launch = {
        "host_after_post_exit_only": True,
        "leased_uuids": ["GPU-u0"],
        "late_foreign_processes": [],
        "post_exit_foreign_processes": [{
            "pid": 123,
            "gpu_uuid": "GPU-u0",
            "classification": "post-exit (started after this run finished)",
            "seconds_after_container_exit": 1.5,
        }],
    }
    assert proven_post_exit_processes(launch) == {(123, "GPU-u0")}
    launch["post_exit_foreign_processes"][0]["seconds_after_container_exit"] = 0.5
    assert proven_post_exit_processes(launch) is None
    launch["post_exit_foreign_processes"][0]["seconds_after_container_exit"] = 1.5
    launch["late_foreign_processes"] = [{"pid": 456}]
    assert proven_post_exit_processes(launch) is None


def test_final_coverage_indexes_conditional_breadth_gates_by_full_matrix_id():
    source = inspect.getsource(matrix_coverage_prebundle)
    for matrix_id in (
        "P21_K2_DENSITY_BREADTH",
        "P42_DRAW_AGG_BREADTH",
        "P52_SELECTOR_BREADTH",
    ):
        assert matrix_id in source

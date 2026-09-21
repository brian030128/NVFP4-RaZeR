import pytest

from campaign.extension_p75_selector_reuse import (
    selector_semantics,
    validate_fallback,
    validated_module_manifest,
)


def test_omitted_default_chunk_size_is_semantically_identical():
    base = ["-m", "campaign.extension_selector_stats", "--model", "llama8b",
            "--freeze", "/campaign/freeze.json", "--freeze-sha256", "a" * 64,
            "--attn", "sdpa"]
    explicit = [*base, "--reconstruction-chunk-rows", "512"]
    assert selector_semantics(base) == selector_semantics(explicit)


def test_fallback_requires_identical_p41_and_p50_exact_map():
    row = {"map_path": "/parent/run/map.mixfp4map", "map_sha256": "b" * 64,
           "type_block": [16, 64]}
    winner = {"global_configuration": {"selector_fallback_to_old": True,
                                         "selector": "first_order_ce_kl"}}
    p50 = {"fallback_to_old": True, "selected_replacement": None,
           "selected_policy": "p50_old_first_order_ce_kl",
           "selected_maps": {"llama8b": dict(row)}}
    p41 = {"selected_maps": {"llama8b": dict(row)}}
    assert validate_fallback(winner, p50, p41) == row


def test_module_manifest_comes_from_matching_reports_when_map_header_omits_it():
    digest = "c" * 64
    assert validated_module_manifest(
        {"schema": "mixfp4map/v1"},
        {"module_manifest_sha256": digest},
        {"module_manifest_sha256": digest},
    ) == digest


def test_module_manifest_report_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match="module manifests differ"):
        validated_module_manifest(
            {},
            {"module_manifest_sha256": "c" * 64},
            {"module_manifest_sha256": "d" * 64},
        )

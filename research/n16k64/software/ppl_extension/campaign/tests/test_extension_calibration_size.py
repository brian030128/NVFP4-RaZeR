import torch


def _state(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    return {"n": values.shape[0], "ce_sum": values.sum(0), "ce_sq": (values * values).sum(0),
            "kl_sum": (2 * values).sum(0), "kl_sq": (4 * values * values).sum(0),
            "cross": (2 * values * values).sum(0)}


def test_incremental_moment_addition_equals_direct_concatenation():
    from campaign.extension_calibration_size import add_moment_states

    a = torch.tensor([[1.0, 3.0], [2.0, 5.0]], dtype=torch.float64)
    b = torch.tensor([[7.0, 11.0], [13.0, 17.0], [19.0, 23.0]], dtype=torch.float64)
    got = add_moment_states(_state(a), _state(b))
    expected = _state(torch.cat((a, b)))
    assert got["n"] == 5
    for key in ("ce_sum", "ce_sq", "kl_sum", "kl_sq", "cross"):
        assert torch.equal(got[key], expected[key])


def test_combined_manifest_is_domain_major_prefix_then_addition():
    from campaign.extension_calibration_size import combined_manifest

    prefix = {d: {"documents": [{"document_sha256": d + "0"}], "token_sha256": [d + "t0"]}
              for d in ("math", "code")}
    addition = {d: {"draw": "size128_v1", "documents": [{"document_sha256": d + "1"}],
                    "token_sha256": [d + "t1"]} for d in ("math", "code")}
    got = combined_manifest(prefix, addition, "p31", "abc")
    assert [x["document_sha256"] for x in got["math"]["documents"]] == ["math0", "math1"]
    assert got["code"]["token_sha256"] == ["codet0", "codet1"]
    assert got["math"]["addition_draw"] == "size128_v1"


def test_p31_p32_uses_frozen_extension_model_set_not_legacy_parent_panel():
    import pytest

    from campaign.extension_calibration_size import assert_p31_p32_model

    # Mistral is explicitly in the locked P31/P32 development set even though
    # its shared parent-campaign registry entry retains a confirmatory label.
    assert assert_p31_p32_model("mistral7b") is True
    with pytest.raises(ValueError, match="development-only"):
        assert_p31_p32_model("granite8b")


def test_parent_seed0_manifests_match_frozen_parent_report_digests():
    from campaign import data as D
    from campaign.extension_calibration_size import parent_seed0_manifest

    expected = {
        "llama8b": "9fa00fa12842f886dd26285a9de516446149aaee3ed6868135b5c802b0c99696",
        "qwen4b": "7722a6ee8af07883d2193a6693d4bc6852c547e0c306ea378bf0c1c597f88a5d",
        "mistral7b": "4018df383c3cc951b8905b77b1b3bcdf60da1c5d6f85669ee62431b9720e7953",
    }
    for model, digest in expected.items():
        manifest = parent_seed0_manifest(model)
        assert D.manifest_sha256(manifest) == digest
        assert len(manifest["math"]["documents"]) == 64
        assert len(manifest["code"]["documents"]) == 64

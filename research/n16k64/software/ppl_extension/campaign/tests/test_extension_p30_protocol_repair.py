import pytest

from campaign.extension_p30_protocol_repair import corrected_entry


def entry():
    return {"name": "n16_k2_draw1", "kind": "map", "map_policy": "n16_k2",
            "map_path": "/logical/map", "map_sha256": "a" * 64,
            "type_block": [16, 64], "protocol_id": "aligned-primary",
            "expected_total_tiles": 17}


def header(protocol="aligned-robustness"):
    return {"protocol_id": protocol, "policy": {"name": "n16_k2"},
            "type_block": [16, 64], "totals": {"total_tiles": 17}}


def test_repair_changes_only_protocol_label():
    original = entry()
    repaired, evidence = corrected_entry(original, header(), "a" * 64)
    assert original["protocol_id"] == "aligned-primary"
    assert repaired == {**original, "protocol_id": "aligned-robustness"}
    assert evidence["map_sha256"] == "a" * 64


def test_repair_refuses_unexpected_header_protocol():
    with pytest.raises(RuntimeError, match="map-header protocol"):
        corrected_entry(entry(), header("unreviewed-protocol"), "a" * 64)

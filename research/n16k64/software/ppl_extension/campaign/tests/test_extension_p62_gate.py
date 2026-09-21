from campaign.extension_p62_gate import resolution


def test_p62_frozen_default_stops_without_attempt():
    spec = {"P62_local_transform": {
        "required_pre_result_proof": "proof", "current_evidence": "none",
        "fallback": "stop"}}
    out = resolution(spec)
    assert out["status"] == "stopped_by_gate"
    assert out["attempts_launched"] == 0
    assert out["gate"] == "G7"

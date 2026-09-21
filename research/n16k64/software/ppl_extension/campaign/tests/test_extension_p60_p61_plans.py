def _winner():
    row = {"map_path": "/x", "map_sha256": "a" * 64, "map_policy": "p",
           "type_block": [16, 64], "expected_total_tiles": 10}
    return {"selected_maps": {model: row for model in ("llama8b", "qwen4b", "mistral7b")}}


def test_p60_axis_and_order(monkeypatch):
    import campaign.extension_p60_p61_plans as module

    monkeypatch.setattr(module, "map_entry", lambda name, row, **kw: {"name": name, **kw})
    plan = module.build("p60", "llama8b", _winner())
    assert len(plan) == 6
    assert [row["name"] for row in plan] == [
        "p60_fos_s1p0", "p60_fos_s0p95", "p60_fos_s1p05",
        "p60_winner_s1p0", "p60_winner_s0p95", "p60_winner_s1p05"]
    assert [row["weight_scale_multiplier"] for row in plan] == [1.0, .95, 1.05] * 2


def test_p61_is_two_weight_arms_by_five_activation_rules(monkeypatch):
    import campaign.extension_p60_p61_plans as module

    monkeypatch.setattr(module, "map_entry", lambda name, row, **kw: {"name": name, **kw})
    plan = module.build("p61", "qwen4b", _winner())
    assert len(plan) == 10
    assert len({row["name"] for row in plan}) == 10
    assert plan[0]["activation_kind"] == "four_over_six_rows"
    assert plan[4]["activation_kind"] == "four_over_six_rows_percentile_100"
    assert plan[5]["name"] == "p61_winner_baseline_absmax"

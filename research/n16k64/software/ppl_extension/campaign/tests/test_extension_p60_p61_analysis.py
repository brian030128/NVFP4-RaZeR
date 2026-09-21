def _row(name, median, worst, eligible=True):
    return {"candidate": name, "winner_median": median, "all_arm_worst": worst,
            "eligible": eligible}


def test_quantizer_selection_rule_and_fallback():
    from campaign.extension_p60_p61_analysis import select_summary

    rows = [_row("a", -0.0010, 0.0008), _row("b", -0.00105, 0.0004),
            _row("c", -0.1, 0.1, False)]
    assert select_summary(rows, {"a": 0, "b": 1, "c": 2})["candidate"] == "b"
    assert select_summary([dict(row, eligible=False) for row in rows], {"a": 0, "b": 1, "c": 2}) is None


def test_quantizer_exact_tie_uses_predeclared_order():
    from campaign.extension_p60_p61_analysis import select_summary

    rows = [_row("later", -0.001, 0), _row("first", -0.001, 0)]
    assert select_summary(rows, {"first": 0, "later": 1})["candidate"] == "first"

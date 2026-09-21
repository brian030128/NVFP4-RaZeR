def _row(name, median, worst, eligible=True):
    return {"policy_name": name, "median_six_cells": median,
            "worst_cell": worst, "eligible": eligible}


def test_p50_selection_eligibility_and_ties():
    from campaign.extension_p50_analysis import CANDIDATES, select_candidate

    rows = [_row(CANDIDATES[0], -0.0007, 0.0008),
            _row(CANDIDATES[1], -0.00075, 0.0004),
            _row(CANDIDATES[2], -0.002, 0.002, False)]
    # First two medians are within 0.0001; the better worst cell wins.
    assert select_candidate(rows)["policy_name"] == CANDIDATES[1]
    assert select_candidate([dict(row, eligible=False) for row in rows]) is None
    assert select_candidate(rows, require_eligible=False)["policy_name"] == CANDIDATES[2]


def test_p50_frozen_order_breaks_exact_tie():
    from campaign.extension_p50_analysis import CANDIDATES, select_candidate

    rows = [_row(name, -0.001, 0.0) for name in reversed(CANDIDATES)]
    assert select_candidate(rows)["policy_name"] == CANDIDATES[0]


def test_audit_uses_eligible_winner_when_one_exists_and_best_diagnostic_otherwise():
    from campaign.extension_p50_analysis import CANDIDATES, select_audit_candidate

    rows = [_row(CANDIDATES[0], -0.001, 0.0005, True),
            _row(CANDIDATES[1], -0.010, 0.010, False),
            _row(CANDIDATES[2], 0.0, 0.0, False)]
    assert select_audit_candidate(rows)["policy_name"] == CANDIDATES[0]
    rows[0]["eligible"] = False
    assert select_audit_candidate(rows)["policy_name"] == CANDIDATES[1]


def test_p50_policy_contract_accepts_canonical_json_key_order():
    from campaign.extension_p50_analysis import validate_policy_contract

    expected = ("frozen_first", "frozen_second", "frozen_third")
    report = {
        # Canonical JSON serialization sorts mapping keys and must not be used
        # as evidence of plan-order drift.
        "evaluation": {name: {} for name in sorted(expected)},
        "plan": [{"name": name} for name in expected],
        "installs": [{"name": name} for name in expected],
    }
    validate_policy_contract(report, expected)


def test_p50_policy_contract_rejects_real_sequence_or_membership_drift():
    import pytest
    from campaign.extension_p50_analysis import validate_policy_contract

    expected = ("a", "b", "c")
    report = {
        "evaluation": {name: {} for name in expected},
        "plan": [{"name": name} for name in expected],
        "installs": [{"name": name} for name in ("b", "a", "c")],
    }
    with pytest.raises(RuntimeError, match="installs policy order drift"):
        validate_policy_contract(report, expected)
    report["installs"] = [{"name": name} for name in expected]
    del report["evaluation"]["c"]
    with pytest.raises(RuntimeError, match="evaluation policy membership drift"):
        validate_policy_contract(report, expected)

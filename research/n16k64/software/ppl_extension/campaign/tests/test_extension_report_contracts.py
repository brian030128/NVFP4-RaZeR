import pytest


def report(expected, evaluation=None, plan=None, installs=None):
    return {
        "evaluation": {name: {} for name in (evaluation or sorted(expected))},
        "plan": [{"name": name} for name in (plan or expected)],
        "installs": [{"name": name} for name in (installs or expected)],
    }


def test_canonical_mapping_order_is_not_treated_as_plan_order():
    from campaign.report_contracts import validate_policy_contract

    expected = ("z_frozen_first", "a_frozen_second", "m_frozen_third")
    validate_policy_contract(report(expected), expected, "stage model")


def test_policy_membership_drift_is_rejected():
    from campaign.report_contracts import validate_policy_contract

    expected = ("a", "b", "c")
    with pytest.raises(RuntimeError, match="evaluation policy membership drift"):
        validate_policy_contract(report(expected, evaluation=("a", "b", "d")),
                                 expected, "stage model")


def test_canonical_lmeval_results_mapping_uses_same_contract():
    from campaign.report_contracts import validate_policy_contract

    expected = ("z_frozen_first", "a_frozen_second", "m_frozen_third")
    value = report(expected)
    value["results"] = value.pop("evaluation")
    validate_policy_contract(value, expected, "P72 model", mapping_field="results")


@pytest.mark.parametrize("field", ("plan", "installs"))
def test_sequence_bearing_policy_order_drift_is_rejected(field):
    from campaign.report_contracts import validate_policy_contract

    expected = ("a", "b", "c")
    kwargs = {field: ("b", "a", "c")}
    with pytest.raises(RuntimeError, match=rf"{field} policy order drift"):
        validate_policy_contract(report(expected, **kwargs), expected, "stage model")

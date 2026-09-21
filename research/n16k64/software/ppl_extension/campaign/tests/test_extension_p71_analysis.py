from campaign.extension_p71_analysis import SHARED_ARMS, SYSTEM_ARMS, strongest, strongest_model


def _report():
    return {"evaluation": {arm: {"wiki": {"mean_nll": float(i)},
                                      "c4": {"mean_nll": float(10 - i)}}
                           for i, arm in enumerate(SYSTEM_ARMS)}}


def test_strongest_is_selected_without_winner_arm():
    report = _report()
    assert strongest(report, "wiki", SYSTEM_ARMS) == SYSTEM_ARMS[0]
    assert strongest(report, "c4", SYSTEM_ARMS) == SYSTEM_ARMS[-1]
    assert strongest_model(report) in SYSTEM_ARMS
    assert strongest_model(report, SHARED_ARMS) in SHARED_ARMS

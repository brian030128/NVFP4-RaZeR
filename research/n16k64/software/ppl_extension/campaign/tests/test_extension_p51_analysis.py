import numpy as np


def test_exact_binomial_boundaries_and_weighted_metrics():
    from campaign.extension_p51_analysis import exact_binomial_ci, metric_bundle

    assert exact_binomial_ci(0, 10)[0] == 0.0
    assert exact_binomial_ci(10, 10)[1] == 1.0
    result = metric_bundle(np.arange(6.0), np.array([1, 1, 1, 0, 0, 0], dtype=bool),
                           np.array([-3, -2, -1, 1, 2, 3], dtype=float), np.ones(6))
    assert result["weighted_sign_accuracy"] == 1.0
    assert result["weighted_spearman_rho"] == 1.0
    assert result["weighted_kendall_tau_b"] == 1.0


def test_bootstrap_is_named_stream_deterministic():
    from campaign.extension_p51_analysis import bootstrap_metrics

    score = np.arange(8.0); actual = score + np.array([0, .1, 0, -.1, 0, .1, 0, -.1])
    pred = score < 4; weight = np.ones(8); labels = [i // 2 for i in range(8)]
    a = bootstrap_metrics(score, pred, actual, weight, labels, "unit-test", B=100)
    b = bootstrap_metrics(score, pred, actual, weight, labels, "unit-test", B=100)
    assert a == b


def test_weighted_tau_handles_ties():
    from campaign.extension_p51_analysis import weighted_tau_b

    tau = weighted_tau_b([0, 0, 1, 2], [0, 0, 2, 3], [1, 2, 1, 1])
    assert 0.99 <= tau <= 1.0


def test_point_strata_routes_selector_and_outcome():
    from campaign.extension_p51_analysis import point_strata

    rows = [{"layer_quartile": i // 2, "old_score": float(i),
             "old_predicted_beneficial": i < 2, "actual_ce": float(i - 2),
             "sampling_weight": 1.0} for i in range(4)]
    result = point_strata(rows, "old_first_order_ce_kl", "full_model_CE", "layer_quartile")
    assert set(result) == {"0", "1"}
    assert all(row["n"] == 2 for row in result.values())

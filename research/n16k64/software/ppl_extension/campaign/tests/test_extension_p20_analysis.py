def test_median_random_is_middle_aggregate_seed_with_stable_seed_tie():
    from campaign.extension_p20_analysis import median_random_policy, RANDOM_SEEDS

    evaluation = {}
    values = [4.0, 1.0, 3.0, 2.0, 5.0]
    for seed, value in zip(RANDOM_SEEDS, values):
        evaluation[f"p20_global_random_{seed}"] = {"wiki": {"mean_nll": value}}
    policy, ranking = median_random_policy({"evaluation": evaluation}, "global", "wiki")
    assert policy == f"p20_global_random_{RANDOM_SEEDS[2]}"
    assert [x[0] for x in ranking] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_strongest_control_uses_median_tolerance_then_worst_then_lexical():
    from campaign.extension_p20_analysis import select_strongest

    rows = [
        {"scope": "per_module", "selector": "z", "median_control_minus_natural": -0.0010,
         "worst_control_minus_natural": 0.0002},
        {"scope": "global", "selector": "b", "median_control_minus_natural": -0.00095,
         "worst_control_minus_natural": 0.0001},
        {"scope": "global", "selector": "a", "median_control_minus_natural": -0.00095,
         "worst_control_minus_natural": 0.0001},
    ]
    assert select_strongest(rows)["selector"] == "a"

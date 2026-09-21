import numpy as np


def test_two_level_bootstrap_is_deterministic_and_separates_components():
    from campaign.extension_draw_analysis import two_level_interval

    components = []
    for draw in range(5):
        dc = np.array([-1.0, 0.5, -0.25]) + draw * 0.01
        nc = np.array([100.0, 120.0, 80.0])
        components.append((dc, nc))
    a = two_level_interval(components, B=300, seed=7)
    b = two_level_interval(components, B=300, seed=7)
    assert a == b
    assert a["between_draw_variance"] > 0
    assert a["mean_within_evaluation_bootstrap_variance"] > 0
    assert len(a["hierarchical_ci95"]) == 2

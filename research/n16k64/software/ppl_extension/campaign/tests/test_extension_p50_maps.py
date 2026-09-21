from collections import OrderedDict

import pytest
import torch


def test_strong_score_direction_transform():
    from campaign.extension_p50_maps import transformed_strong_scores

    names = ["a", "b"]
    weight = {
        "mse_gain16": OrderedDict((n, torch.tensor([[v]], dtype=torch.float64))
                                  for n, v in zip(names, (2.0, -3.0))),
        "l2_16": OrderedDict((n, torch.tensor([[v]], dtype=torch.float64))
                             for n, v in zip(names, (4.0, 1.0))),
        "dnorm16": OrderedDict((n, torch.tensor([[v]], dtype=torch.float64))
                               for n, v in zip(names, (5.0, 7.0))),
    }
    stats = {"scores": {"activation_weighted_error": OrderedDict(
        (n, torch.tensor([[v]], dtype=torch.float64)) for n, v in zip(names, (-1.0, 2.0)))}}
    got, negated = transformed_strong_scores("weight_mse", weight, stats)
    assert negated and got["a"].item() == -2.0 and got["b"].item() == 3.0
    got, negated = transformed_strong_scores("activation_weighted", weight, stats)
    assert not negated and got is stats["scores"]["activation_weighted_error"]
    with pytest.raises(ValueError, match="unsupported"):
        transformed_strong_scores("unknown", weight, stats)


def test_validate_scores_requires_exact_geometry_and_finite():
    from campaign.extension_p50_maps import validate_scores

    names = ["z"]
    shapes = {"z": (32, 128)}
    validate_scores(names, shapes, {"grid": OrderedDict(z=torch.zeros(2, 2))})
    validate_scores(names, shapes, {"flat": OrderedDict(z=torch.zeros(4))})
    with pytest.raises(RuntimeError, match="score size"):
        validate_scores(names, shapes, {"bad": OrderedDict(z=torch.zeros(3))})
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_scores(names, shapes, {"bad": OrderedDict(z=torch.tensor([[0.0, float("nan")], [0.0, 0.0]]))})


def test_global_selection_is_exact_and_stable_on_ties():
    from campaign.extension_maps import select_global

    names = ["z", "a"]
    scores = OrderedDict(z=torch.zeros(1, 3), a=torch.zeros(1, 3))
    masks = select_global(names, scores, 4)
    assert int(sum(x.sum() for x in masks.values())) == 4
    # Module-name tie ordering is a before z, while returned mapping preserves input order.
    assert masks["a"].tolist() == [[True, True, True]]
    assert masks["z"].tolist() == [[True, False, False]]

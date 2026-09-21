import numpy as np
import torch

from campaign.followup_analyze import holm, weighted_fit
from campaign.followup_maps import dose_maps, objective_maps


def synthetic():
    name = "model.layers.0.self_attn.q_proj"
    full = {name: torch.zeros((4, 8), dtype=torch.bool)}
    full[name].view(-1)[:19] = True
    uce = torch.linspace(-2.0, 1.0, 32, dtype=torch.float64)
    ukl = torch.linspace(-1.5, 1.5, 32, dtype=torch.float64)
    # Ensure the supplied frozen conjunction is a subset of both pass sets.
    uce[:19] = torch.linspace(-2.0, -0.1, 19)
    ukl[:19] = torch.linspace(-1.8, -0.05, 19)
    stats = {name: {"uce": uce, "ukl": ukl, "combined_margin": -torch.maximum(uce, ukl),
                    "ce_mean": uce - 0.1, "kl_mean": ukl - 0.1,
                    "ce_se": torch.full_like(uce, 0.01), "kl_se": torch.full_like(ukl, 0.01)}}
    return name, full, stats


def test_dose_bins_are_equal_disjoint_and_accounted():
    name, full, stats = synthetic()
    maps, detail = dose_maps("synthetic", full, stats)
    bins = [maps[f"dose_group_only_bin{i:02d}"][name] for i in range(1, 9)]
    assert [int(x.sum()) for x in bins] == [2] * 8
    assert detail["excluded_remainder_count"] == 3
    union = torch.zeros_like(full[name])
    for i, left in enumerate(bins):
        for right in bins[i + 1:]:
            assert not (left & right).any()
        union |= left
    assert int(union.sum()) + detail["excluded_remainder_count"] == int(full[name].sum())


def test_objective_matched_k_uses_exact_conjunction_quota():
    name, full, stats = synthetic()
    maps, detail = objective_maps("synthetic", full, stats)
    quota = int(full[name].sum())
    for policy in ("objective_conjunction", "objective_ce_matched_k", "objective_kl_matched_k", "objective_random_matched_k"):
        assert int(maps[policy][name].sum()) == quota
    assert detail["invariants"]["matched_k_per_stratum_exact"]
    assert len(detail["pairwise_overlap"]) == 15


def test_weighted_calibration_and_holm_are_deterministic():
    x = np.arange(8, dtype=np.float64)
    y = 1.5 + 2.0 * x
    slope, intercept, r2 = weighted_fit(x, y, np.ones(8))
    assert np.isclose(slope, 2.0)
    assert np.isclose(intercept, 1.5)
    assert np.isclose(r2, 1.0)
    rows = [{"p_two_sided_plus_one": p} for p in (0.001, 0.02, 0.2)]
    holm(rows)
    assert [r["holm_adjusted_p"] for r in rows] == [0.003, 0.04, 0.2]


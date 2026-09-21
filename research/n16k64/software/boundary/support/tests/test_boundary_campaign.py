import torch

from campaign.boundary_common import round_half_up, stratified_priority
from campaign.boundary_prepare import rotating_exclusion_ranks


def test_rotating_remainder_is_unique_and_not_weakest_only():
    seen = []
    for partition in range(4):
        ranks = rotating_exclusion_ranks(5, 4, partition)
        assert len(ranks) == len(set(ranks)) == 1
        seen.extend(ranks)
    assert min(seen) == 0
    assert max(seen) == 4


def test_rotating_remainder_count_for_all_small_strata():
    for k in range(4, 100):
        for partition in range(4):
            ranks = rotating_exclusion_ranks(k, 4, partition)
            assert len(ranks) == k % 4
            assert len(ranks) == len(set(ranks))
            assert all(0 <= rank < k for rank in ranks)


def test_stratified_priority_is_deterministic_permutation():
    indices = list(range(37))
    scores = torch.linspace(3.01, 12.0, 37)
    first = stratified_priority(indices, scores, "unit-test")
    second = stratified_priority(indices, scores, "unit-test")
    assert first == second
    assert sorted(first) == indices
    assert len(set(first[:8])) == 8


def test_round_half_up_and_nested_targets():
    assert round_half_up(0.5) == 1
    for k in range(1, 100):
        targets = [round_half_up(p * k) for p in (0.10, 0.25, 0.50, 0.75, 1.00)]
        assert targets == sorted(targets)
        assert targets[-1] == k

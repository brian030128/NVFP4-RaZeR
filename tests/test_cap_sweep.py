"""CPU checks for the tile-count sweep's re-election path."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantize.relinearized_format import common_descent_scores, stale_map
from run_cap_sweep import COUNTS, policy_name


def elect(upper, order, count):
    """The sweep's inline prefix rule, isolated for testing."""
    flat = torch.zeros(upper.numel(), dtype=torch.bool)
    if count != 0:
        take = order if count is None else order[:count]
        flat[take[upper[take] < 0]] = True
    return flat


def build(tiles=4096, seqs=192, seed=0):
    g = torch.Generator().manual_seed(seed)
    # a minority of tiles carry a real negative mean; the rest are noise
    mean = torch.zeros(tiles)
    mean[:300] = -torch.rand(300, generator=g) * 0.01
    ce = mean + torch.randn(seqs, tiles, generator=g) * 0.001
    kl = mean + torch.randn(seqs, tiles, generator=g) * 0.001
    return ce, kl


def test_matches_stale_map():
    ce, kl = build()
    upper = common_descent_scores(ce, kl, torch.zeros(ce.shape[1], dtype=torch.bool))
    order = torch.argsort(upper, stable=True)
    for count in (16, 64, 256, 1024):
        assert torch.equal(elect(upper, order, count), stale_map(ce, kl, count=count)), count
    print('OK election matches stale_map at every finite count')


def test_nesting_and_eligibility():
    ce, kl = build()
    upper = common_descent_scores(ce, kl, torch.zeros(ce.shape[1], dtype=torch.bool))
    order = torch.argsort(upper, stable=True)
    eligible = int((upper < 0).sum())
    previous = None
    for count in COUNTS:
        flat = elect(upper, order, count)
        selected = int(flat.sum())
        if count == 0:
            assert selected == 0
        elif count is None:
            assert selected == eligible, (selected, eligible)
        else:
            assert selected == min(count, eligible), (count, selected, eligible)
        if previous is not None:
            assert bool((previous & ~flat).sum() == 0), f'{policy_name(count)} is not a superset'
        previous = flat
    assert 0 < eligible < ce.shape[1], eligible
    print(f'OK nested prefixes; {eligible} eligible of {ce.shape[1]}')


def test_never_selects_positive_scores():
    ce, kl = build()
    upper = common_descent_scores(ce, kl, torch.zeros(ce.shape[1], dtype=torch.bool))
    order = torch.argsort(upper, stable=True)
    # a count far above the eligible set must still refuse positive-score tiles
    flat = elect(upper, order, 1_000_000)
    assert bool((upper[flat] < 0).all())
    assert int(flat.sum()) == int((upper < 0).sum())
    print('OK an oversized count never elects a nonnegative tile')


def test_conjunction_is_the_binding_filter():
    """A tile negative on CE but positive on KL must not be eligible."""
    seqs = 192
    ce = -0.01 + torch.zeros(seqs, 4)
    kl = torch.stack([torch.full((seqs,), v) for v in (-0.01, 0.01, -0.01, 0.01)], dim=1)
    upper = common_descent_scores(ce, kl, torch.zeros(4, dtype=torch.bool))
    assert bool((upper[[0, 2]] < 0).all()) and bool((upper[[1, 3]] > 0).all())
    print('OK CE-negative/KL-positive tiles are excluded by the max')


if __name__ == '__main__':
    test_matches_stale_map()
    test_nesting_and_eligibility()
    test_never_selects_positive_scores()
    test_conjunction_is_the_binding_filter()
    print('PASS tests/test_cap_sweep.py')

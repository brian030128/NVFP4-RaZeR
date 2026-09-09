import torch
from calibration_sensitivity import calibration_subsets, select_stream, selection_overlap, validate_losses
from quantize.relinearized_format import stale_map


def main():
    subsets = calibration_subsets()
    assert len(subsets) == 17
    assert subsets['pooled64'] == list(range(22)) + list(range(64, 85)) + list(range(128, 149))
    assert subsets['pooled192'] == list(range(192))
    for a, b in zip((16, 32, 64, 128), (32, 64, 128, 192)):
        assert set(subsets[f'pooled{a}']) < set(subsets[f'pooled{b}'])
    torch.manual_seed(42)
    ce, kl = torch.randn(192, 400) * .01 - .02, torch.randn(192, 400) * .01 - .02
    # Exactly representable constants keep mathematical ties exact under both
    # reduction layouts; decimal constants can differ by a float32 rounding bit.
    ce[:, :20] = -.0625; kl[:, :20] = -.0625
    ce[:, -10:] = 1; kl[:, -10:] = 1
    ranges = {'a': (0, 7), 'b': (7, 157), 'c': (157, 400)}
    for cap in (5, 256, 500):
        sparse, counts, sizes = select_stream(
            ((n, ce[:, lo:hi], kl[:, lo:hi]) for n, (lo, hi) in ranges.items()), cap=cap)
        for policy, indices in subsets.items():
            actual = torch.zeros(400, dtype=torch.bool)
            for name, values in sparse[policy].items():
                actual[[ranges[name][0] + v for v in values]] = True
            assert torch.equal(actual, stale_map(ce[indices], kl[indices], count=cap)), policy
            assert counts[policy]['selected_blocks'] == min(cap, 390)
            assert counts[policy]['eligible_negative_score_blocks'] == 390
    assert selection_overlap({'a': [0, 1]}, {'a': [1, 2]}) == dict(intersection=1, union=3, jaccard=1/3)
    empty, stats, _ = select_stream([('a', torch.ones(192, 3), torch.ones(192, 3))])
    assert all(v['selected_blocks'] == 0 for v in stats.values())
    for values, count in [([1., float('nan')], 2), ([1.], 2), ([1.], 1)]:
        try:
            validate_losses(values, count)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid losses accepted')
    print('All 17 subsets match full-table selection, including ties, eligibility, and cap boundaries')


if __name__ == '__main__':
    main()

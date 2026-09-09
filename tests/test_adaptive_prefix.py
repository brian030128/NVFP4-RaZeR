import torch
from quantize.adaptive_prefix import adaptive_prefix, derive_maps, source_subsets


def main():
    subsets = source_subsets()
    assert len(subsets) == 10
    assert all(0 <= i < 128 for rows in subsets.values() for i in rows)
    assert subsets['math_code128'] == list(range(128))
    # Same benefit with different curvature must produce different counts.
    u = torch.full((600,), -.01, dtype=torch.float64)
    selected, stats = adaptive_prefix(u, torch.full_like(u, .001))
    assert len(selected) == 10 and stats['count_cap'] is None
    selected, _ = adaptive_prefix(u, torch.full_like(u, .00001))
    assert len(selected) == 600  # Explicitly exceeds the historical 256 cap.
    selected, _ = adaptive_prefix(u, torch.ones_like(u))
    assert len(selected) == 0
    # Check exact minimization over every prefix and the PSD cross-term bound.
    torch.manual_seed(15)
    v = torch.randn(12, 31, dtype=torch.float64) * .02
    u = -torch.rand(31, dtype=torch.float64) * .08
    diag = v.square().sum(0)
    chosen, stats = adaptive_prefix(u, diag)
    order = torch.argsort(u, stable=True)
    values = [0.] + [float(u[order[:k]].sum()+.5*diag[order[:k]].sqrt().sum().square()) for k in range(1,32)]
    assert len(chosen) == min(range(32), key=values.__getitem__)
    for k in range(32):
        indices = order[:k]
        assert float(v[:,indices].sum(1).square().sum()) <= float(diag[indices].sqrt().sum().square()) + 1e-12
    # Derivation must never select an ineligible tile and must report units.
    ce = torch.full((128, 12), -.01); kl = ce.clone(); fisher = torch.full_like(ce, .001)
    ce[:, -2:] = 1; kl[:, -2:] = 1
    maps, stats = derive_maps([('a', (8, 768), ce, kl, fisher)])
    assert len(maps) == 20
    for p, ms in maps.items():
        assert all(i < 10 for i in ms['a'])
        assert stats[p]['selected_blocks'] == len(ms['a'])
        assert stats[p]['selected_weights'] == len(ms['a'])*512
        assert stats[p]['selected_scale_blocks'] == len(ms['a'])*32
    print('Adaptive counts include zero and >256; all-prefix optimum and PSD bound verified')


if __name__ == '__main__':
    main()

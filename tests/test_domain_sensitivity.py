"""Numerical checks for cross-domain score pooling and eligibility; Slurm only."""
import torch
from run_domain_sensitivity import pool, consensus_scores
from analyze_task_sensitivity import trust_masks


def stats(x):
    return {'x': x.mean(0)}, {'x': x.std(0)/len(x)**.5}


torch.manual_seed(42)
a, b = torch.randn(32, 2, 3, dtype=torch.float64), torch.randn(32, 2, 3, dtype=torch.float64)+3
actual = pool(stats(a), stats(b))
expected = stats(torch.cat([a, b]))
for x, y in zip(actual, expected):
    torch.testing.assert_close(x['x'], y['x'])
m, s = consensus_scores(({'x': torch.tensor([[-.02, -.03, -.01]])}, {'x': torch.tensor([[.001, .001, .006]])}),
                        ({'x': torch.tensor([[-.01, .01, -.02]])}, {'x': torch.zeros(1, 3)}))
assert trust_masks(m, s)['x'].tolist() == [[True, False, False]]
assert not trust_masks(m, s, budget=.005)['x'].any()
print('PASS: exact pooled moments, conflicting/uncertain tile exclusion, budget fallback', flush=True)

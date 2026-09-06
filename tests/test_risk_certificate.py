"""Run through Slurm; mathematical boundary and negative-control checks."""
import os
assert os.environ.get('SLURM_JOB_ID')
import math
from quantize.risk_certificate import retention_certificate, bayes_sequence_bound

# A large, strongly improving sample can certify retention; a small sample
# with identical observed means must remain inconclusive.
large = retention_certificate([0.1]*10000, [0.8]*10000, {'reference': [0.4]*10000},
                              bound=1., rho=.5, delta=.05)
assert large['certified']
small = retention_certificate([0.1]*4, [0.8]*4, {'reference': [0.4]*4},
                              bound=1., rho=.5, delta=.05)
assert not small['certified']
# Even a real baseline win fails when it does not retain enough reference gain.
failure = retention_certificate([.7]*10000, [.8]*10000, {'reference': [.2]*10000},
                                bound=1., rho=.5, delta=.05)
assert not failure['certified']
# Known range is mandatory; the function must never infer it from sample extrema.
try:
    retention_certificate([2.], [.8], {'reference': [.2]}, bound=1., rho=.5, delta=.05)
except AssertionError:
    pass
else:
    raise AssertionError('Out-of-range loss accepted')
# Numerical mixture identity agrees with explicit normalized two-token experts.
experts = [[.8, .3], [.2, .9]]
nlls = [-sum(math.log(p) for p in expert) for expert in experts]
value = bayes_sequence_bound(nlls)
assert abs(value['mixture_nll'] + math.log((.8*.3+.2*.9)/2)) < 1e-12
assert 0 <= value['regret'] <= math.log(2)+1e-12
# Extreme losses must not overflow/underflow the stable computation.
value = bayes_sequence_bound([1e6, 2e6])
assert abs(value['regret']-math.log(2)) < 1e-9
print('Risk-certificate and mixture-bound checks passed', flush=True)

import torch
# Exact isolated-tile geometry: aligned smaller errors are universally safe;
# moving error into a previously error-free subspace creates an adverse input.
eb = torch.zeros(8, 64, dtype=torch.float64)
eb[:, :8] = 2*torch.eye(8, dtype=torch.float64)
aligned = eb/2
assert torch.linalg.eigvalsh(aligned.T@aligned-eb.T@eb).max() <= 0
shifted = torch.zeros_like(eb)
shifted[:, 8:16] = torch.eye(8, dtype=torch.float64)
a = shifted.T@shifted-eb.T@eb
eigenvalues, eigenvectors = torch.linalg.eigh(a)
x = eigenvectors[:, -1]
assert (shifted@x).square().sum() > (eb@x).square().sum()
# A positive worst-direction eigenvalue can coexist with a robust expected
# error win in a restricted second-moment neighborhood.
s0 = torch.eye(64, dtype=torch.float64)
eta = .5
s = s0 + eta*torch.diag(torch.sign(torch.diag(a)))
bound = torch.trace(a@s0) + eta*eigenvalues.abs().sum()
assert torch.linalg.eigvalsh(s).min() >= 0
assert abs(float(torch.trace(a@s)-bound)) < 1e-12 and bound < 0
print('Local-error geometry and covariance-neighborhood checks passed', flush=True)

# Capturing any positive fraction of two conflicting experts' gains for
# every label would require a predictive vector with total mass above one.
for rho in (.1, .5, 1.):
    required_each = .5**(1-rho)*.9**rho
    assert 2*required_each > 1
print('Conflicting-gain normalization counterexample checked', flush=True)

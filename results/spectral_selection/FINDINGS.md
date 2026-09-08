# Initial spectral-selection mechanism findings

Slurm job 331501 completed the numerical tests and all twelve prespecified
synthetic cases. The frozen protocol is [PROTOCOL.md](PROTOCOL.md); the full
table is [job_331501/REPORT.md](job_331501/REPORT.md), with maps, source hashes,
evaluation geometries, and solver traces in that directory's report.json.

## What was established

- Canonical quantizer candidates, legal 8x64 decisions, monotone objective
  descent, deterministic replay, fixed-candidate scale invariance, zero/tie
  behavior, and the spectral variational identity passed their numerical tests.
- Spectral error decreased in seven cases and was unchanged in five. The
  largest reduction was approximately 21.68% (row-correlated, seed 102).
- Exhaustive enumeration found the same objective as the solver in nine cases.
  The remaining relative objective gaps were 4.83%, 2.64%, and 0.16%.
  Coordinate descent can therefore miss useful combinations even with eight
  decisions. Dense-SVD reference correctness does not imply global optimality.
- All three column-outlier cases retained the baseline, which also achieved
  the exhaustive minimum. Extra format flexibility is not automatically useful.

## Evidence against treating the objective as sufficient

Heavy-tail seed 101 reduces spectral error by only 0.27% while increasing
Frobenius error by 10.89%. Row-correlated seed 102 reduces spectral error by
21.68% while increasing Frobenius error by 6.51%. Because all compared matrices
have the same shape, these are also the relative changes in expected output
error under the unit-trace isotropic input covariance.

These are concrete counterexamples to inferring typical-input improvement
from a smaller worst-case error. They do not establish an LLM task regression,
but weaken the motivation for using pure spectral error as the only objective.
No policy or coefficient was changed in response; every result is retained.

## Next research boundary

This is a small-matrix reference implementation, not a scalable model selector
or evidence of cross-domain generalization. The next useful step is a weight-only
feasibility study of matrix spectra and scalable objective approximations,
with solver accuracy and cost measured independently of text loss. A new
objective (for example a constrained one) would require an explicit theoretical
motivation and a new development protocol, not a coefficient sweep on datasets.

Do not launch a large task evaluation merely on the strength of the seven
spectral improvements. The full transfer manifest and scalable solver remain
unimplemented. Task results must never choose the model's map or configuration.

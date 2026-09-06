"""Distribution-free gain-retention bounds for a declared bounded loss.

This does not certify raw, unbounded perplexity or distribution shift.
Candidate maps must be frozen before IID certification examples are observed.
"""
import math


def retention_certificate(selected, baseline, references, *, bound, rho, delta):
    """Simultaneously certify at least rho of each reference's expected gain.

The caller supplies losses in [0, bound] and is responsible for the sampling
and independence assumptions. Baseline is included as a no-harm constraint.
"""
    assert bound > 0 and 0 < rho <= 1 and 0 < delta < 1
    assert references and len(selected) > 0
    n = len(selected)
    comparisons = {'baseline_no_harm': baseline, **references}
    assert len(comparisons) == len(references)+1, 'Reserved reference name.'
    for values in [selected, baseline, *references.values()]:
        assert len(values) == n
        assert all(math.isfinite(x) and 0 <= x <= bound for x in values)
    radius = 2*bound*math.sqrt(math.log(len(comparisons)/delta)/(2*n))
    entries = {}
    for name, values in comparisons.items():
        mean = sum(s-(1-rho)*b-rho*r for s, b, r in zip(selected, baseline, values))/n
        entries[name] = {'mean_retention_contrast': mean, 'upper_bound': mean+radius}
    return {'certified': all(v['upper_bound'] <= 0 for v in entries.values()),
            'n': n, 'rho': rho, 'delta': delta, 'loss_bound': bound,
            'radius': radius, 'comparisons': entries,
            'assumptions': 'Frozen candidates; independent IID certification data from deployment distribution; bounded loss.'}


def bayes_sequence_bound(cumulative_nll):
    """Exact uniform-prior mixture sequence NLL; requires full expert predictions.

This is an ensemble comparison point, not a static quantized weight map.
"""
    assert cumulative_nll and all(math.isfinite(x) for x in cumulative_nll)
    best = min(cumulative_nll)
    mixture = best - math.log(sum(math.exp(best-x) for x in cumulative_nll)) + math.log(len(cumulative_nll))
    return {'mixture_nll': mixture, 'best_nll': best,
            'regret': mixture-best, 'universal_regret_bound': math.log(len(cumulative_nll))}

import numpy as np

from campaign import stats


def test_finite_monte_carlo_pvalues_are_never_zero():
    a = np.full(12, -2.0)
    b = np.zeros(12)
    tokens = np.ones(12)
    clusters = np.arange(12)
    result = stats.paired_dlogppl(a, b, tokens, clusters, B=99, seed=7, margin=0.1)
    assert result['p_two_sided'] == 2 / 100
    assert result['noninferiority']['p_one_sided'] == 1 / 100


def test_named_accuracy_streams_are_stable_and_independent():
    a = {i: float(i % 3 == 0) for i in range(50)}
    b = {i: float(i % 4 == 0) for i in range(50)}
    x1 = stats.paired_accuracy(a, b, B=200, seed=11, stream_label='task-a')
    x2 = stats.paired_accuracy(a, b, B=200, seed=11, stream_label='task-a')
    y = stats.paired_accuracy(a, b, B=200, seed=11, stream_label='task-b')
    assert np.array_equal(x1['boot'], x2['boot'])
    assert not np.array_equal(x1['boot'], y['boot'])
    assert x1['bootstrap_rng']['stream_label'] == 'task-a'


def test_macro_uses_independent_task_bootstrap_arrays():
    a = {i: float(i % 2) for i in range(40)}
    b = {i: float(i % 5 == 0) for i in range(40)}
    per = {
        'alpha': stats.paired_accuracy(a, b, B=100, stream_label='alpha'),
        'beta': stats.paired_accuracy(a, b, B=100, stream_label='beta'),
    }
    assert not np.array_equal(per['alpha']['boot'], per['beta']['boot'])
    result = stats.macro_accuracy(per)
    assert result['tasks'] == ['alpha', 'beta']
    assert len(result['ci95']) == 2

"""Regression tests for the representative-accuracy block of the decision document.

final_reports.py carried the hard-coded sentence "Accuracy at each k is pending (V40
representative-accuracy jobs)". It was true when written and became false the moment the V40 jobs
finished, at which point the generator would have printed it into N16_DECISION.md - a required
deliverable - while a complete K_SENSITIVITY_ACCURACY.json sat unread in the same run's inputs.
These tests pin the data-derived behaviour so that class of staleness cannot recur silently.
"""


def _cell(diff, lo, hi, below=0, above=0):
    return dict(macro=dict(diff=diff, ci95=[lo, hi], tasks=['a', 'b', 'c', 'd']),
                tasks_with_ci_below_zero=['t'] * below, tasks_with_ci_above_zero=['t'] * above)


def test_absent_artifact_reports_pending():
    from campaign import final_reports as FR
    out = FR.accuracy_block(None, None, 'NULL', 'V40 jobs')
    assert out[0] == 'Accuracy is not available yet (V40 jobs).'


def test_all_missing_cells_report_pending_with_count():
    from campaign import final_reports as FR
    art = dict(models=dict(m1={'p1': dict(status='missing'), 'p2': dict(status='missing')}))
    out = FR.accuracy_block(art, None, 'NULL', 'V80 jobs')
    assert 'not available yet' in out[0] and '2 model x policy cells outstanding' in out[0]


def test_null_result_when_every_ci_spans_zero():
    from campaign import final_reports as FR
    art = dict(reference='four_over_six', models=dict(m1={'n16_k3': _cell(0.0036, -0.0036, 0.0110)}))
    body = '\n'.join(FR.accuracy_block(art, None, 'NULL SENTENCE', 'V40 jobs'))
    assert 'No macro accuracy CI excludes zero' in body
    assert 'NULL SENTENCE' in body
    assert 'is pending' not in body          # the defect this file exists to prevent
    assert '| m1 | n16_k3 | +0.36 | [-0.36, +1.10] |' in body   # percentage points, signed


def test_significant_cell_is_reported_not_suppressed():
    from campaign import final_reports as FR
    art = dict(reference='four_over_six', models=dict(m1={'p': _cell(0.02, 0.01, 0.03)}))
    body = '\n'.join(FR.accuracy_block(art, None, 'NULL', 'V40 jobs'))
    assert '1 of 1 model x policy cells have a macro accuracy CI excluding zero' in body
    assert 'm1 p' in body


def test_partial_artifact_reports_rows_and_outstanding_count():
    from campaign import final_reports as FR
    art = dict(reference='ref', models=dict(m1={'good': _cell(0.001, -0.002, 0.004),
                                                'bad': dict(status='missing')}))
    body = '\n'.join(FR.accuracy_block(art, None, 'NULL', 'V80 jobs'))
    assert '| m1 | good |' in body
    assert '1 further model x policy cells are still outstanding (V80 jobs).' in body

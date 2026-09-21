"""V43 noise-floor control: block statistics, drift, and the resolution context, dose-response and
order-confound controls added to the fidelity metrics. All CPU, no model and no GPU."""
import math

import numpy as np

from campaign import fidelity_noise as FN


def _m(block, state, ce, kl=0.0, tile=None):
    return dict(block=block, state=state, tile=tile, ce=ce, kl=kl, ce_per_seq=[ce], kl_per_seq=[kl], seconds=1.0, utc='x')


def test_pooled_within_sd_ignores_block_offsets():
    a = [1.0, 1.1, 0.9, 1.0]
    b = [5.0, 5.1, 4.9, 5.0]  # identical spread, large offset
    pooled = FN.pooled_within_sd([a, b])
    assert abs(pooled - float(np.std(a, ddof=1))) < 1e-12
    # pooling the two blocks naively would be dominated by the offset, not the noise
    assert float(np.std(a + b, ddof=1)) > 10 * pooled


def test_pooled_within_sd_needs_two_points():
    assert FN.pooled_within_sd([[1.0]]) is None
    assert FN.pooled_within_sd([]) is None


def test_replicates_for_3se_scales_as_inverse_square_effect():
    assert FN.replicates_for_3se(1e-3, 3e-7) == math.ceil(18.0 * (1e-3 ** 2) / (3e-7 ** 2))
    assert FN.replicates_for_3se(1e-3, 6e-7) < FN.replicates_for_3se(1e-3, 3e-7)
    assert FN.replicates_for_3se(0.0, 1e-6) is None
    assert FN.replicates_for_3se(1e-3, 0.0) is None


def test_block_stats_shapes():
    s = FN.block_stats([1.0, 2.0, 3.0])
    assert s['n'] == 3 and s['mean'] == 2.0 and s['range'] == 2.0
    assert FN.block_stats([])['n'] == 0
    assert FN.block_stats([1.0])['sd'] is None


def test_summarize_reports_zero_noise_when_measurements_repeat_exactly():
    """The observed case on an A6000: consecutive identical-weight evaluations are bitwise equal.
    The spread must be reported as exactly zero, not as the ~1e-16 residue of subtracting a mean."""
    meas = [_m('baseline', 'baseline', 1.5349352)]
    meas += [_m('repeat', 'baseline', 1.5349352) for _ in range(6)]
    s = FN.summarize(meas, [])
    assert s['noise']['ce']['single_measurement_sd'] == 0.0
    assert s['noise']['ce']['difference_of_two_measurements_sd'] == 0.0
    assert s['noise']['ce']['max_abs_deviation_from_first_baseline'] == 0.0
    assert s['noise']['ce']['all_baselines']['range'] == 0.0
    assert s['repeat']['ce']['sd'] == 0.0
    assert FN.pooled_within_sd([[1.5349352] * 6]) == 0.0


def test_summarize_reports_noise_and_drift():
    meas = [_m('baseline', 'baseline', 1.0)]
    meas += [_m('repeat', 'baseline', v) for v in (1.000, 1.001, 0.999, 1.000)]
    meas += [_m('drift', 'baseline', v) for v in (1.010, 1.011, 1.009, 1.010)]
    s = FN.summarize(meas, [])
    assert s['repeat']['ce']['n'] == 4
    assert s['noise']['ce']['single_measurement_sd'] > 0
    assert s['noise']['ce']['difference_of_two_measurements_sd'] > s['noise']['ce']['single_measurement_sd']
    d = s['noise']['ce']['drift_end_minus_start']
    assert abs(d['delta'] - 0.010) < 1e-9
    assert d['t'] > 5  # a drift far larger than the repeat noise, which V43 would charge to the tile


def test_summarize_tile_effect_against_replicate_noise():
    meas = [_m('tiles', 'baseline', v) for v in (1.000, 1.001, 0.999, 1.000)]
    meas += [_m('tiles', 'tile_on', v, tile=dict(module='m', tile=7)) for v in (0.990, 0.991, 0.989, 0.990)]
    meas += [_m('repeat', 'baseline', v) for v in (1.000, 1.001, 0.999, 1.000)]
    tiles = [dict(label='selected_best', module='m', tile=7, U3=-1.0, pred_ce=-1e-6, pred_kl=-1e-6)]
    s = FN.summarize(meas, tiles)
    row = s['tiles'][0]
    assert row['label'] == 'selected_best'
    assert row['ce']['replicates'] == 4 and row['ce']['baselines'] == 4
    assert abs(row['ce']['effect'] + 0.010) < 1e-9
    assert row['ce']['t'] < -5
    # a predicted effect of 1e-6 against this noise needs an absurd number of replicates
    assert row['ce']['replicates_for_3se_of_prediction'] > 1000


def _rec(stratum, i, actual, pred, u3, layer_third=0):
    return dict(stratum=stratum, U3=u3, pred_ce=pred, pred_kl=pred, actual_ce=actual, actual_kl=actual,
                actual_ce_se=5e-4, actual_kl_se=5e-4,
                actual_ce_per_seq=[actual + 1e-6 * j for j in range(8)], actual_kl_per_seq=[actual + 1e-6 * j for j in range(8)],
                module='m', tile=i, layer=layer_third, layer_third=layer_third, module_type='q_proj')


def _records():
    """Strata in V43 measurement order, with a deliberate block shift inside the last (random) block."""
    rs = [_rec('selected', i, -1e-4 + 1e-5 * i, -1e-6, -2e-6 - 1e-7 * i, 0) for i in range(4)]
    rs += [_rec('rejected', 10 + i, 1e-4 + 1e-5 * i, 1e-6, 5e-6 + 1e-7 * i, 1) for i in range(4)]
    rs += [_rec('random', 20 + i, (1e-4 if i < 4 else 1e-3) + 1e-6 * i, 1e-6, 1e-6 * (i + 1), 2) for i in range(8)]
    return rs


def _dose_records():
    """A random stratum whose measured effect tracks the predicted one exactly."""
    return [_rec('random', i, 1e-4 * i, 1e-6 * i, 1e-6 * i, 2) for i in range(8)]


def test_paired_aggregate_uses_sequence_pairing():
    from campaign import analyze_misc as AM
    agg = AM.paired_aggregate([r for r in _records() if r['stratum'] == 'selected'], 'ce')
    assert agg['tiles'] == 4 and agg['sequences'] == 8
    assert agg['se'] is not None and agg['t'] < 0
    assert AM.paired_aggregate([dict(stratum='x', pred_ce=0.0)], 'ce') is None


def test_stratum_contrast_is_generic_over_strata():
    from campaign import analyze_misc as AM
    c = AM.stratum_contrast(_records(), 'ce')
    assert c['a'] == 'selected' and c['b'] == 'rejected'
    assert c['n_a'] == 4 and c['n_b'] == 4
    assert c['difference'] < 0 and 0.0 <= c['welch_p'] <= 1.0
    assert AM.stratum_contrast([r for r in _records() if r['stratum'] == 'selected'], 'ce') is None


def test_rejected_vs_random_is_a_dose_response_not_a_null():
    """`rejected` is the upper half of positive-U3 tiles and `random` is the whole population, so
    these two strata differ by construction - this pair is not a null comparison."""
    from campaign import analyze_misc as AM
    n = AM.stratum_contrast(_records(), 'ce', 'rejected', 'random')
    assert n['a'] == 'rejected' and n['b'] == 'random'
    assert n['n_a'] == 4 and n['n_b'] == 8
    assert isinstance(n['difference'], float)


def test_dose_response_detects_ordering_within_one_stratum():
    from campaign import analyze_misc as AM
    o = AM.dose_response(_dose_records(), 'ce')
    assert o['stratum'] == 'random' and o['n'] == 8
    assert o['spearman_u3_vs_actual'] > 0.99 and o['p_u3'] < 0.05
    assert o['spearman_pred_vs_actual'] > 0.99
    assert o['high_minus_low'] > 0 and o['welch_p'] < 0.05
    assert AM.dose_response(_dose_records()[:4], 'ce') is None  # needs at least 6 tiles


def test_order_confound_detects_a_block_shift_inside_one_stratum():
    from campaign import analyze_misc as AM
    o = AM.order_confound(_records(), 'ce')
    assert o['stratum'] == 'random'
    assert abs(o['half_difference'] + 9e-4) < 1e-5
    assert o['half_welch_p'] < 0.05
    assert o['trend_spearman'] > 0 and 'trend_total_over_all_records' in o


def test_fidelity_metrics_reports_resolution_context_and_controls():
    from campaign import analyze_misc as AM
    out = AM.fidelity_metrics(_records())
    # |actual| ~1e-4 against an SE of 5e-4: nothing is resolved
    assert out['ce']['resolvable_fraction_3se'] == 0.0
    assert out['ce']['actual_se_median'] == 5e-4
    # the predicted effect is orders of magnitude under the measurement's own SE
    assert out['ce']['predicted_over_actual_se_median'] < 0.01
    assert out['ce']['selected_vs_rejected']['welch_p'] is not None
    # the contrast must ship with the dose-response test that is free of block confounds
    assert out['ce']['rejected_vs_random']['a'] == 'rejected'
    assert out['ce']['dose_response_within_random']['n'] == 8
    assert out['ce']['order_confound']['half_welch_p'] < 0.05
    assert out['ce']['paired_aggregate_by_stratum']['selected']['tiles'] == 4

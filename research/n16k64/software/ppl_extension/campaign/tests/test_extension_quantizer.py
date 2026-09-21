import pytest
import torch

from campaign import quant as Q


def _weights(seed=20260914):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(32, 128, generator=g) * 0.02).bfloat16()
    w[0, 0] = 1.0
    return w


@pytest.mark.parametrize("fmt,anchor", [("four_over_six", Q.four_over_six), ("e0m3", Q.e0m3)])
def test_weight_scale_one_is_bitwise_anchor(fmt, anchor):
    w = _weights()
    assert torch.equal(Q.format_preserving_weight_scale(w, fmt, 1.0), anchor(w))


@pytest.mark.parametrize("fmt", ["four_over_six", "e0m3"])
@pytest.mark.parametrize("multiplier", [0.95, 1.05])
def test_weight_scale_preserves_shape_dtype_and_finiteness(fmt, multiplier):
    out = Q.format_preserving_weight_scale(_weights(), fmt, multiplier)
    assert out.shape == (32, 128)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_activation_percentile_100_is_bitwise_anchor():
    g = torch.Generator().manual_seed(8)
    x = (torch.randn(2, 7, 64, generator=g) * 3).bfloat16()
    assert torch.equal(Q.four_over_six_rows_percentile_100(x), Q.ACTIVATION["four_over_six_rows"](x))


@pytest.mark.parametrize("name", ["four_over_six_rows_mse_grid",
                                  "four_over_six_rows_percentile_999",
                                  "four_over_six_rows_percentile_9999",
                                  "four_over_six_rows_percentile_100"])
def test_new_activation_rules_are_row_independent_and_chunk_stable(name):
    g = torch.Generator().manual_seed(9)
    x = (torch.randn(3, 5, 64, generator=g) * 4).bfloat16()
    fn = Q.ACTIVATION[name]
    whole = fn(x)
    assert whole.dtype == torch.bfloat16 and torch.isfinite(whole).all()
    rows = torch.stack([fn(x.reshape(-1, 64)[i:i + 1])[0] for i in range(15)]).reshape_as(x)
    assert torch.equal(whole, rows)
    assert torch.equal(Q.chunked_rows(fn, x, max_rows=4), whole)


def test_mse_grid_never_has_higher_float_candidate_error_than_absmax():
    g = torch.Generator().manual_seed(10)
    x = (torch.randn(11, 64, generator=g) * 2).bfloat16()
    x[:, 0] *= 8
    got = Q.four_over_six_rows_mse_grid(x).float()
    baseline = Q.ACTIVATION["four_over_six_rows"](x).float()
    got_error = (got - x.float()).square().sum(-1)
    baseline_error = (baseline - x.float()).square().sum(-1)
    # Selection uses pre-BF16 float candidates; BF16 storage can move the comparison by roundoff.
    assert torch.all(got_error <= baseline_error + 1e-4 * baseline_error.clamp_min(1e-12))


def test_installer_records_and_applies_weight_and_activation_overrides():
    from campaign.policies import Installer

    linear = torch.nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(_weights(seed=11)[:16, :64])
    pristine = linear.weight.detach().clone()
    holder = torch.nn.Module()
    holder.lin = linear
    installer = Installer("llama8b", holder, {"lin": linear}, cache="cpu")
    info = installer.install({"name": "screen", "kind": "four_over_six",
                              "weight_scale_multiplier": 0.95,
                              "activation_kind": "four_over_six_rows_percentile_999"})
    assert info["weight_scale_multiplier"] == 0.95
    assert info["activation"] == "four_over_six_rows_percentile_999"
    assert torch.equal(linear.weight, Q.format_preserving_weight_scale(pristine, "four_over_six", 0.95))
    installer.remove()

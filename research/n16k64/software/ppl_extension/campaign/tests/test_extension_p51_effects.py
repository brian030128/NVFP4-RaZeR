import pytest
import torch


def test_tile_slices_and_bounds():
    from campaign.extension_p51_effects import tile_slices

    rs, cs = tile_slices((32, 128), 1, 1)
    x = torch.arange(32 * 128).reshape(32, 128)
    assert x[rs, cs].shape == (16, 64)
    assert (rs.start, rs.stop, cs.start, cs.stop) == (16, 32, 64, 128)
    with pytest.raises(IndexError):
        tile_slices((32, 128), 2, 0)
    with pytest.raises(ValueError):
        tile_slices((31, 128), 0, 0)


def test_stderr_uses_sequence_unit():
    from campaign.extension_p51_effects import stderr

    assert stderr([1.0]) == 0.0
    torch.testing.assert_close(torch.tensor(stderr([1.0, 3.0])), torch.tensor(1.0))


def test_installed_weight_digest_detects_and_accepts_exact_restoration():
    from collections import OrderedDict
    from campaign.extension_p51_effects import installed_weight_sha256

    modules = OrderedDict((name, torch.nn.Linear(4, 2, bias=False))
                          for name in ("first", "second"))
    pristine = modules["first"].weight.detach().clone()
    baseline = installed_weight_sha256(modules)
    with torch.no_grad():
        modules["first"].weight[0, 0] += 1
    assert installed_weight_sha256(modules) != baseline
    with torch.no_grad():
        modules["first"].weight.copy_(pristine)
    assert installed_weight_sha256(modules) == baseline

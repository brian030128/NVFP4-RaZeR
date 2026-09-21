import pytest
import torch


def _brute_reconstruction_scores(x_chunks, w, qf, qe):
    x = torch.cat([chunk.float() for chunk in x_chunks])
    base_error = x @ (qf.float() - w.float()).T
    scores = []
    for k0 in range(0, w.shape[1], 64):
        candidate = qf.float().clone()
        candidate[:, k0:k0 + 64] = qe.float()[:, k0:k0 + 64]
        candidate_error = x @ (candidate - w.float()).T
        scores.append((candidate_error.square() - base_error.square()).mean())
    return torch.stack(scores).double()


def test_reconstruction_scores_match_independent_brute_force():
    from campaign.extension_selector_stats import reconstruction_scores

    generator = torch.Generator().manual_seed(20260914)
    w = torch.randn(16, 128, generator=generator, dtype=torch.float32)
    qf = w + 0.03 * torch.randn(16, 128, generator=generator)
    qe = w + 0.05 * torch.randn(16, 128, generator=generator)
    chunks = [torch.randn(3, 128, generator=generator),
              torch.randn(5, 128, generator=generator)]
    expected = _brute_reconstruction_scores(chunks, w, qf, qe)
    got = reconstruction_scores(chunks, w, qf, qe)
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)


def test_reconstruction_scores_reject_empty_or_nondivisible_inputs():
    from campaign.extension_selector_stats import reconstruction_scores

    w = torch.zeros(16, 64)
    with pytest.raises(ValueError, match="no reconstruction inputs"):
        reconstruction_scores([], w, w, w)
    with pytest.raises(ValueError, match="not tile divisible"):
        reconstruction_scores([torch.zeros(1, 63)], torch.zeros(16, 63),
                              torch.zeros(16, 63), torch.zeros(16, 63))


def test_reserved_heldout_selector_models_are_explicit():
    from campaign.extension_selector_stats import HELDOUT

    assert HELDOUT == ("granite8b", "falcon3_10b")

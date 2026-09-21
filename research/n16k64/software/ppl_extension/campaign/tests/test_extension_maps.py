import torch


def test_global_selection_ties_follow_module_then_row_major():
    from campaign.extension_maps import select_global
    names = ["z", "a"]
    scores = {"z": torch.zeros(2, 2), "a": torch.zeros(2, 2)}
    got = select_global(names, scores, 3)
    assert int(got["a"].sum()) == 3
    assert int(got["z"].sum()) == 0
    assert got["a"].reshape(-1).tolist() == [True, True, True, False]


def test_per_module_selection_and_descending():
    from campaign.extension_maps import select_per_module
    scores = {"x": torch.tensor([3.0, 2.0, 2.0, 1.0]), "y": torch.tensor([0.0, 5.0])}
    got = select_per_module(["x", "y"], scores, {"x": 2, "y": 1}, descending=True)
    assert got["x"].tolist() == [True, True, False, False]
    assert got["y"].tolist() == [False, True]


def test_named_random_stream_is_repeatable_and_scope_distinct():
    from campaign.extension_maps import random_scores
    names = ["b", "a"]
    shapes = {"a": (2, 3), "b": (1, 4)}
    a, _ = random_scores(names, shapes, "global", 2026091401)
    b, _ = random_scores(names, shapes, "global", 2026091401)
    c, _ = random_scores(names, shapes, "per_module", 2026091401)
    assert all(torch.equal(a[n], b[n]) for n in names)
    assert any(not torch.equal(a[n], c[n]) for n in names)


def test_fixed_consensus_uses_frozen_lexicographic_keys():
    from campaign.extension_maps import _fixed_consensus
    names = ["z", "a"]
    votes = {"z": torch.tensor([4, 4]), "a": torch.tensor([4, 4])}
    med = {"z": torch.tensor([0.0, 0.0]), "a": torch.tensor([0.0, 0.0])}
    avg = {"z": torch.tensor([0.0, 0.0]), "a": torch.tensor([0.0, 0.0])}
    got = _fixed_consensus(names, votes, med, avg, 3)
    assert got["a"].tolist() == [True, True]
    assert got["z"].tolist() == [True, False]


def test_writer_supports_per_map_calibration_manifest_identity():
    import inspect
    from campaign.extension_maps import Writer
    signature = inspect.signature(Writer.write)
    assert "calibration_manifest_sha256" in signature.parameters

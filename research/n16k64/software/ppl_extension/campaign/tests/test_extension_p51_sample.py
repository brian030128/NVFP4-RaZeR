import torch


def test_stratum_allocation_exact_and_capped():
    from campaign.extension_p51_sample import allocate_strata

    sizes = {(0,): 1, (1,): 9, (2,): 20}
    got = allocate_strata(sizes, 15)
    assert sum(got.values()) == 15
    assert got[(0,)] == 1
    assert all(1 <= got[k] <= sizes[k] for k in sizes)
    assert allocate_strata({(0,): 2, (1,): 2}, 4) == {(0,): 2, (1,): 2}


def test_microcell_round_robin_exact_and_deterministic():
    from campaign.extension_p51_sample import micro_counts

    sizes = {("a", 0, 0): 1, ("b", 1, 1): 10, ("c", 2, 2): 10}
    first = micro_counts("model", sizes, 7)
    second = micro_counts("model", sizes, 7)
    assert first == second and sum(first.values()) == 7
    assert first[("a", 0, 0)] == 1
    assert all(first[k] <= sizes[k] for k in sizes)


def test_stable_quintiles_use_canonical_tie_order():
    from campaign.extension_p51_sample import stable_quintiles

    values = torch.zeros(10)
    assert stable_quintiles(values).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_relative_quintile_categories():
    from campaign.extension_p51_sample import rel_quintile

    assert rel_quintile(0, 4) == 0
    assert rel_quintile(2, 2) == 1
    assert rel_quintile(4, 0) == 2

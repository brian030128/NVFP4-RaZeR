"""
    CPU tests for the co-clustering reorder search.

        python tests/test_reorder.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize.reorder import (              # noqa: E402
    _labels_from_order, elect_mask, expand_chunk_perm, gain_features, objective,
    scale_block_gain, search_permutation, shuffle_control, tile_value,
)
from quantize.quantizer import _elect_e0m3, row_preference   # noqa: E402


def _tiled_score(gain, row_perm, chunk_perm, block_m, chunks, rule, margin):
    """ Brute-force recomputation of the objective straight from a permuted tag grid. """
    g = gain[row_perm][:, chunk_perm]
    M, N = g.shape
    pad_m, pad_n = (-M) % block_m, (-N) % chunks
    g = torch.nn.functional.pad(g, (0, pad_n, 0, pad_m))
    tiles = g.reshape(g.shape[0] // block_m, block_m, g.shape[1] // chunks, chunks)
    tiles = tiles.permute(0, 2, 1, 3).reshape(-1, block_m * chunks)
    phi   = gain_features(tiles).sum(dim=1)
    return float(tile_value(phi, rule, margin, block_m * chunks).sum())


def test_elect_matches_quantizer():
    """ The additive-statistic election must agree cell-for-cell with `_elect_e0m3`. """
    torch.manual_seed(0)
    gain = torch.randn(64, 12, 1).double() * 0.1
    gain[3] = gain[3].abs()                       # a dominance-eligible tile
    ref = gain.squeeze(-1)

    for rule, margin in [("argmin", 0.0), ("dominance", 0.0), ("harm", 1.0), ("harm", 1.5),
                         ("harm", 3.0), ("vote", 0.5), ("margin", 1.0), ("margin", 2.0),
                         ("never", 0.0), ("always", 0.0)]:
        want = _elect_e0m3(gain, rule=rule, margin=margin, ref=gain.abs()).flatten()
        phi  = gain_features(ref).sum(dim=1)
        got  = elect_mask(phi, rule, margin, ref.shape[1])
        assert torch.equal(want, got), f"election mismatch for {rule}/{margin}"
    print("  [ok] elect_mask matches _elect_e0m3 on all rules")


def test_gain_grid_matches_row_preference():
    """ The tag grid summed over a row must be exactly `row_preference`. """
    torch.manual_seed(1)
    w = torch.randn(37, 128).float()
    for clip in ("base", "heade0"):
        g    = scale_block_gain(w, 16, "mse", clip)
        want = row_preference(w, 16, "mse", clip)
        assert torch.allclose(g.sum(dim=1).float(), want.float(), atol=1e-5), clip
    print("  [ok] scale_block_gain row-sums == row_preference")


def test_objective_matches_bruteforce():
    torch.manual_seed(2)
    gain = torch.randn(48, 20).double()
    feat = gain_features(gain)
    for block_m, chunks, rule, margin in [(8, 4, "argmin", 0.0), (16, 2, "harm", 1.5),
                                          (8, 5, "margin", 1.0)]:
        rp = torch.randperm(48)
        cp = torch.randperm(20)
        rlab = _labels_from_order(rp, block_m, 48)
        clab = _labels_from_order(cp, chunks, 20)
        got  = objective(feat, rlab, clab, 48 // block_m, 20 // chunks, rule, margin,
                         block_m * chunks)
        want = _tiled_score(gain, rp, cp, block_m, chunks, rule, margin)
        assert abs(got - want) < 1e-8, f"{rule}: {got} vs {want}"
    print("  [ok] incremental objective == brute force")


def test_planted_structure_is_recovered():
    """
        A planted rank-1 checkerboard: row sign a_i, column sign b_j, cell g = a_i b_j |x| + noise.
        A product partition can represent this exactly, so the search should reach near the 1x16
        ceiling from a shuffled start, and the returned permutations must reproduce the score.
    """
    torch.manual_seed(3)
    M, N, BM, C = 256, 64, 8, 4
    a = torch.where(torch.rand(M) < 0.5, -1.0, 1.0).double()
    b = torch.where(torch.rand(N) < 0.5, -1.0, 1.0).double()
    gain = a[:, None] * b[None, :] * torch.rand(M, N).double() + 0.05 * torch.randn(M, N).double()

    res = search_permutation(gain, BM, C * 16, rule="argmin", seed=0)
    chk = _tiled_score(gain, res["row_perm"], res["chunk_perm"], BM, C, "argmin", 0.0)

    assert abs(chk - res["score"]) < 1e-6, f"permutation does not reproduce score: {chk} vs {res['score']}"
    assert res["recovered"] > 0.90, f"planted structure not recovered: {res['recovered']:.3f}"
    assert res["recovered"] > res["baseline_recovered"] + 0.3, \
        f"no lift over shuffled baseline: {res['baseline_recovered']:.3f} -> {res['recovered']:.3f}"
    print(f"  [ok] planted checkerboard: {res['baseline_recovered']:.3f} -> {res['recovered']:.3f} "
          f"(init {res['init']})")


def test_noise_floor_is_measured_not_assumed():
    """
        The negative control, and the reason `shuffle_control` exists.

        A balanced co-clustering search with thousands of tiles and full permutation freedom will
        concentrate positive mass even in i.i.d. cells -- that is overfitting, not structure. So the
        test is NOT "the search finds nothing in noise" (it does, ~0.38 of the ceiling against an
        identity baseline of ~0.18); it is that the noise floor is far below what real structure
        gives, and that shuffling the CELLS of a structured grid collapses it to that same floor.
    """
    torch.manual_seed(4)
    noise = torch.randn(256, 64).double()
    floor = search_permutation(noise, 8, 64, rule="argmin", seed=0)
    assert floor["recovered"] < 0.55, f"noise floor implausibly high: {floor['recovered']:.3f}"
    assert floor["recovered"] > floor["baseline_recovered"], "search should beat identity on noise"

    # a planted grid, then the same grid with its cells shuffled -> must fall back to the floor
    a = torch.where(torch.rand(256) < 0.5, -1.0, 1.0).double()
    b = torch.where(torch.rand(64) < 0.5, -1.0, 1.0).double()
    planted = a[:, None] * b[None, :] * torch.rand(256, 64).double()
    real = search_permutation(planted, 8, 64, rule="argmin", seed=0)
    ctrl = search_permutation(shuffle_control(planted, torch.Generator().manual_seed(7)),
                              8, 64, rule="argmin", seed=0)
    assert real["recovered"] > ctrl["recovered"] + 0.3, \
        f"structure not separable from its own shuffle: {real['recovered']:.3f} vs {ctrl['recovered']:.3f}"
    print(f"  [ok] noise floor {floor['recovered']:.3f}; planted {real['recovered']:.3f} "
          f"vs its cell-shuffle {ctrl['recovered']:.3f}")


def test_search_never_loses_to_identity():
    torch.manual_seed(5)
    for rule, margin in [("argmin", 0.0), ("harm", 1.5), ("margin", 1.0)]:
        gain = torch.randn(128, 32).double() * torch.rand(128, 1).double()
        res  = search_permutation(gain, 8, 64, rule=rule, margin=margin, seed=0)
        assert res["score"] >= res["baseline"] - 1e-9, \
            f"{rule}: search {res['score']} < identity {res['baseline']}"
    print("  [ok] search never scores below the identity order")


def test_chunk_perm_expansion():
    cp = torch.tensor([2, 0, 1])
    got = expand_chunk_perm(cp, 4)
    assert torch.equal(got, torch.tensor([8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7]))
    print("  [ok] chunk permutation expands to a column permutation")


def test_permuted_weights_reproduce_gain():
    """
        End-to-end: permuting the actual weight matrix by the returned permutations must produce a
        tag grid equal to the permuted tag grid -- i.e. moving whole 16-column chunks really does
        leave G invariant, which is the assumption the whole search rests on.
    """
    torch.manual_seed(6)
    w    = torch.randn(64, 256).float() * torch.rand(64, 1)
    gain = scale_block_gain(w, 16, "mse", "heade0")
    res  = search_permutation(gain, 8, 64, rule="harm", margin=1.5, seed=0)

    cols = expand_chunk_perm(res["chunk_perm"], 16)
    g2   = scale_block_gain(w[res["row_perm"]][:, cols], 16, "mse", "heade0")
    g_ref = gain[res["row_perm"]][:, res["chunk_perm"]]
    assert torch.allclose(g2, g_ref, atol=1e-9), "chunk permutation changed the tag grid"
    print("  [ok] 16-column chunk permutation leaves the tag grid invariant")


def test_quantizer_permute_modes():
    """
        The `cocl` / `coclcol` family, end to end through `quant_mix_4_6`.

        Three things must hold: the permutation round-trip is exact (quantizing the permuted tensor
        and undoing the permutation gives the same answer as the built-in mode), the co-clustered
        result is no worse than the unpermuted one on the loss the election optimizes, and the
        name parser reaches all three modes.
    """
    from quantize.quantizer import parse_mix_4_6_dtype, quant_mix_4_6
    from quantize.reorder import expand_chunk_perm

    torch.manual_seed(7)
    w = (torch.randn(256, 512) * torch.rand(256, 1)).bfloat16()
    kw = dict(groupsize=16, type_block=(8, 64), clip="heade0", elect="harm", margin=1.5)

    for name, want in [("mix_4_6_cocl_h1.5", "cocluster"),
                       ("mix_4_6_coclcol_h1.5", "colchunk"),
                       ("mix_4_6_coclrow_h1.5", "coclrows")]:
        assert parse_mix_4_6_dtype(name)[7] == want, name

    base = quant_mix_4_6(w, **kw, permute="none")
    cocl = quant_mix_4_6(w, **kw, permute="cocluster")
    ccol = quant_mix_4_6(w, **kw, permute="colchunk")
    for out in (base, cocl, ccol):
        assert out.shape == w.shape and out.dtype == torch.bfloat16

    err = lambda q: float((q.float() - w.float()).pow(2).sum())
    assert err(cocl) <= err(base), f"co-clustering raised the error: {err(cocl)} > {err(base)}"
    assert err(ccol) <= err(base), f"column reordering raised the error: {err(ccol)} > {err(base)}"

    # round-trip: permute by hand, quantize unpermuted, undo -> must match `colchunk` exactly
    gscale = (w.float().abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
    gain   = scale_block_gain(w.float() / gscale, 16, "mse", "heade0")
    found  = search_permutation(gain, 8, 64, 16, rule="harm", margin=1.5, axes="cols")
    cols   = expand_chunk_perm(found["chunk_perm"], 16)
    manual = quant_mix_4_6(w[:, cols], **kw, permute="none")
    undone = torch.empty_like(manual).index_copy_(1, cols, manual)
    assert torch.equal(undone, ccol), "colchunk round-trip does not match a hand-applied permutation"
    print(f"  [ok] quant_mix_4_6 permute modes: sse none={err(base):.4g} "
          f"colchunk={err(ccol):.4g} cocluster={err(cocl):.4g}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    test_elect_matches_quantizer()
    test_gain_grid_matches_row_preference()
    test_objective_matches_bruteforce()
    test_chunk_perm_expansion()
    test_permuted_weights_reproduce_gain()
    test_planted_structure_is_recovered()
    test_noise_floor_is_measured_not_assumed()
    test_search_never_loses_to_identity()
    test_quantizer_permute_modes()
    print("\nall reorder tests passed")

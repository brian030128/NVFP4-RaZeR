"""
    CPU tests for the MixFP4 fake quantizer.

    Run with:  python tests/test_mixfp4.py     (or: pytest tests/test_mixfp4.py)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize.quantizer import (
    _tile_type_blocks,
    _untile_type_blocks,
    quant_mixfp4,
    quant_mix_4_6,
    quant_nvif4,
    quant_nvfp4_4over6,
)
from quantize.utils import parse_type_block, format_type_block


TYPE_BLOCKS = ["1x16", "16x16", "256x16", "32x64", "32x128"]

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E0M3_GRID = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


def test_parse_type_block():
    assert parse_type_block("32x128") == (32, 128)
    assert parse_type_block("1X16") == (1, 16)
    assert parse_type_block((256, 16)) == (256, 16)
    assert format_type_block([32, 64]) == "32x64"

    for bad in ["32x24", "0x16", "32", "32x16x2"]:
        try:
            parse_type_block(bad)
        except AssertionError:
            continue
        raise AssertionError(f"parse_type_block should have rejected {bad!r}")
    print("ok  parse_type_block")


def test_tiling_roundtrip():
    for block_m, block_k in [(1, 16), (16, 16), (256, 16), (32, 64), (32, 128)]:
        for num_row in [128, 300, 4096]:
            x = torch.randn(num_row, 512)
            tiled, meta = _tile_type_blocks(x, block_m, block_k, 16)
            assert tiled.shape[1] == block_m * block_k // 16, tiled.shape
            assert tiled.shape[2] == 16
            back = _untile_type_blocks(tiled, block_m, block_k, meta)
            assert torch.equal(back, x), f"round-trip failed for {block_m}x{block_k}, rows={num_row}"
    print("ok  tiling round-trip (incl. zero-padded outer dimension)")


def test_tiling_groups_are_contiguous_along_k():
    """A scale block must be 16 consecutive elements of one row, inside one type block."""
    num_row, num_col = 64, 256
    x = torch.arange(num_row * num_col, dtype=torch.float32).view(num_row, num_col)
    tiled, _ = _tile_type_blocks(x, 32, 128, 16)
    # every scale block holds 16 consecutive values -> constant stride of 1
    diff = tiled[:, :, 1:] - tiled[:, :, :-1]
    assert torch.all(diff == 1), "scale blocks are not 16 contiguous elements along K"
    print("ok  scale blocks are 16 contiguous elements along K")


def test_shape_and_dtype_preserved():
    for shape in [(128, 256), (2, 64, 512), (2, 8, 37, 128)]:
        x = torch.randn(*shape).to(torch.bfloat16)
        for tb in TYPE_BLOCKS:
            y = quant_mixfp4(x, groupsize=16, type_block=tb)
            assert y.shape == x.shape, (shape, tb, y.shape)
            assert y.dtype == torch.bfloat16
    print("ok  shape / dtype preserved for 2D, 3D and 4D inputs")


def test_narrow_reduction_dim_shrinks_type_block():
    """head_dim=64 with a 32x128 type block: K shrinks to the full row instead of failing."""
    x = torch.randn(4, 8, 37, 64).to(torch.bfloat16)
    y = quant_mixfp4(x, groupsize=16, type_block="32x128")
    assert y.shape == x.shape
    print("ok  narrow reduction dimension falls back to a full-row type block")


def test_1x16_matches_nvif4():
    """
        With a 1x16 type block, the type block IS the scale block, so MixFP4 must reproduce the
        existing per-block FP4/INT4 selection of NVIF4 exactly.
    """
    torch.manual_seed(0)
    for shape in [(512, 512), (128, 4096)]:
        x = torch.randn(*shape).to(torch.bfloat16)
        a = quant_mixfp4(x, groupsize=16, type_block="1x16")
        b = quant_nvif4(x, groupsize=16)
        assert torch.equal(a, b), "mixfp4 1x16 diverges from nvif4"
    print("ok  mixfp4 1x16 == nvif4 (per-scale-block type selection)")


def _fits_grid(values, grid, atol=0.06):
    """
        True if every value of a scale block lies on `grid` up to one positive scale factor.

        The block scale is unknown here, so the factor is inferred by matching the block maximum
        against each nonzero grid point in turn (the block maximum is not necessarily the largest
        grid point -- a block whose values top out at 4 never uses the E2M1 value 6). `atol` is in
        grid units and is loose enough to absorb the bfloat16 rounding of the dequantized output,
        yet far below the 0.5 spacing of both grids.
    """
    vals = values.abs().flatten()
    peak = vals.max()
    if peak == 0:
        return True
    for grid_point in grid[grid > 0]:
        rescaled = vals * (grid_point / peak)
        dist     = (rescaled.unsqueeze(-1) - grid).abs().min(dim=-1).values
        if dist.max().item() <= atol:
            return True
    return False


def _type_block_grid_conflicts(y, block_m, block_k):
    """
        Count type blocks that contain one scale block that only fits E2M1 and another that only
        fits E0M3, i.e. type blocks whose element data type is not uniform.
    """
    tiled, _  = _tile_type_blocks(y.to(torch.float32), block_m, block_k, 16)
    conflicts = 0
    unknown   = 0
    for i in range(tiled.shape[0]):
        only_e2m1 = only_e0m3 = 0
        for j in range(tiled.shape[1]):
            blk       = tiled[i, j]
            fits_e2m1 = _fits_grid(blk, E2M1_GRID)
            fits_e0m3 = _fits_grid(blk, E0M3_GRID)
            assert fits_e2m1 or fits_e0m3, f"scale block ({i},{j}) is on neither grid"
            only_e2m1 += int(fits_e2m1 and not fits_e0m3)
            only_e0m3 += int(fits_e0m3 and not fits_e2m1)
        if only_e2m1 and only_e0m3:
            conflicts += 1
        if not only_e2m1 and not only_e0m3:
            unknown += 1
    return conflicts, unknown, tiled.shape[0]


def test_one_data_type_per_type_block():
    """
        Every scale block inside a type block must use the same element grid, and a per-scale-block
        selection (NVIF4) must fail that same check -- otherwise the check proves nothing.
    """
    torch.manual_seed(0)
    x = torch.randn(512, 512)
    x[:, ::11] *= 15.0   # outliers, so the two data types genuinely disagree
    x = x.to(torch.bfloat16)

    for tb in TYPE_BLOCKS:
        block_m, block_k = parse_type_block(tb)
        y = quant_mixfp4(x, groupsize=16, type_block=tb)
        conflicts, unknown, total = _type_block_grid_conflicts(y, block_m, block_k)
        assert conflicts == 0, f"{conflicts}/{total} type blocks of {tb} mix E2M1 and E0M3"
        print(f"ok  uniform data type across all {total} type blocks of {tb} "
              f"({unknown} fully degenerate)")

    # negative control: NVIF4 picks a type per 16-element scale block, so 16x16 type blocks
    # of its output must be mixed
    conflicts, _, total = _type_block_grid_conflicts(quant_nvif4(x, groupsize=16), 16, 16)
    assert conflicts > 0, "the uniformity check cannot detect per-scale-block type selection"
    print(f"ok  negative control: nvif4 mixes data types in {conflicts}/{total} 16x16 blocks")


def test_finer_type_block_is_never_worse():
    """
        1x16 divides every other configuration, and the type is chosen by minimizing squared error,
        so the total error of 1x16 must be <= the total error of any coarser type block.
    """
    torch.manual_seed(0)
    x = torch.randn(1024, 512)
    x[:, ::37] *= 25.0  # outliers, so the two data types actually disagree
    x = x.to(torch.bfloat16)

    def nmse(y):
        return ((x.float() - y.float()).pow(2).sum() / x.float().pow(2).sum()).item()

    base = nmse(quant_mixfp4(x, groupsize=16, type_block="1x16"))
    for tb in TYPE_BLOCKS[1:]:
        err = nmse(quant_mixfp4(x, groupsize=16, type_block=tb))
        assert base <= err * (1 + 1e-6), f"1x16 ({base:.4e}) worse than {tb} ({err:.4e})"
        print(f"ok  nmse 1x16 {base:.4e} <= {tb} {err:.4e}")


def test_type_selection_actually_switches():
    """
        A tensor whose left half is uniform (favours E0M3) and right half is heavy tailed
        (favours E2M1) must produce both data types.
    """
    torch.manual_seed(0)
    num_row, num_col = 256, 256
    x = torch.empty(num_row, num_col)
    x[:, : num_col // 2] = torch.rand(num_row, num_col // 2) * 2 - 1                # uniform
    x[:, num_col // 2 :] = torch.randn(num_row, num_col // 2).sign() * \
                           torch.rand(num_row, num_col // 2).pow(6) * 8            # heavy tailed
    x = x.to(torch.bfloat16)

    y        = quant_mixfp4(x, groupsize=16, type_block="16x16")
    tiled, _ = _tile_type_blocks(y.to(torch.float32), 16, 16, 16)

    # each scale block carries its own block scale, so the grid fit is tested per scale block
    n_e2m1 = n_e0m3 = 0
    for i in range(tiled.shape[0]):
        only_e2m1 = only_e0m3 = 0
        for j in range(tiled.shape[1]):
            fits_e2m1 = _fits_grid(tiled[i, j], E2M1_GRID)
            fits_e0m3 = _fits_grid(tiled[i, j], E0M3_GRID)
            only_e2m1 += int(fits_e2m1 and not fits_e0m3)
            only_e0m3 += int(fits_e0m3 and not fits_e2m1)
        n_e2m1 += int(only_e2m1 > 0)
        n_e0m3 += int(only_e0m3 > 0)
    assert n_e2m1 > 0 and n_e0m3 > 0, f"expected both data types, got E2M1={n_e2m1} E0M3={n_e0m3}"
    print(f"ok  both data types selected (E2M1={n_e2m1}, E0M3={n_e0m3})")


def test_padded_rows_do_not_leak():
    """Zero padding of the outer dimension must not change the result of the real rows."""
    torch.manual_seed(0)
    x = torch.randn(300, 256).to(torch.bfloat16)   # 300 is not a multiple of 32
    y = quant_mixfp4(x, groupsize=16, type_block="32x128")
    # the first 288 rows form 9 whole type-block rows and must match the unpadded quantization
    y_head = quant_mixfp4(x[:288], groupsize=16, type_block="32x128")
    assert torch.equal(y[:288], y_head), "padding leaked into the unpadded type blocks"
    print("ok  zero padding of the outer dimension does not leak")


########################### mix_4_6 ###########################

def _nmse(x, y):
    x, y = x.float(), y.float()
    return ((x - y).pow(2).sum() / x.pow(2).sum()).item()


def _outlier_tensor(rows=1024, cols=512, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(rows, cols)
    x[:, ::13] *= 18.0
    return x.to(torch.bfloat16)


def test_mix_4_6_shapes():
    for shape in [(128, 256), (2, 64, 512), (2, 8, 37, 128)]:
        x = torch.randn(*shape).to(torch.bfloat16)
        for tb in TYPE_BLOCKS:
            y = quant_mix_4_6(x, groupsize=16, type_block=tb)
            assert y.shape == x.shape and y.dtype == torch.bfloat16, (shape, tb)
    print("ok  mix_4_6 preserves shape / dtype for 2D, 3D and 4D inputs")


def test_mix_4_6_never_worse_than_mixfp4():
    """
        mix_4_6 chooses from a superset of mixfp4's options (it adds the 4-normalization of the
        same E2M1 grid), so its squared error can never be larger at the same type block.
    """
    x = _outlier_tensor()
    for tb in TYPE_BLOCKS:
        a = _nmse(x, quant_mix_4_6(x, groupsize=16, type_block=tb))
        b = _nmse(x, quant_mixfp4(x, groupsize=16, type_block=tb))
        assert a <= b * (1 + 1e-6), f"mix_4_6 {a:.4e} worse than mixfp4 {b:.4e} at {tb}"
        print(f"ok  mix_4_6 {a:.4e} <= mixfp4 {b:.4e}  ({tb})")


def test_mix_4_6_1x16_never_worse_than_either_parent():
    """At 1x16 the option set is a strict superset of both nvif4's and nvfp4_4over6's."""
    x = _outlier_tensor()
    mix = _nmse(x, quant_mix_4_6(x, groupsize=16, type_block="1x16"))
    for name, ref in [("nvif4", quant_nvif4(x, groupsize=16)),
                      ("nvfp4_4over6", quant_nvfp4_4over6(x, groupsize=16))]:
        r = _nmse(x, ref)
        assert mix <= r * (1 + 1e-6), f"mix_4_6 {mix:.4e} worse than {name} {r:.4e}"
        print(f"ok  mix_4_6 1x16 {mix:.4e} <= {name} {r:.4e}")


def _classify_scale_block(block, atol=0.06):
    """Which (grid, normalization) a quantized scale block is consistent with."""
    vals = block.abs().flatten()
    peak = vals.max()
    if peak == 0:
        return set()
    fits = set()
    for label, grid, qmax in [("e2m1@6", E2M1_GRID, 6.0),
                              ("e2m1@4", E2M1_GRID, 4.0),
                              ("e0m3", E0M3_GRID, 7.0)]:
        rescaled = vals * (qmax / peak)
        if (rescaled.unsqueeze(-1) - grid).abs().min(dim=-1).values.max().item() <= atol:
            fits.add(label)
    return fits


def test_mix_4_6_uses_both_normalizations():
    """
        If only the 6-normalization were ever chosen, mix_4_6 would just be mixfp4. Blocks
        normalized to 4 put their maximum on the E2M1 grid point 4, blocks normalized to 6 put it
        on 6, so the two are distinguishable from the dequantized output.
    """
    x = _outlier_tensor()
    y = quant_mix_4_6(x, groupsize=16, type_block="1x16")
    tiled, _ = _tile_type_blocks(y.to(torch.float32), 1, 16, 16)

    only6 = only4 = e0m3 = 0
    for i in range(0, tiled.shape[0], 7):          # subsample, the tensor has 32k blocks
        fits = _classify_scale_block(tiled[i, 0])
        only6 += int(fits == {"e2m1@6"})
        only4 += int(fits == {"e2m1@4"})
        e0m3  += int(fits == {"e0m3"})
    assert only4 > 0, "the 4-normalization is never selected -- mix_4_6 degenerates to mixfp4"
    assert only6 > 0, "the 6-normalization is never selected"
    print(f"ok  mix_4_6 uses both normalizations (only-6={only6}, only-4={only4}, e0m3={e0m3})")


def test_mix_4_6_type_uniform_per_type_block():
    """
        The 4/6 choice is metadata-free and may vary per scale block, but the E2M1-vs-E0M3 choice
        must still be uniform across a type block, because that is what the MMA declares.
    """
    x = _outlier_tensor(rows=512)
    for tb in TYPE_BLOCKS:
        block_m, block_k = parse_type_block(tb)
        y = quant_mix_4_6(x, groupsize=16, type_block=tb)
        tiled, _ = _tile_type_blocks(y.to(torch.float32), block_m, block_k, 16)

        conflicts = 0
        for i in range(tiled.shape[0]):
            any_e2m1 = any_e0m3 = 0
            for j in range(tiled.shape[1]):
                fits = _classify_scale_block(tiled[i, j])
                if not fits:
                    continue
                any_e2m1 += int(fits <= {"e2m1@6", "e2m1@4"} and len(fits) > 0)
                any_e0m3 += int(fits == {"e0m3"})
            conflicts += int(bool(any_e2m1) and bool(any_e0m3))
        assert conflicts == 0, f"{conflicts}/{tiled.shape[0]} type blocks of {tb} mix E2M1 and E0M3"
        print(f"ok  mix_4_6 keeps one element type per type block ({tb}, {tiled.shape[0]} blocks)")


def test_mix_4_6_rejects_bad_group_size():
    try:
        quant_mix_4_6(torch.randn(64, 128), groupsize=32, type_block="16x16")
    except AssertionError:
        print("ok  mix_4_6 rejects a non-16 scale-block size")
        return
    raise AssertionError("quant_mix_4_6 should require a scale-block size of 16")


def test_selection_loss_edge_cases():
    from quantize.quantizer import _selection_loss
    zero = torch.zeros(2, 3, 16)
    for m in ("mse", "sqnr", "cossim"):
        loss = _selection_loss(zero, zero, m)
        assert torch.isfinite(loss).all(), f"{m} produced non-finite loss on an all-zero block"
    x = torch.randn(2, 3, 16)
    for m in ("mse", "sqnr", "cossim"):
        exact = _selection_loss(x, x, m)              # perfectly represented block
        assert torch.isfinite(exact).all(), f"{m} non-finite when error is exactly 0"
    print("ok  selection loss is finite for all-zero and zero-error blocks")


def test_sqnr_equals_mse_at_1x16():
    """
        Within one scale block the signal energy is identical for every candidate, so ranking by
        SQNR is the same as ranking by MSE. At a 1x16 type block BOTH decisions are per scale
        block, so the two metrics are equivalent in exact arithmetic.

        They are NOT bit-identical, though: the log transform loses precision, so candidates whose
        errors differ by less than float32 epsilon can tie in the log domain and break the other
        way. Measured here that is 1 scale block in 32768 (relative error gap 7e-8, below the
        1.2e-7 float32 epsilon). So assert agreement to well within that, not equality.
    """
    x = _outlier_tensor()
    a = quant_mix_4_6(x, groupsize=16, type_block="1x16", metric="mse")
    b = quant_mix_4_6(x, groupsize=16, type_block="1x16", metric="sqnr")

    disagree = (a.float() != b.float()).float().mean().item()
    assert disagree < 1e-4, f"sqnr and mse differ on {disagree:.2%} of elements at 1x16, expected ~0"

    nmse_a, nmse_b = _nmse(x, a), _nmse(x, b)
    assert abs(nmse_a - nmse_b) / nmse_a < 1e-6, (nmse_a, nmse_b)
    print(f"ok  sqnr == mse at 1x16 up to float32 ties "
          f"({disagree*100:.4f}% of elements, nmse {nmse_a:.6e} vs {nmse_b:.6e})")


def test_sqnr_differs_from_mse_at_coarse_type_blocks():
    """
        Once a type block spans several scale blocks the aggregation differs: MSE sums raw squared
        error (high-energy blocks dominate), SQNR sums per-block dB (every block weighs the same).
    """
    x = _outlier_tensor()
    differed = []
    for tb in ["16x16", "8x64", "32x128"]:
        a = quant_mix_4_6(x, groupsize=16, type_block=tb, metric="mse")
        b = quant_mix_4_6(x, groupsize=16, type_block=tb, metric="sqnr")
        if not torch.equal(a, b):
            differed.append(tb)
    assert differed, "sqnr never changed the type-block decision -- aggregation is not normalizing"
    print(f"ok  sqnr changes the type-block choice at {differed}")


def test_cossim_differs_and_stays_valid():
    """cosine similarity is scale invariant, so it is not a monotone function of the error."""
    x = _outlier_tensor()
    differed = []
    for tb in ["1x16", "8x64", "32x128"]:
        a = quant_mix_4_6(x, groupsize=16, type_block=tb, metric="mse")
        c = quant_mix_4_6(x, groupsize=16, type_block=tb, metric="cossim")
        assert c.shape == x.shape and c.dtype == torch.bfloat16
        assert torch.isfinite(c.float()).all(), f"cossim produced non-finite output at {tb}"
        if not torch.equal(a, c):
            differed.append(tb)
    assert differed, "cossim never changed a decision"
    print(f"ok  cossim changes decisions at {differed}, output finite everywhere")


def test_mse_stays_the_best_by_mse():
    """
        Sanity check on the metrics: selecting by MSE must give the lowest MSE. If SQNR or cossim
        ever won on this measure, the MSE path would be buggy.
    """
    x = _outlier_tensor()
    for tb in ["1x16", "8x64"]:
        scores = {m: _nmse(x, quant_mix_4_6(x, groupsize=16, type_block=tb, metric=m))
                  for m in ("mse", "sqnr", "cossim")}
        assert scores["mse"] <= min(scores.values()) * (1 + 1e-9), (tb, scores)
        print(f"ok  {tb}: nmse by metric " +
              ", ".join(f"{m}={v:.4e}" for m, v in scores.items()))


def test_rejects_unknown_metric():
    try:
        quant_mix_4_6(torch.randn(64, 128), groupsize=16, type_block="16x16", metric="l1")
    except (AssertionError, ValueError):
        print("ok  unknown selection metric is rejected")
        return
    raise AssertionError("quant_mix_4_6 should reject an unknown metric")


def test_rejects_bad_group_size():
    x = torch.randn(64, 128)
    try:
        quant_mixfp4(x, groupsize=32, type_block="16x16")
    except AssertionError:
        print("ok  non-16 scale-block size is rejected")
        return
    raise AssertionError("quant_mixfp4 should require a scale-block size of 16")


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
    print(f"\nAll {len(tests)} MixFP4 CPU tests passed.")

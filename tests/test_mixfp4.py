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
    quant_nvfp4_nover6,
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
    # "l1"/"l0.5" ARE valid (the generalized power loss), so the negative case has to be a name
    # that parses as neither a named metric nor "l<p>".
    for bad in ("kldiv", "lx", "l"):
        try:
            quant_mix_4_6(torch.randn(64, 128), groupsize=16, type_block="16x16", metric=bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError(f'quant_mix_4_6 should reject the unknown metric "{bad}"')
    print("ok  unknown selection metric is rejected")


def test_mae_is_l1_and_differs_from_mse():
    """
        MAE must be a real alternative, not a relabelling of MSE: it has to change at least some
        decisions on a tensor with outliers, and "mae" and "l1" must be the same thing.
    """
    x = _outlier_tensor()
    for tb in ["1x16", "8x64"]:
        mae = quant_mix_4_6(x, groupsize=16, type_block=tb, metric="mae", clip="bothx")
        l1  = quant_mix_4_6(x, groupsize=16, type_block=tb, metric="l1",  clip="bothx")
        mse = quant_mix_4_6(x, groupsize=16, type_block=tb, metric="mse", clip="bothx")
        assert torch.equal(mae, l1), f'"mae" and "l1" disagree at {tb}'
        assert not torch.equal(mae, mse), f'"mae" reproduces "mse" exactly at {tb}'
        # and MAE must win on its own criterion
        e_mae = (mae.float() - x.float()).abs().mean().item()
        e_mse = (mse.float() - x.float()).abs().mean().item()
        assert e_mae <= e_mse * (1 + 1e-6), (tb, e_mae, e_mse)
        print(f"ok  {tb}: mae == l1, and mean|err| mae={e_mae:.4e} <= mse={e_mse:.4e}")


def test_corr_metric_sees_coherent_error():
    """
        The whole reason "corr<r>" exists: MSE cannot tell an error that accumulates in the dot
        product from one that cancels. Two blocks with identical squared error, one all-positive and
        one alternating, must score identically under mse and differently under corr.
    """
    from quantize.quantizer import _selection_loss
    x  = torch.zeros(2, 1, 16)
    dq = torch.empty(2, 1, 16)
    dq[0] = 0.1                                     # coherent: every element pulled the same way
    dq[1] = 0.1 * torch.tensor([1.0, -1.0]).repeat(8)   # cancelling

    mse = _selection_loss(x, dq, "mse")
    assert torch.allclose(mse[0], mse[1]), "mse should be blind to the sign pattern"

    c = _selection_loss(x, dq, "corr0.2")
    assert c[0].item() > c[1].item() * 5, (c[0].item(), c[1].item())
    # r=0 must reproduce mse exactly
    assert torch.allclose(_selection_loss(x, dq, "corr0"), mse)
    print(f"ok  corr0.2 separates coherent {c[0].item():.4f} from cancelling {c[1].item():.4f}; "
          "mse cannot")


def test_corr_changes_decisions_and_rejects_bad_r():
    x = _outlier_tensor()
    for tb in ["1x16", "8x64"]:
        a = quant_mix_4_6(x, groupsize=16, type_block=tb, metric="mse",     clip="bothx")
        b = quant_mix_4_6(x, groupsize=16, type_block=tb, metric="corr0.2", clip="bothx")
        assert not torch.equal(a, b), f"corr0.2 reproduced mse exactly at {tb}"
    for bad in ("corr1", "corr1.5"):
        try:
            quant_mix_4_6(torch.randn(64, 128), groupsize=16, type_block="16x16", metric=bad)
        except AssertionError:
            continue
        raise AssertionError(f'quant_mix_4_6 should reject "{bad}" (r must be < 1)')
    print("ok  corr changes decisions, and r outside [0, 1) is rejected")


def test_nover6_matches_the_measured_preset():
    """
        `quant_nvfp4_nover6` is the deployable extract of this study: the widened FourOverSix search
        with the type-block machinery removed. It must be bit-identical to the general path with the
        preset its default alphas come from and the E0M3 branch switched off -- otherwise the
        standalone function is a different format from the one the perplexity numbers were measured
        on. The preset moved from `a1` to `dense9` when the dense grid won, and this test is what
        catches the default and the preset drifting apart.
    """
    from quantize.quantizer import CLIP_PRESETS, NOVER6_ALPHAS
    assert tuple(CLIP_PRESETS["dense9"]["e2m1"]) == tuple(NOVER6_ALPHAS), \
        "nover6's default alphas no longer match the dense9 preset"
    for x in (_outlier_tensor(rows=256), torch.randn(128, 512).to(torch.bfloat16)):
        a = quant_nvfp4_nover6(x, groupsize=16)
        b = quant_mix_4_6(x, groupsize=16, type_block="1x16", clip="dense9", elect="never")
        assert torch.equal(a, b), "nover6 diverges from mix_4_6(clip=dense9, elect=never)"
    print("ok  nvfp4_nover6 == mix_4_6(clip=dense9, elect=never)")


def test_nover6_beats_4over6_on_mse():
    """
        `nover6` searches a superset of 4over6's scale candidates ({1, 1.5} is inside
        {1, 1.25, 1.5, 2, 3}) under the same per-block squared error, so it cannot score worse.
    """
    for x in (_outlier_tensor(), torch.randn(512, 512).to(torch.bfloat16)):
        n = _nmse(x, quant_nvfp4_nover6(x, groupsize=16))
        f = _nmse(x, quant_mix_4_6(x, groupsize=16, type_block="1x16", elect="never"))
        assert n <= f * (1 + 1e-6), f"nover6 {n:.4e} worse than 4over6 {f:.4e}"
        print(f"ok  nover6 {n:.4e} <= 4over6 {f:.4e}")


########################### clipping ###########################

def test_clip_presets_reduce_mse():
    """
        Every clip preset is a SUPERSET of `base`'s scale candidates on both grids, and the choice
        is per scale block by the same loss, so a richer preset can never score worse under the
        metric it selects with. This is the analogue of the mixfp4 <= 4over6 monotonicity check.
    """
    from quantize.quantizer import CLIP_PRESETS
    for name in ("e0", "e0x", "e2", "e2x", "both", "bothx", "wide"):
        base_a = CLIP_PRESETS["base"]
        this_a = CLIP_PRESETS[name]
        if not (set(base_a["e2m1"]) <= set(this_a["e2m1"]) and
                set(base_a["e0m3"]) <= set(this_a["e0m3"])):
            continue     # not a superset -- monotonicity does not apply, skip
        x = _outlier_tensor()
        for tb in ["1x16", "8x64", "32x128"]:
            b = _nmse(x, quant_mix_4_6(x, groupsize=16, type_block=tb, clip="base"))
            c = _nmse(x, quant_mix_4_6(x, groupsize=16, type_block=tb, clip=name))
            assert c <= b * (1 + 1e-6), f"clip={name} {c:.4e} worse than base {b:.4e} at {tb}"
        print(f"ok  clip={name} never worse than base (superset of scale candidates)")


def test_clipping_actually_clips():
    """
        A clip ratio below 1 must genuinely saturate values above alpha*block_max -- otherwise the
        preset is a no-op and any measured difference comes from somewhere else.
    """
    from quantize.quantizer import _quant_e0m3
    # one scale block whose maximum is a lone outlier the rest of the block pays for:
    # at alpha=1 the step is 10/7, so every 0.4 rounds to zero and the bulk is entirely lost.
    x = torch.full((1, 16), 0.4)
    x[0, 0] = 10.0
    bmax = x.abs().amax()
    dq_exact = _quant_e0m3(x, bmax / 7.0)
    dq_clip  = _quant_e0m3(x, 0.2 * bmax / 7.0)
    assert dq_exact[0, 0].item() == 10.0, dq_exact[0, 0]
    assert abs(dq_clip[0, 0].item() - 2.0) < 1e-5, dq_clip[0, 0]
    assert (dq_exact[0, 1:] == 0).all(), "alpha=1 should round the whole bulk to zero here"
    # ... and the clipped version represents the bulk strictly better
    assert (dq_clip[0, 1:] - x[0, 1:]).abs().sum() < (dq_exact[0, 1:] - x[0, 1:]).abs().sum()
    print("ok  alpha < 1 saturates the outlier and refines the bulk")


def test_clip_min_gain_gates_only_the_clipping_candidates():
    """
        `clipmin<t>` must gate alpha < 1 only. A huge threshold has to fall back exactly to the
        preset with the clipping alphas removed -- not to `base`, and not to something in between,
        or the knob would be silently changing the non-clipping search too.
    """
    from quantize.quantizer import CLIP_PRESETS
    x = _outlier_tensor()
    for preset in ("e0x", "bothx", "wide"):
        gated = quant_mix_4_6(x, groupsize=16, type_block="8x64",
                              clip=preset, clip_min_gain=1e9)
        # the same preset with every alpha < 1 dropped, done by hand
        kept = {g: tuple(a for a in v if a >= 1.0) or (1.0,)
                for g, v in CLIP_PRESETS[preset].items()}
        saved = CLIP_PRESETS["__tmp"] = kept
        manual = quant_mix_4_6(x, groupsize=16, type_block="8x64", clip="__tmp")
        del CLIP_PRESETS["__tmp"]
        assert torch.equal(gated, manual), f"clipmin(inf) on {preset} is not the no-clip search"

        # and a zero threshold must reproduce the ungated preset exactly
        open_ = quant_mix_4_6(x, groupsize=16, type_block="8x64", clip=preset, clip_min_gain=0.0)
        plain = quant_mix_4_6(x, groupsize=16, type_block="8x64", clip=preset)
        assert torch.equal(open_, plain), f"clipmin(0) on {preset} changed the result"
        # gating must be observable -- but only where the gated candidates can reach the output.
        # In "e0x" only E0M3 clips, and if no tile elects E0M3 on this tensor the gate is invisible,
        # which is correct behaviour rather than a dead knob.
        if any(a < 1.0 for a in CLIP_PRESETS[preset]["e2m1"]):
            assert not torch.equal(gated, plain), f"clipmin(inf) on {preset} changed nothing"
    print("ok  clipmin gates alpha < 1 only; t=0 is the plain preset, t=inf drops clipping")


def test_alpha_min_gain_gates_the_scale_search():
    """
        `amin<t>` applies the decisive-margin principle to the scale search itself. A huge threshold
        must collapse the search to alpha = 1 exactly -- plain NVFP4 -- and t = 0 must leave the
        ordinary argmin search untouched.
    """
    x = _outlier_tensor()
    # These must be MULTI-alpha presets -- `alpha_min_gain` gates a search, so a single-candidate
    # preset like the reported `a1` cannot exercise it and would make the test vacuous.
    for preset in ("base", "dense5", "dense9"):
        gated = quant_mix_4_6(x, groupsize=16, type_block="1x16",
                              clip=preset, elect="never", alpha_min_gain=1e9)
        only1 = quant_mix_4_6(x, groupsize=16, type_block="1x16",
                              clip="base", elect="never", alpha_min_gain=1e9)
        assert torch.equal(gated, only1), f"amin(inf) on {preset} did not collapse to alpha=1"

        open_ = quant_mix_4_6(x, groupsize=16, type_block="1x16",
                              clip=preset, elect="never", alpha_min_gain=0.0)
        plain = quant_mix_4_6(x, groupsize=16, type_block="1x16", clip=preset, elect="never")
        assert torch.equal(open_, plain), f"amin(0) on {preset} changed the result"
        if preset != "base":
            assert not torch.equal(gated, plain), f"amin(inf) on {preset} changed nothing"
    print("ok  amin(inf) collapses the scale search to alpha=1; amin(0) is the plain search")


def test_rejects_unknown_clip_preset():
    try:
        quant_mix_4_6(torch.randn(64, 128), groupsize=16, type_block="16x16", clip="nope")
    except (AssertionError, ValueError):
        print("ok  unknown clip preset is rejected")
        return
    raise AssertionError("quant_mix_4_6 should reject an unknown clip preset")


def test_rejects_bad_group_size():
    x = torch.randn(64, 128)
    try:
        quant_mixfp4(x, groupsize=32, type_block="16x16")
    except AssertionError:
        print("ok  non-16 scale-block size is rejected")
        return
    raise AssertionError("quant_mixfp4 should require a scale-block size of 16")




########################### election rules ###########################

def test_dominance_never_harms_a_block():
    """
        The "dominance" rule elects E0M3 only when it is at least as good on EVERY scale block of
        the tile, so no block can come out worse than under the always-E2M1 fallback (4over6).
        That restores at coarse type blocks the pointwise property 1x16 has for free.
    """
    from quantize.quantizer import _tile_type_blocks, _quant_e2m1
    x = _outlier_tensor()
    for tb in ["16x16", "8x64", "32x128"]:
        dom = quant_mix_4_6(x, groupsize=16, type_block=tb, elect="dominance")
        # per-scale-block squared error must never exceed the all-E2M1 baseline
        bm, bk = parse_type_block(tb)
        err_dom = _tile_type_blocks((dom.float() - x.float()).pow(2), bm, bk, 16)[0].sum(-1)
        # all-E2M1 reference: force no election by demanding an impossible margin
        ref = quant_mix_4_6(x, groupsize=16, type_block=tb, elect="margin", margin=1e9)
        err_ref = _tile_type_blocks((ref.float() - x.float()).pow(2), bm, bk, 16)[0].sum(-1)
        worse = (err_dom > err_ref * (1 + 1e-6)).float().mean().item()
        assert worse == 0.0, f"{tb}: dominance made {worse:.2%} of scale blocks worse than all-E2M1"
        print(f"ok  dominance harms 0 blocks at {tb}")


def test_margin_monotonically_shrinks_election():
    """A larger margin must be strictly more conservative: it can only elect a subset of tiles."""
    from quantize.quantizer import _elect_e0m3
    torch.manual_seed(0)
    gain = torch.randn(500, 16, 1)
    prev = None
    for m in [0.0, 0.5, 1.0, 2.0, 4.0]:
        elected = _elect_e0m3(gain, rule="margin", margin=m).squeeze()
        if prev is not None:
            assert (elected & ~prev).sum() == 0, f"margin {m} elected a tile that margin<{m} did not"
        prev = elected
    dom = _elect_e0m3(gain, rule="dominance").squeeze()
    argmin = _elect_e0m3(gain, rule="argmin").squeeze()
    assert (dom & ~argmin).sum() == 0, "dominance elected a tile argmin did not"
    print("ok  election rules are nested: dominance <= margin(large) <= ... <= argmin")


def test_importance_weighting_changes_selection():
    """A non-uniform importance vector must be able to change which type a tile elects."""
    x = _outlier_tensor(rows=256, cols=512)
    uniform = quant_mix_4_6(x, groupsize=16, type_block="8x64")
    torch.manual_seed(1)
    imp = torch.rand(512) ** 4 * 100        # strongly anisotropic, like LLM activations
    weighted = quant_mix_4_6(x, groupsize=16, type_block="8x64", importance=imp)
    assert not torch.equal(uniform, weighted), "importance had no effect on the result"
    assert weighted.shape == x.shape and torch.isfinite(weighted.float()).all()
    print("ok  importance weighting changes the selection and stays finite")


def test_importance_uniform_is_a_noop():
    """A constant importance vector rescales every loss equally and must not change any decision."""
    x = _outlier_tensor(rows=256, cols=512)
    base = quant_mix_4_6(x, groupsize=16, type_block="8x64")
    for c in (1.0, 7.5):
        same = quant_mix_4_6(x, groupsize=16, type_block="8x64",
                             importance=torch.full((512,), c))
        assert torch.equal(base, same), f"constant importance {c} changed the result"
    print("ok  constant importance is a no-op")


def test_never_rule_suppresses_e0m3_and_margin_does_not():
    """
        A large margin does NOT suppress E0M3 at a 1x16 type block: the tile holds one scale block,
        so std(gain) is 0 and the test degenerates to argmin. Only elect="never" is a valid
        "E2M1 only" control. This caught a broken experiment, so it is pinned.
    """
    from quantize.quantizer import _elect_e0m3
    torch.manual_seed(0)
    gain = torch.randn(200, 1, 1)                      # 1x16: one scale block per tile
    assert _elect_e0m3(gain, rule="never").sum() == 0
    assert _elect_e0m3(gain, rule="margin", margin=999).sum() > 0, \
        "margin unexpectedly suppressed E0M3 at 1x16; the control would be silently valid"
    # and with several blocks per tile a huge margin does suppress it
    gain_multi = torch.randn(200, 16, 1)
    assert _elect_e0m3(gain_multi, rule="margin", margin=999).sum() == 0
    print("ok  elect='never' suppresses E0M3; a large margin only does so for multi-block tiles")


def test_dominance_degenerates_to_e2m1_at_realizable_blocks():
    """
        At any type block spanning more than a couple of scale blocks, requiring EVERY block to
        prefer E0M3 is so strict that it never happens on real data: dominance becomes bit-identical
        to elect="never", i.e. to plain 4over6. Measured on Llama-2-7B weights AND activations at
        8x64/16x64/32x64/32x128: zero elements differ.

        So a "dominance" row in a results table is a 4over6 row, and any delta it shows against
        nvfp4_4over6 is the E2M1 rounding-tie convention, not a benefit of the E0M3 type block.
        Only at 1x16 (one scale block per tile) does dominance coincide with argmin instead.
    """
    x = _outlier_tensor(rows=512)
    for tb in ["8x64", "32x64", "32x128"]:
        dom = quant_mix_4_6(x, groupsize=16, type_block=tb, elect="dominance")
        nev = quant_mix_4_6(x, groupsize=16, type_block=tb, elect="never")
        assert torch.equal(dom, nev), f"dominance elected E0M3 somewhere at {tb} -- rerun the analysis"
    dom1 = quant_mix_4_6(x, groupsize=16, type_block="1x16", elect="dominance")
    nev1 = quant_mix_4_6(x, groupsize=16, type_block="1x16", elect="never")
    assert not torch.equal(dom1, nev1), "dominance should still elect E0M3 at 1x16"
    print("ok  dominance == plain E2M1 (4over6) at realizable blocks, but not at 1x16")


def test_election_rules_are_nested():
    """
        The rules form a chain of increasing caution. On the SAME per-block gains, each of them must
        elect a subset of what `argmin` elects -- none of them may elect a tile whose total gain is
        negative, because then the tile is worse than 4over6 under its own criterion.

        `harm` is the one rule where this needs stating carefully: harm(1) IS argmin (won > lost is
        the same as total > 0), and larger lambda only shrinks the set.
    """
    from quantize.quantizer import _elect_e0m3
    torch.manual_seed(0)
    gain = torch.randn(500, 32, 1)
    ref  = gain.abs() + 0.5                     # a plausible positive E2M1 loss per block
    base = _elect_e0m3(gain, rule="argmin")

    cases = [
        ("dominance", 0.0), ("margin", 2.0), ("relmargin", 2.0),
        ("tol", 0.5), ("harm", 2.0), ("vote", 0.5),
    ]
    for rule, m in cases:
        e = _elect_e0m3(gain, rule=rule, margin=m, ref=ref)
        assert bool((e & ~base).any()) is False, \
            f'rule "{rule}" elected a tile that argmin rejects (total gain <= 0)'
        print(f"ok  {rule}({m}) elects {int(e.sum())}/{int(base.sum())} of argmin's tiles")

    # harm(1) is argmin exactly
    assert torch.equal(_elect_e0m3(gain, rule="harm", margin=1.0), base), \
        "harm(lambda=1) must reproduce argmin"
    print("ok  harm(1) == argmin")


########################### Hadamard rotation ###########################

def test_hadamard_is_orthogonal_and_chunked_rotation_inverts():
    from quantize.quantizer import _hadamard, _rotate_chunks
    for n in (2, 4, 16, 64, 128):
        h = _hadamard(n, torch.float64, "cpu")
        assert torch.allclose(h @ h.t(), torch.eye(n, dtype=torch.float64), atol=1e-10), n
    x = torch.randn(37, 256, dtype=torch.float64)
    for n in (16, 64, 128):
        back = _rotate_chunks(_rotate_chunks(x, n), n, transpose=True)
        assert torch.allclose(back, x, atol=1e-10), f"chunked rotation of size {n} is not invertible"
    print("ok  Hadamard is orthogonal and the chunked rotation inverts exactly")


def test_rotation_preserves_the_gemm():
    """
        The identity the whole scheme rests on: rotating a chunk of the reduction dimension in BOTH
        operands leaves Y = X W^T unchanged. If this failed, a rotated layer would quietly compute
        something else and the perplexity numbers would be meaningless.
    """
    from quantize.quantizer import _rotate_chunks
    torch.manual_seed(0)
    X = torch.randn(11, 256, dtype=torch.float64)     # tokens x K
    W = torch.randn(7, 256, dtype=torch.float64)      # out    x K
    for n in (16, 64):
        Y  = X @ W.t()
        Yr = _rotate_chunks(X, n) @ _rotate_chunks(W, n).t()
        assert torch.allclose(Y, Yr, atol=1e-9), f"rotation of size {n} changed the GEMM"
    print("ok  rotating both operands leaves the GEMM identical")


def test_rotation_changes_and_shrinks_block_max():
    """
        Rotation earns its keep by spreading a block's outlier over the whole block, which is what
        drops the block maximum the 4-bit grid has to span. Check the mechanism, not just that the
        output moved.
    """
    from quantize.quantizer import _rotate_chunks
    x = torch.zeros(4, 16)
    x[:, 0] = 10.0
    x[:, 1:] = 0.3
    peak_before = x.abs().amax(dim=-1)
    peak_after  = _rotate_chunks(x, 16).abs().amax(dim=-1)
    assert (peak_after < peak_before * 0.5).all(), (peak_before, peak_after)
    print(f"ok  rotation cuts an outlier block's peak from {peak_before[0]:.2f} "
          f"to {peak_after[0]:.2f}")


def test_rotate_modes_behave():
    x = _outlier_tensor()
    base = quant_mix_4_6(x, groupsize=16, type_block="8x64")
    rot  = quant_mix_4_6(x, groupsize=16, type_block="8x64", rotate="all")
    col  = quant_mix_4_6(x, groupsize=16, type_block="8x64", rotate="col")
    assert not torch.equal(base, rot), 'rotate="all" changed nothing'
    assert x.shape == rot.shape == col.shape

    # "col" picks the better basis per column chunk, so by construction it cannot be worse than
    # EITHER of the two fixed choices under the squared error it selects on
    e = lambda y: (y.float() - x.float()).pow(2).sum().item()
    assert e(col) <= min(e(base), e(rot)) * (1 + 1e-6), (e(base), e(rot), e(col))
    print(f"ok  rotate: none={e(base):.4e}, all={e(rot):.4e}, col={e(col):.4e} (col <= both)")


def test_rejects_unknown_rotate_mode():
    for bad in ("hadamard", "yes"):
        try:
            quant_mix_4_6(torch.randn(64, 128), groupsize=16, type_block="16x16", rotate=bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError(f'quant_mix_4_6 should reject rotate="{bad}"')
    print("ok  unknown rotate mode is rejected")


########################### row permutation ###########################

def test_permutation_is_undone_exactly():
    """
        The row sort must be invisible in the output: every row has to come back to where it started.
        If it did not, the layer would silently compute a permuted GEMM and perplexity would be
        garbage rather than merely worse -- so this checks against a per-row fingerprint, not just
        against the aggregate error.
    """
    x = _outlier_tensor(rows=520)                     # 520 is not a multiple of 8, 16 or 32
    for tb in ["8x64", "16x64", "32x128"]:
        y = quant_mix_4_6(x, groupsize=16, type_block=tb, permute="rows")
        assert y.shape == x.shape
        # each row of the output must be the quantization of the SAME row of the input: check that
        # the row is closer to its own source row than to any other row
        xf, yf = x.float(), y.float()
        for i in (0, 7, 8, 137, 519):
            d_self  = (yf[i] - xf[i]).pow(2).sum()
            d_other = (yf[i].unsqueeze(0) - xf).pow(2).sum(-1)
            best    = int(d_other.argmin())
            assert best == i, f"{tb}: output row {i} matches input row {best}, not itself"
            assert d_self <= xf[i].pow(2).sum() * 0.5, f"{tb}: row {i} is not a quantization of itself"
    print("ok  the row permutation is inverted exactly (rows land where they started)")


def test_permutation_is_a_noop_at_one_row_tiles():
    """A type block one row tall has nothing to group, so sorting must change nothing at all."""
    x = _outlier_tensor()
    a = quant_mix_4_6(x, groupsize=16, type_block="1x16", permute="none")
    b = quant_mix_4_6(x, groupsize=16, type_block="1x16", permute="rows")
    assert torch.equal(a, b), "row sorting changed the 1x16 result"
    print("ok  row sorting is a no-op at a 1x16 type block")


def test_permutation_makes_tiles_homogeneous():
    """
        The point of sorting is that tiles stop straddling the E0M3/E2M1 boundary. Measure it
        directly: the share of scale blocks whose individually-best grid differs from the one their
        tile elected must drop.
    """
    from quantize.quantizer import row_preference

    # `_outlier_tensor` gives every row the same outlier pattern, so every row has the same
    # preference and there is nothing to sort. Build a tensor whose rows genuinely disagree:
    # odd rows are heavy tailed (E2M1 territory), even rows are near-uniform (E0M3 territory),
    # interleaved so that every 8-row tile straddles the boundary before sorting.
    torch.manual_seed(0)
    rows, cols = 1024, 512
    x = torch.rand(rows, cols) * 2 - 1                 # flat: favours the uniform grid
    heavy = torch.randn(rows // 2, cols)
    heavy[:, ::13] *= 18.0                             # spiky: favours the log-spaced grid
    x[1::2] = heavy
    x  = x.to(torch.bfloat16)
    xf = x.float().reshape(-1, x.shape[-1])
    gs = (xf.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
    pref = row_preference(xf / gs, 16, "mse", "base")

    # how much a tile of 8 consecutive rows disagrees with itself, before and after sorting
    def straddle(p):
        tiles = p[: (len(p) // 8) * 8].reshape(-1, 8)
        return ((tiles > 0).any(dim=1) & (tiles <= 0).any(dim=1)).float().mean().item()

    before = straddle(pref)
    after  = straddle(pref[pref.argsort(descending=True)])
    assert after < before, f"sorting did not reduce straddling tiles ({before:.3f} -> {after:.3f})"
    print(f"ok  row sorting cuts straddling 8-row tiles from {before:.1%} to {after:.1%}")


def test_harm_is_the_robust_decision():
    """
        `harm(lambda)` claims to be the robust decision under an unknown per-block importance
        w_b in [1/kappa, kappa] with lambda = kappa^2: elect E0M3 iff sum_b w_b gain_b > 0 for EVERY
        admissible w. Check that against the explicit worst case, which puts the smallest weight on
        every gain and the largest on every harm.
    """
    from quantize.quantizer import _elect_e0m3
    torch.manual_seed(0)
    gain = torch.randn(400, 32, 1)
    for kappa in (1.0, 1.5, 2.0, 3.0):
        worst = (gain.clamp(min=0) / kappa - kappa * (-gain).clamp(min=0)).sum(dim=(-1, -2))
        want  = (worst > 0)[:, None, None]
        got   = _elect_e0m3(gain, rule="harm", margin=kappa ** 2)
        assert torch.equal(got, want), f"harm({kappa**2}) is not the robust rule for kappa={kappa}"
        print(f"ok  harm({kappa**2:.2f}) == robust decision for importance spread kappa={kappa}"
              f"  ({int(got.sum())}/{gain.shape[0]} tiles)")


def test_tol_interpolates_dominance_to_argmin():
    """
        "tol" is dominance with a slack: tol(0) must be dominance, and a huge tolerance must be
        argmin. If it did not bracket both, the knob would not be the interpolation it claims.
    """
    from quantize.quantizer import _elect_e0m3
    torch.manual_seed(0)
    gain = torch.randn(500, 16, 1)
    ref  = gain.abs() + 0.5
    assert torch.equal(_elect_e0m3(gain, rule="tol", margin=0.0, ref=ref),
                       _elect_e0m3(gain, rule="dominance"))
    assert torch.equal(_elect_e0m3(gain, rule="tol", margin=1e9, ref=ref),
                       _elect_e0m3(gain, rule="argmin"))
    print("ok  tol(0) == dominance and tol(inf) == argmin")


def test_vote_ignores_magnitude():
    """
        The point of "vote" is that one huge scale block cannot carry a tile. Construct a tile where
        a single block has an enormous gain and every other block is mildly harmed: argmin elects it,
        a majority vote must not.
    """
    from quantize.quantizer import _elect_e0m3
    gain = torch.full((1, 16, 1), -1.0)
    gain[0, 0, 0] = 1000.0
    assert _elect_e0m3(gain, rule="argmin").item() is True
    assert _elect_e0m3(gain, rule="vote", margin=0.5).item() is False
    print("ok  vote(0.5) rejects a tile carried by one high-energy block")


def test_dtype_name_parsing():
    """
        The data type name is the only channel the sweep has for these settings, so a typo must
        raise rather than silently fall back to the default.
    """
    from quantize.quantizer import parse_mix_4_6_dtype
    cases = {
        "mix_4_6":                ("mse",    "argmin",    0.0,  False, "base", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
        "mix_4_6_m2":             ("mse",    "margin",    2.0,  False, "base", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
        "mix_4_6_mae":            ("mae",    "argmin",    0.0,  False, "base", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
        "mix_4_6_l0.5":           ("l0.5",   "argmin",    0.0,  False, "base", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
        "mix_4_6_corr0.2_clipe0_h2": ("corr0.2", "harm", 2.0, False, "e0", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
        "mix_4_6_clipbothx":      ("mse",    "argmin",    0.0,  False, "bothx", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
        "mix_4_6_mae_clipwide_rm2": ("mae",  "relmargin", 2.0,  False, "wide", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
        "mix_4_6_tol0.25":        ("mse",    "tol",       0.25, False, "base", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
        "mix_4_6_h3":             ("mse",    "harm",      3.0,  False, "base", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
        "mix_4_6_v0.6":           ("mse",    "vote",      0.6,  False, "base", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
        "mix_4_6_hess_dom":       ("mse",    "dominance", 0.0,  True,  "base", 0.0, 0.0, "none", "none", 16, 0.0, 2.1),
    }
    # Compare only the fields each case enumerates. This function grows a new trailing field every
    # time a knob is added (peak_veto, imp_alpha/imp_elect, imp_gran ...), and a full-tuple compare
    # turns every such addition into a spurious failure here -- which is exactly what happened.
    # Fields beyond the prefix get their own explicit cases below.
    for name, want in cases.items():
        got = parse_mix_4_6_dtype(name)
        assert got[:len(want)] == want, f"{name}: got {got[:len(want)]}, want {want}"

    # the trailing knobs, checked by name so they cannot silently drift
    fields = ("metric", "elect", "margin", "use_importance", "clip", "clip_min_gain",
              "alpha_min_gain", "permute", "rotate", "rotate_size", "rotate_min_gain",
              "rotate_outlier_max", "peak_veto", "imp_alpha", "imp_elect", "imp_gran")
    def field(name, key):
        got = parse_mix_4_6_dtype(name)
        assert len(got) == len(fields), \
            f"parse_mix_4_6_dtype returns {len(got)} fields, this test knows {len(fields)}"
        return got[fields.index(key)]
    assert field("mix_4_6", "imp_gran") == 0
    assert field("mix_4_6_hess_impg16_h1.5", "imp_gran") == 16
    assert field("mix_4_6_hess_impg64_h1.5", "imp_gran") == 64
    assert field("mix_4_6_hess_impg16_h1.5", "elect") == "harm"
    assert field("mix_4_6_hess_impg16_h1.5", "use_importance") is True
    assert field("mix_4_6_hesst", "imp_alpha") is False
    assert field("mix_4_6_hessa", "imp_elect") is False
    for bad in ("mix_4_6_zzz", "mix_4_6_clipnope", "mix_4_6_m"):
        try:
            parse_mix_4_6_dtype(bad)
        except ValueError:
            continue
        raise AssertionError(f'parse_mix_4_6_dtype should reject "{bad}"')
    print(f"ok  {len(cases)} data type names parse, and unknown qualifiers are rejected")


REPORTED_CLIP = "a1"          # the ONLY preset a reported configuration may use


def test_rate_rule_is_model_independent():
    """
        `rate<f>` elects a FRACTION of tiles rather than everything past a threshold.

        The point of the rule is that `h<lambda>` is not comparable across models: at lambda = 1.5,
        56% of Qwen3-4B's tiles elect E0M3 against 31% of Llama-3.1-8B's, so the same lambda is a
        different intervention on each. Ranking makes the elected fraction the SAME by construction,
        whatever the gain distribution looks like.

        That invariance is what this test checks, on two tensors with deliberately different gain
        distributions: the elected share must track `f` on both, while `h<lambda>` must NOT.
    """
    from quantize.quantizer import _elect_e0m3

    torch.manual_seed(0)

    def elected_share(gain, rule, margin):
        return float(_elect_e0m3(gain, rule=rule, margin=margin).float().mean())

    # two distributions: one centred near zero, one strongly shifted towards E0M3
    balanced = torch.randn(4000, 32, 1)
    shifted  = torch.randn(4000, 32, 1) + 0.55

    for f in (0.05, 0.25, 0.5):
        for tag, g in (("balanced", balanced), ("shifted", shifted)):
            share = elected_share(g, "rate", f)
            assert abs(share - f) < 0.02, \
                f"rate{f} elected {share:.3f} of {tag} tiles, expected ~{f}"

    # the contrast: a fixed harm threshold gives very different shares on the two distributions
    hb = elected_share(balanced, "harm", 1.5)
    hs = elected_share(shifted, "harm", 1.5)
    assert abs(hs - hb) > 0.10, \
        f"h1.5 gave {hb:.3f} and {hs:.3f} -- the two distributions are not different enough " \
        f"for this test to mean anything"

    # endpoints
    assert elected_share(balanced, "rate", 0.0) == 0.0
    assert elected_share(balanced, "rate", 1.0) == elected_share(balanced, "argmin", 0.0)
    print(f"ok  rate<f> elects ~f of tiles on both distributions; h1.5 gives "
          f"{hb:.2f} vs {hs:.2f}")


def test_no_scale_search():
    """
        The scale search is deliberately NOT a factor in this work.

        Reported configurations use `a1`: alpha = 1 on BOTH grids, so the block maximum sits on the
        top code of whichever grid the tile elected -- exactly what plain NVFP4 does. The only thing
        MixFP4 varies is then the element data type, which is the claim being made.

        Removed for this reason:
          `head` / `headx` / `headxx`   E2M1 headroom, alpha > 1
          `heade0` / `heade0x`          the same on the E0M3 branch (alpha = 7/6, 7/5)

        This test exists so the removal cannot be quietly undone. Presets that still search a scale
        (`base`, the `dense*` family, the clipping family) are kept as controls and historical
        record; the test asserts only that `a1` itself performs no search, which is what makes the
        MixFP4-vs-NVFP4 comparison attributable to the type election alone.
    """
    from quantize.quantizer import CLIP_PRESETS, parse_mix_4_6_dtype

    for gone in ("head", "headx", "headxx", "heade0", "heade0x"):
        assert gone not in CLIP_PRESETS, f'"{gone}" is back in CLIP_PRESETS'
        try:
            parse_mix_4_6_dtype(f"mix_4_6_clip{gone}_h1.5")
        except ValueError:
            pass
        else:
            raise AssertionError(f'"clip{gone}" still parses')

    preset = CLIP_PRESETS[REPORTED_CLIP]
    assert preset == {"e2m1": (1.0,), "e0m3": (1.0,)}, \
        f'the reported preset "{REPORTED_CLIP}" must do no scale search, got {preset}'
    print(f'ok  no scale search: head*/heade0* gone, "{REPORTED_CLIP}" is alpha=1 on both grids')


def test_a1_e2m1_is_nvfp4():
    """
        The equivalence the whole comparison rests on.

        With alpha = 1 and the election disabled, MixFP4 has no freedom left: every block takes the
        E2M1 grid with the block maximum on the top code, which IS plain NVFP4. So any delta between
        `mix_4_6_clipa1_<rule>` and `nvfp4` is attributable to the type election and nothing else.

        If this ever fails, every reported delta is confounded.
    """
    from quantize.quantizer import quant_nvfp4

    torch.manual_seed(0)
    for shape in [(512, 512), (128, 4096), (2, 64, 512)]:
        x = torch.randn(*shape).to(torch.bfloat16)
        a = quant_mix_4_6(x, 4, 16, type_block=(8, 64), clip=REPORTED_CLIP, elect="never")
        b = quant_nvfp4(x, 4, 16)
        assert torch.equal(a, b), \
            f"clip={REPORTED_CLIP} + elect=never diverges from nvfp4 at {shape}: " \
            f"max |d| {(a.float() - b.float()).abs().max().item():.3e}"
    print(f'ok  mix_4_6(clip={REPORTED_CLIP}, elect=never) == nvfp4 exactly -- '
          f"deltas isolate the type election")


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
    print(f"\nAll {len(tests)} MixFP4 CPU tests passed.")

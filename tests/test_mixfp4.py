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
    quant_nvif4,
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

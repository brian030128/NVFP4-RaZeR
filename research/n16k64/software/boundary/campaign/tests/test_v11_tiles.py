"""V11: N8/N16 mask expansion, validation, flatten order, all-false/all-true and audited tile totals."""
import json
import os
from pathlib import Path

import pytest
import torch

from campaign import quant as Q
from campaign import tiles as T
from quantize.interacting_format import apply_mask as archived_apply_mask

FINDINGS = {}
ROOT = Path(__file__).resolve().parents[2]


def record(k, v):
    FINDINGS[k] = v
    if os.environ.get('V11_FINDINGS'):
        json.dump(FINDINGS, open(os.environ['V11_FINDINGS'], 'w'), indent=1, sort_keys=True, default=str)


@pytest.mark.parametrize('shape', [(8, 64), (16, 128), (64, 256), (1024, 512), (4096, 1024)])
def test_generalized_n8_bitwise_equals_archived(shape):
    g = torch.Generator().manual_seed(sum(shape))
    base = torch.randn(shape, generator=g).bfloat16()
    alt = torch.randn(shape, generator=g).bfloat16()
    mask = torch.rand(shape[0] // 8, shape[1] // 64, generator=g) < 0.3
    new = T.apply_mask(base, alt, mask, (8, 64))
    old = archived_apply_mask(base, alt, mask)
    assert torch.equal(new, old)
    record(f'archived_n8_equal_{shape[0]}x{shape[1]}', True)


def test_hand_constructed_n16_2x2_layout():
    base = torch.zeros(32, 128, dtype=torch.bfloat16)
    alt = torch.ones(32, 128, dtype=torch.bfloat16)
    mask = torch.tensor([[True, False], [False, True]])
    out = T.apply_mask(base, alt, mask, (16, 64))
    expected = torch.zeros(32, 128, dtype=torch.bfloat16)
    expected[0:16, 0:64] = 1
    expected[16:32, 64:128] = 1
    assert torch.equal(out, expected)
    record('n16_2x2_layout', 'tile (0,0)->rows 0:16 cols 0:64; tile (1,1)->rows 16:32 cols 64:128')


def test_shape_validation_errors():
    b = torch.zeros(32, 128, dtype=torch.bfloat16)
    a = torch.ones(32, 128, dtype=torch.bfloat16)
    m = torch.zeros(2, 2, dtype=torch.bool)
    cases = {
        'wrong_rank_base': (lambda: T.apply_mask(b[None], a[None], m, (16, 64)), ValueError),
        'nonpositive_block': (lambda: T.apply_mask(b, a, m, (0, 64)), ValueError),
        'negative_block': (lambda: T.apply_mask(b, a, m, (16, -64)), ValueError),
        'nondivisible_matrix': (lambda: T.apply_mask(torch.zeros(24, 128, dtype=torch.bfloat16), torch.zeros(24, 128, dtype=torch.bfloat16), m, (16, 64)), ValueError),
        'unequal_candidates': (lambda: T.apply_mask(b, torch.ones(32, 64, dtype=torch.bfloat16), m, (16, 64)), ValueError),
        'unequal_dtypes': (lambda: T.apply_mask(b, a.float(), m, (16, 64)), ValueError),
        'wrong_mask_shape': (lambda: T.apply_mask(b, a, torch.zeros(4, 2, dtype=torch.bool), (16, 64)), ValueError),
        'n8_mask_on_n16_call': (lambda: T.apply_mask(b, a, torch.zeros(4, 2, dtype=torch.bool), (16, 64)), ValueError),
        'non_bool_mask': (lambda: T.apply_mask(b, a, torch.zeros(2, 2), (16, 64)), TypeError),
        'mask_rank': (lambda: T.apply_mask(b, a, torch.zeros(4, dtype=torch.bool), (16, 64)), ValueError),
        'float_block': (lambda: T.apply_mask(b, a, m, (16.5, 64)), ValueError),
    }
    raised = {}
    for name, (fn, exc) in cases.items():
        with pytest.raises(exc):
            fn()
        raised[name] = exc.__name__
    record('validation_errors', raised)


@pytest.mark.parametrize('tb', [(8, 64), (16, 64)])
def test_all_false_and_all_true(tb):
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(64, 256, generator=g) * 0.02).bfloat16()
    base, alt = Q.four_over_six(w), Q.e0m3(w)
    go, gk = T.grid_shape(w.shape, tb)
    assert torch.equal(T.apply_mask(base, alt, torch.zeros(go, gk, dtype=torch.bool), tb), base)
    assert torch.equal(T.apply_mask(base, alt, torch.ones(go, gk, dtype=torch.bool), tb), alt)
    record(f'all_false_all_true_{tb[0]}x{tb[1]}', 'all-false == FourOverSix bitwise; all-true == E0M3 bitwise')


def test_flatten_order_row_major():
    o, k = 48, 192
    labels = torch.arange((o // 16) * (k // 64)).reshape(o // 16, k // 64)
    for flat_index in range(labels.numel()):
        mask = torch.zeros(labels.numel(), dtype=torch.bool)
        mask[flat_index] = True
        out = T.apply_mask(torch.zeros(o, k, dtype=torch.bfloat16), torch.ones(o, k, dtype=torch.bfloat16),
                           mask.reshape(o // 16, k // 64), (16, 64))
        r, c = divmod(flat_index, k // 64)
        rows, cols = out.nonzero(as_tuple=True)
        assert rows.min() == 16 * r and rows.max() == 16 * r + 15 and cols.min() == 64 * c and cols.max() == 64 * c + 63
    record('flatten_order', 'flat tile index i -> (row i // (K/64), col i % (K/64)) for N16 tiles')


def test_n16_mask_equals_union_of_both_children_layout():
    """An N16 tile covers exactly its two vertically adjacent N8 children (layout, not election)."""
    o, k = 64, 128
    for idx in range(((o // 16) * (k // 64))):
        m16 = torch.zeros((o // 16) * (k // 64), dtype=torch.bool)
        m16[idx] = True
        m16 = m16.reshape(o // 16, k // 64)
        m8 = m16.repeat_interleave(2, 0)
        z, one = torch.zeros(o, k, dtype=torch.bfloat16), torch.ones(o, k, dtype=torch.bfloat16)
        assert torch.equal(T.apply_mask(z, one, m16, (16, 64)), T.apply_mask(z, one, m8, (8, 64)))


def test_audited_tile_totals_match_known_results():
    known = json.loads((ROOT.parents[1] / 'handoff' / 'agent_handoff' / 'KNOWN_RESULTS.json').read_text())['models']
    from campaign.models import REGISTRY
    out = {}
    for key in ('llama8b', 'qwen4b', 'qwen27b'):
        rep = json.loads((ROOT / REGISTRY[key]['archived_calibration'] / 'report.json').read_text())
        shapes = [tuple(m['shape']) for m in rep['matrices'].values()]
        bad16 = [s for s in shapes if s[0] % 16 or s[1] % 64]
        n8 = sum((s[0] // 8) * (s[1] // 64) for s in shapes)
        n16 = sum((s[0] // 16) * (s[1] // 64) for s in shapes)
        out[key] = dict(matrices=len(shapes), n8=n8, n16=n16, not_n16_divisible=bad16)
        assert not bad16
        assert n8 == known[key]['n8_total_tiles'] == REGISTRY[key]['n8_total']
        assert n16 == known[key]['n16_total_tiles'] == REGISTRY[key]['n16_total']
    record('audited_tile_totals', out)

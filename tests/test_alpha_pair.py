"""CPU checks for the paired FourOverSix quantizer used by the alpha study."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantize.quantizer import quant_nvfp4_4over6, quant_nvfp4_4over6_pair

G = 16


def sample(rows=64, cols=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, cols, generator=g) * 0.02).to(torch.bfloat16)


def test_selected_is_bit_identical():
    """The chosen branch must reproduce the shipped quantizer exactly."""
    for seed in range(4):
        w = sample(seed=seed)
        chosen, _, _ = quant_nvfp4_4over6_pair(w, 4, G)
        assert torch.equal(chosen, quant_nvfp4_4over6(w, 4, G)), seed
    print('OK chosen branch is bit-identical to quant_nvfp4_4over6')


def test_flip_differs_and_is_a_real_alternative():
    w = sample()
    chosen, other, select_4 = quant_nvfp4_4over6_pair(w, 4, G)
    assert chosen.shape == other.shape == w.shape
    assert select_4.numel() == w.numel() // G
    # Both alphas are used somewhere, otherwise the study has nothing to flip.
    assert 0 < int(select_4.sum()) < select_4.numel(), int(select_4.sum())
    blocks_changed = (chosen != other).reshape(-1, G).any(dim=-1)
    assert bool(blocks_changed.any()), 'flipping alpha changed nothing'
    print(f'OK {int(select_4.sum())}/{select_4.numel()} blocks pick alpha=1.5; '
          f'{int(blocks_changed.sum())} blocks change under the flip')


def test_flip_essentially_always_costs_weight_error():
    """MSE already picked the better branch, so a flip should cost reconstruction.

    The comparison in quant_nvfp4_4over6 is made on the globally scaled float32
    values, while the returned tensor is cast to bfloat16. That cast reorders a
    small number of near-ties, so a few flips measure as marginally better here.
    Those are rounding ties, not a failure of the selection: the property that
    matters for the study is that a flip is never a free win.
    """
    w = sample()
    wf = w.float()
    chosen, other, _ = quant_nvfp4_4over6_pair(w, 4, G)
    e_chosen = ((chosen.float() - wf) ** 2).reshape(-1, G).sum(-1)
    e_other = ((other.float() - wf) ** 2).reshape(-1, G).sum(-1)
    changed = (chosen != other).reshape(-1, G).any(dim=-1)
    better = (e_other < e_chosen) & changed
    fraction = int(better.sum()) / max(int(changed.sum()), 1)
    assert fraction < 0.05, f'{fraction:.1%} of flips lowered weight error'
    if int(better.sum()):
        relative = ((e_chosen - e_other)[better] / e_chosen[better].clamp(min=1e-30))
        assert float(relative.max()) < 0.1, float(relative.max())
    print(f'OK flips cost weight error except {fraction:.1%} of near-ties from the '
          f'bfloat16 cast')


def test_block_granularity_matches_scale_block():
    """Differences must be confined to whole 16-element scale blocks."""
    w = sample()
    chosen, other, _ = quant_nvfp4_4over6_pair(w, 4, G)
    diff = (chosen != other).reshape(-1, G)
    per_block = diff.any(dim=-1)
    assert diff[~per_block].sum() == 0
    print('OK differences are confined to whole scale blocks')


if __name__ == '__main__':
    test_selected_is_bit_identical()
    test_flip_differs_and_is_a_real_alternative()
    test_flip_essentially_always_costs_weight_error()
    test_block_granularity_matches_scale_block()
    print('PASS tests/test_alpha_pair.py')

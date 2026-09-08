import torch
from quantize.relinearized_format import common_descent_scores, next_bit, stale_map


def main():
    ce = torch.tensor([[-3., -5., -2.]]).repeat(4, 1)
    kl = torch.tensor([[-2., 1., -4.]]).repeat(4, 1)
    zero = torch.zeros(3, dtype=torch.bool)
    assert next_bit(ce, kl, zero)[0] == 0  # deterministic first tie
    assert torch.equal(stale_map(ce, kl, 3), torch.tensor([True, False, True]))
    selected = torch.tensor([True, False, False])
    assert next_bit(-ce, -kl, selected)[0] == 0  # undo at changed gradient
    assert next_bit(-ce, -kl, zero)[0] is None
    noisy = ce.clone(); noisy[:, 0] = torch.tensor([-20., 20., -20., 20.])
    assert common_descent_scores(noisy, kl, zero)[0] > 0
    # A single-bit current gradient update follows the exact descent direction
    # on a linear objective, including undo; a stale multi-bit update need not
    # descend on a quadratic with interactions.
    a = torch.tensor([-2., -1.9]); h = torch.full((2, 2), 3.)
    s = torch.zeros(2, dtype=torch.bool)
    i, _ = next_bit(a.repeat(4, 1), a.repeat(4, 1), s); s[i] = True
    g = a + h @ s.float()
    j, _ = next_bit(g.repeat(4, 1), g.repeat(4, 1), s)
    assert j == 0  # full-bit finite curvature still permits cycling: no guarantee
    print('Relinearized score, common descent, undo and uncertainty tests passed')


if __name__ == '__main__': main()

import torch
from quantize.consensus_format import consensus_map, elect_all


def main():
    a = torch.tensor([[-5., 1., -2.], [1., -5., -2.], [-2., -2., -2.]])[:, None, :].repeat(1, 4, 1)
    chosen = consensus_map(a, a)
    assert torch.equal(chosen, torch.tensor([False, False, True]))
    # Each convex combination improves under the exact linear objective.
    for weights in (torch.tensor([1., 0., 0.]), torch.tensor([.2, .3, .5])):
        assert float((weights[:, None]*a.mean(1))[:, chosen].sum()) < 0
    maps = elect_all(a, a)
    assert torch.equal(maps['without_web'], torch.tensor([False, True, True]))
    assert torch.equal(maps['without_math'], torch.tensor([True, False, True]))
    assert int(consensus_map(a, a, 0).sum()) == 0
    print('Consensus, convex-mixture linear objective and leave-source-out tests passed')


if __name__ == '__main__': main()

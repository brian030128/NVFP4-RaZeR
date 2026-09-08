import math
import torch
from quantize.description_format import elect


def main():
    ce = torch.tensor([[-.1, -.01, .1, -.04]], dtype=torch.float64).repeat(4, 1)
    selected, stats = elect(ce, ce, 10)
    assert torch.equal(selected, torch.tensor([True, False, False, True]))
    p = stats['prior_probability']; d = len(selected)
    logprior0 = d*math.log1p(-p)
    logprior1 = int(selected.sum())*math.log(p)+(d-int(selected.sum()))*math.log1p(-p)
    assert abs(logprior0-logprior1-stats['relative_description_nats']) < 1e-12
    # Exact minimizer of the additive penalized surrogate, by enumeration.
    risks = []
    for bits in range(1 << d):
        mask = torch.tensor([bool(bits & (1 << i)) for i in range(d)])
        risks.append(float((ce.mean(0)[mask]+stats['penalty']).sum()))
    actual = float((ce.mean(0)[selected]+stats['penalty']).sum())
    assert abs(actual-min(risks)) < 1e-12
    more, _ = elect(ce, ce, 100)
    assert torch.all(~selected | more)
    print('Sparse-prior code length and exact additive election tests passed')


if __name__ == '__main__': main()

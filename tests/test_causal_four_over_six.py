import torch
from quantize.quantizer import quant_nvfp4_4over6
from quantize.causal_four_over_six import quantize_rows


def main():
    torch.manual_seed(20260928)
    for dtype in (torch.float32, torch.bfloat16):
        x = (torch.randn(2, 9, 128)*torch.logspace(-3, 3, 18).reshape(2,9,1)).to(dtype)
        expected = torch.stack([quant_nvfp4_4over6(row, 4, 16) for row in x.reshape(-1,128)]).reshape_as(x)
        actual = quantize_rows(x)
        assert torch.equal(actual, expected)
        assert torch.equal(quantize_rows(x[:, :4]), actual[:, :4])
        other = x.clone(); other[:, 4:] *= 1000
        assert torch.equal(quantize_rows(other)[:, :4], actual[:, :4])
        assert torch.equal(quantize_rows(x[0]), actual[0])
    assert torch.equal(quantize_rows(torch.zeros(3,64)), torch.zeros(3,64, dtype=torch.bfloat16))
    print('Per-token reference equivalence, prefix/batch independence and zero-input tests passed')


if __name__ == '__main__': main()

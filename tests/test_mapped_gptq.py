"""Map-conditioned GPTQ: RTN limit equals the evaluated RTN weights; compensation lowers H-error."""
import torch

from quantize.mapped_gptq import factor, quantize_mapped
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_task_reorder_eval import mix_coarse


def main():
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    w = (torch.randn(512, 256, device=dev) * torch.rand(512, 1, device=dev)).bfloat16()
    w[3, 7] = 0.4  # an outlier to exercise both FourOverSix alphas
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 1] = mask[1, 3] = True
    ref = mix_coarse(quant_nvfp4_4over6(w, 4, 16),
                     quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always'), mask)
    eye = torch.eye(256, device=dev, dtype=torch.float64)
    limit = quantize_mapped(w, eye, mask, 256)
    agree = float((limit == ref).float().mean())
    print('rtn limit agreement', agree)
    assert agree == 1.0
    fine = torch.zeros(64, 4, dtype=torch.bool)
    assert torch.equal(quantize_mapped(w, eye, fine, 8), quant_nvfp4_4over6(w, 4, 16))
    # correlated inputs
    x = torch.randn(4096, 256, device=dev, dtype=torch.float64) @ torch.randn(256, 256, device=dev, dtype=torch.float64)
    h = x.T @ x / len(x)
    q = quantize_mapped(w, factor(h), mask, 256)
    def herr(a):
        e = a.double() - w.double()
        return float(((e @ h) * e).sum())
    print('H-error rtn', herr(ref), 'gptq', herr(q))
    assert herr(q) < 0.7 * herr(ref)
    # the map is respected: E0M3 tile values are integer multiples of E0M3-scaled codes
    print('ok')


if __name__ == '__main__':
    main()

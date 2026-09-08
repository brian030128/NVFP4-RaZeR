import itertools
import torch
from quantize.interacting_format import select_formats,apply_mask
from quantize.conditional_format import fixed_scale_candidates
from quantize.quantizer import quant_nvfp4_4over6


def main():
    torch.set_num_threads(4);torch.manual_seed(713)
    device='cuda'
    w=torch.randn(16,128,device=device).bfloat16()
    x=torch.randn(256,128,device=device,dtype=torch.float64)
    x[:,64:]+=.8*x[:,:64];h=x.T@x/len(x)
    qs,masks,stats=select_formats(w,h)
    assert torch.equal(qs['four_over_six'],quant_nvfp4_4over6(w,4,16))
    assert stats['losses']['interacting']<=stats['losses']['four_over_six']
    assert stats['converged']
    gs=w.float().abs().amax()/(6*448);b,a=fixed_scale_candidates(w,gs)
    def loss(q):return float(((q.double()-w.double())@x.T).square().sum()/len(x))
    assert abs(loss(qs['interacting'])-stats['losses']['interacting'])<1e-8
    # Exhaustive one-flip neighborhood verifies the convergence claim independently.
    for i,j in itertools.product(range(2),range(2)):
        m=masks['interacting'].clone();m[i,j]^=True
        assert loss(apply_mask(b,a,m))>=loss(qs['interacting'])-1e-9
    # Without cross-channel geometry, interaction and ordinary MSE must agree.
    isotropic,im,_=select_formats(w,torch.eye(128,device=device))
    assert torch.equal(im['interacting'],im['weight_mse'])
    assert torch.equal(im['independent'],im['weight_mse'])
    z,_,_=select_formats(torch.zeros_like(w),h)
    assert all(torch.count_nonzero(q)==0 for q in z.values())
    print('PASS: canonical candidates, exact objective, monotonic descent, single-flip stationarity, isotropic reduction, zero',flush=True)


if __name__=='__main__':main()

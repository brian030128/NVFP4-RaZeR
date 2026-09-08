import torch
from quantize.branched_format import inverse_factor,quantize_branched
from quantize.quantizer import quant_nvfp4_4over6
from quantize.conditional_format import fixed_scale_candidates
from quantize.interacting_format import apply_mask


def main():
    torch.set_num_threads(4);torch.manual_seed(523)
    w=torch.randn(16,128,device='cuda').bfloat16()
    x=torch.randn(256,128,device='cuda',dtype=torch.float64);x[:,64:]+=.7*x[:,:64]
    u,h=inverse_factor(x.T@x/len(x));qs,m,trace=quantize_branched(w,u)
    for p,q in qs.items():
        e=q.double()-w.double();actual=float(((e@h)*e).sum())
        assert abs(actual-trace[p]['conditional_cost_sum'])<1e-7*actual
        assert m[p].shape==(2,2)
    assert torch.count_nonzero(m['e2m1'])==0
    # With no covariance, the sequential algorithm reduces to canonical election.
    qs,m,_=quantize_branched(w,torch.eye(128,device='cuda',dtype=torch.float64))
    assert torch.equal(qs['e2m1'],quant_nvfp4_4over6(w,4,16))
    b,a=fixed_scale_candidates(w,w.float().abs().amax()/(6*448))
    choose=((a.double()-w.double()).square()-(b.double()-w.double()).square()).reshape(2,8,2,64).sum((1,3))<0
    expected=apply_mask(b,a,choose)
    assert torch.equal(qs['dynamic_mse'],expected)
    assert torch.equal(qs['conditional'],expected)
    print('PASS: exact telescoping GPTQ objective, legal shared tile maps, canonical isotropic reduction',flush=True)


if __name__=='__main__':main()

import torch
from quantize.online_format import online_quantize,sketch_weight
from quantize.quantizer import quant_nvfp4_4over6
from quantize.consumer_activation import quantize_input


def main():
    torch.set_num_threads(4);torch.manual_seed(911)
    x=torch.randn(19,128,device='cuda').bfloat16();w=torch.randn(256,128,device='cuda').bfloat16()
    q,s=online_quantize(x,w=w)
    b=quant_nvfp4_4over6(x,4,16)
    actual=float(((q.double()-x.double())@w.double().T).square().sum())
    baseline=float(((b.double()-x.double())@w.double().T).square().sum())
    assert abs(actual-s['final_objective'])<1e-5*actual
    assert actual<=baseline*(1+1e-6)
    eye=torch.eye(128,device='cuda')
    qi,_=online_quantize(x,w=eye);qm,_=quantize_input(x)
    assert torch.equal(qi,qm)
    factor,diagonal=sketch_weight(w)
    assert (factor.numel()+diagonal.numel())*4<=w.numel()/2/16
    q,s=online_quantize(x,sketch=(factor,diagonal))
    e=q.float()-x.float()
    actual=float((e@factor.T).square().sum()+(e.square()*diagonal).sum())
    assert abs(actual-s['final_objective'])<1e-5*actual
    zero,s=online_quantize(torch.zeros_like(x),w=w)
    assert torch.count_nonzero(zero)==0 and s['e0m3']==0
    print('PASS: exact output objective, monotonicity, padded tiles, isotropic MSE reduction, sketch metadata budget, zero',flush=True)


if __name__=='__main__':main()

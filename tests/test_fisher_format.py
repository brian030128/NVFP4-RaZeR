import torch
from quantize.fisher_format import select_fisher,left_metric
from quantize.interacting_format import select_formats,apply_mask
from quantize.conditional_format import fixed_scale_candidates


def main():
    torch.set_num_threads(4);torch.manual_seed(923)
    w=torch.randn(16,128,device='cuda').bfloat16()
    x=torch.randn(256,128,device='cuda',dtype=torch.float64);h=x.T@x/len(x)
    identity=torch.eye(8,device='cuda',dtype=torch.float64).repeat(2,1,1)
    qs,m,_=select_formats(w,h);q,fm,s=select_fisher(w,h,identity)
    assert torch.equal(q,qs['interacting']) and torch.equal(fm,m['interacting'])
    a=torch.randn(2,8,8,device='cuda',dtype=torch.float64);g=a@a.transpose(1,2)
    q,m,s=select_fisher(w,h,g)
    assert s['converged']
    e=q.double()-w.double()
    # Independent dense Kronecker-equivalent objective via output whitening.
    err=(e@x.T).reshape(2,8,-1)
    observed=float((a.transpose(1,2)@err).square().sum()/len(x))
    assert abs(observed-s['trace'][-1])<1e-7
    b,alt=fixed_scale_candidates(w,w.float().abs().amax()/(6*448))
    for i in range(2):
        for j in range(2):
            mm=m.clone();mm[i,j]^=True
            ee=apply_mask(b,alt,mm).double()-w.double()
            assert float((left_metric(ee@h,g)*ee).sum())>=observed-1e-8
    q2,m2,_=select_fisher(w,h,g*7)
    assert torch.equal(q,q2) and torch.equal(m,m2)
    print('PASS: uniform-Fisher reduction, exact whitened objective, stationarity, positive scaling invariance',flush=True)


if __name__=='__main__':main()

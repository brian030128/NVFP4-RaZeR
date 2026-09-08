import torch
from quantize.directed_format import directed_optimize


def main():
    torch.set_num_threads(4);torch.manual_seed(251)
    dtype=torch.float64;device='cuda'
    p=torch.tensor([.2,.3,.5],device=device,dtype=dtype)
    t=torch.tensor([.1,.6,.3],device=device,dtype=dtype)
    j=torch.randn(3,8,device=device,dtype=dtype)
    samples=(p[None,:]-torch.eye(3,device=device,dtype=dtype))@j
    gradient=(p-t)@j;ratio=t/p-1;chi=(p*ratio.square()).sum()
    remainder=samples-ratio[:,None]*gradient[None,:]/chi
    fisher=samples.T@(p[:,None]*samples)
    decomposition=torch.outer(gradient,gradient)/chi+remainder.T@(p[:,None]*remainder)
    assert torch.allclose(fisher,decomposition,rtol=1e-12,atol=1e-12)
    assert torch.allclose((p*ratio)@remainder,torch.zeros_like(gradient),atol=1e-12)
    assert torch.linalg.eigvalsh(fisher-torch.outer(gradient,gradient)/chi).min()>-1e-10
    known=torch.randn(4,32,device=device,dtype=dtype)*.05
    residual=torch.randn(8,32,device=device,dtype=dtype)*.02
    linear=-torch.rand(32,device=device,dtype=dtype)*.02;se=torch.zeros_like(linear);rho=.7
    h=known.T@known+(1-rho)*residual.T@residual+rho*torch.diag(residual.square().sum(0))
    selected,stats=directed_optimize(linear,se,known,residual,rho)
    objective=lambda s:float(linear@s.double()+.5*s.double()@h@s.double())
    assert stats['converged'] and abs(stats['objective']-objective(selected))<1e-10
    for i in range(32):
        s=selected.clone();s[i]^=True
        assert objective(s)>=objective(selected)-1e-10
    print('PASS: exact categorical Fisher decomposition, orthogonal residual, PSD analytic component, dense objective, stationarity',flush=True)


if __name__=='__main__':main()

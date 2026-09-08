import torch
from quantize.predictive_format import optimize,diagonal_shrinkage


def main():
    torch.set_num_threads(4);torch.manual_seed(718)
    v=torch.randn(32,40,device='cuda',dtype=torch.float64)*.03
    linear=-torch.rand(40,device='cuda',dtype=torch.float64)*.01;se=torch.zeros_like(linear)
    rho=diagonal_shrinkage(v);assert 0<=rho<=1
    assert abs(rho-diagonal_shrinkage(v*7))<1e-10
    # Independent entrywise variance calculation checks the shrinkage estimator.
    products=torch.stack([torch.outer(row,row) for row in v])
    s=products.sum(0);mask=~torch.eye(40,device='cuda',dtype=torch.bool)
    numerator=(32*products.square().sum(0)-s.square())/31
    expected=float((numerator[mask].sum()/s[mask].square().sum()).clamp(0,1))
    assert abs(expected-rho)<1e-10
    h=(1-rho)*v.T@v+rho*torch.diag(v.square().sum(0))
    selected,stats=optimize(linear,se,v,rho)
    objective=lambda m:float(linear@m.double()+.5*m.double()@h@m.double())
    assert abs(objective(selected)-stats['objective'])<1e-10 and stats['converged']
    for i in range(40):
        m=selected.clone();m[i]^=True
        assert objective(m)>=objective(selected)-1e-10
    diagonal,_=optimize(linear,se,v,1.)
    assert torch.equal(diagonal,linear+.5*v.square().sum(0)<0)
    print('PASS: covariance-noise formula, scaling invariance, dense quadratic identity, stationarity, diagonal reduction',flush=True)


if __name__=='__main__':main()

import torch
from quantize.global_format import global_election


def main():
    torch.set_num_threads(4);torch.manual_seed(115)
    a=torch.randn(16,48,device='cuda',dtype=torch.float64)*.0002-.0004
    maps,s=global_election(a,511)
    mask=maps['global_fisher'];mu=a.mean(0)
    objective=lambda m:float(mu[m].sum()+511/2*(a[:,m].sum(1).square().mean()))
    assert abs(objective(mask)-s['objective'])<1e-10
    assert s['converged'] and s['objective']<0
    for i in range(a.shape[1]):
        mm=mask.clone();mm[i]^=True
        assert objective(mm)>=objective(mask)-1e-10
    assert all(y<=x for x,y in zip(s['trace'],s['trace'][1:]))
    # Duplicate correlated proposals must not be accepted independently en masse.
    identical=torch.full((16,100),-.001,device='cuda',dtype=torch.float64)
    maps,s=global_election(identical,511)
    assert int(maps['global_fisher'].sum())==2
    assert int(maps['diagonal_fisher'].sum())==100
    print('PASS: exact empirical-Fisher objective, monotonicity, coordinate stationarity, correlated-proposal control',flush=True)


if __name__=='__main__':main()

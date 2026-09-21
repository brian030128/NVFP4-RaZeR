"""Preserve the analytic teacher-error component of predictive Fisher geometry."""
import math
import torch
from quantize.predictive_format import diagonal_shrinkage,optimize


@torch.no_grad()
def directed_optimize(linear,se,known,residual,rho,max_steps=16384):
    eligible=linear+2*se<0
    tail_diag=residual.square().sum(0)
    factors=torch.cat((known,residual*math.sqrt(1-rho)),0)
    ridge=rho*tail_diag;diag=factors.square().sum(0)+ridge
    selected=torch.zeros_like(linear,dtype=torch.bool)
    projected=factors.new_zeros(factors.shape[0]);derivative=linear.clone();trace=[0.]
    for step in range(max_steps):
        gains=torch.where(selected,-derivative,derivative)+.5*diag;gains[~eligible]=torch.inf
        best=int(gains.argmin());change=float(gains[best])
        if change>=-1e-12:break
        sign=-1 if selected[best] else 1;update=sign*factors[:,best]
        projected+=update;derivative+=factors.T@update;derivative[best]+=ridge[best]*sign
        selected[best]^=True;trace.append(trace[-1]+change)
    objective=float(linear[selected].sum()+.5*(projected.square().sum()+ridge[selected].sum()))
    assert abs(objective-trace[-1])<=1e-8*max(1.,abs(objective))
    return selected,dict(rho=rho,steps=len(trace)-1,converged=len(trace)-1<max_steps,
        selected=int(selected.sum()),objective=objective,trace=trace,predicted_linear=float(linear[selected].sum()))


@torch.no_grad()
def directed_election(kl_scores,fisher_scores,chi,ratios,tokens):
    a=kl_scores.double();b=fisher_scores.double();n=a.shape[0]
    mean=a.mean(0);se=a.std(0,unbiased=True)/math.sqrt(n)
    chi=chi.double();ratios=ratios.double()
    assert (chi>=0).all() and torch.isfinite(chi).all()
    inverse=torch.where(chi>0,1/chi,torch.zeros_like(chi))
    known=a*torch.sqrt(tokens*inverse/n)[:,None]
    residual=(b-a*(ratios*inverse)[:,None])*math.sqrt(tokens/n)
    rho=diagonal_shrinkage(residual)
    primary,stats=directed_optimize(mean,se,known,residual,rho)
    raw=b*math.sqrt(tokens/n);raw_rho=diagonal_shrinkage(raw)
    standard,standard_stats=optimize(mean,se,raw,raw_rho,max_steps=16384)
    eligible=mean+2*se<0;diag=known.square().sum(0)+residual.square().sum(0)
    diagonal=eligible & (mean+.5*diag<0)
    order=torch.argsort(torch.where(eligible,mean,torch.inf));benefits=(-mean[order]).clamp_min(0)
    prefix=(benefits.cumsum(0)<=.1)&eligible[order];trust=torch.zeros_like(primary);trust[order[prefix]]=True
    return dict(directed=primary,standard_fisher=standard,diagonal_fisher=diagonal,fixed_budget=trust),dict(
        directed=stats,standard_fisher=standard_stats,mean_chi_per_token=float(chi.mean()/tokens))

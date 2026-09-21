"""Whole-network teacher-KL objective with model-sampled predictive curvature."""
import math
import torch


@torch.no_grad()
def diagonal_shrinkage(v):
    """Estimated off-diagonal covariance noise / observed off-diagonal energy."""
    n=v.shape[0];diagonal=v.square().sum(0)
    off=(v@v.T).square().sum()-diagonal.square().sum()
    if float(off)<=0:return 1.
    fourth=v.square().sum(1).square().sum()-v.pow(4).sum()
    noise=(n*fourth-off)/(n-1)
    return float((noise/off).clamp(0,1))


@torch.no_grad()
def optimize(linear,se,v,rho,max_steps=4096):
    eligible=linear+2*se<0;diag=v.square().sum(0)
    selected=torch.zeros_like(linear,dtype=torch.bool);residual=v.new_zeros(v.shape[0])
    derivative=linear.clone();trace=[0.]
    for step in range(max_steps):
        gains=torch.where(selected,-derivative,derivative)+.5*diag
        gains[~eligible]=torch.inf
        best=int(gains.argmin());change=float(gains[best])
        if change>=-1e-12:break
        sign=-1 if selected[best] else 1;update=sign*v[:,best]
        residual+=update;derivative+=(1-rho)*(v.T@update)
        derivative[best]+=rho*diag[best]*sign;selected[best]^=True
        trace.append(trace[-1]+change)
    objective=float(linear[selected].sum()+.5*((1-rho)*residual.square().sum()+rho*diag[selected].sum()))
    assert abs(objective-trace[-1])<=1e-8*max(1.,abs(objective))
    return selected,dict(rho=rho,steps=len(trace)-1,converged=len(trace)-1<max_steps,
        selected=int(selected.sum()),objective=objective,trace=trace,
        predicted_linear=float(linear[selected].sum()))


@torch.no_grad()
def predictive_election(kl_scores,fisher_scores,tokens):
    a=kl_scores.double();n=a.shape[0]
    mean=a.mean(0);se=a.std(0,unbiased=True)/math.sqrt(n)
    v=fisher_scores.double()*math.sqrt(tokens/n)
    rho=diagonal_shrinkage(v)
    primary,primary_stats=optimize(mean,se,v,rho)
    unshrunk,unshrunk_stats=optimize(mean,se,v,0.)
    eligible=mean+2*se<0;diag=v.square().sum(0)
    diagonal=eligible & (mean+.5*diag<0)
    order=torch.argsort(torch.where(eligible,mean,torch.inf));benefits=(-mean[order]).clamp_min(0)
    prefix=(benefits.cumsum(0)<=.1)&eligible[order]
    trust=torch.zeros_like(primary);trust[order[prefix]]=True
    return dict(predictive=primary,unshrunk=unshrunk,diagonal_fisher=diagonal,fixed_budget=trust),dict(
        predictive=primary_stats,unshrunk=unshrunk_stats,eligible=int(eligible.sum()))

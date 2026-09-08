"""Whole-network binary type election from a low-rank sequence score matrix."""
import math
import torch


@torch.no_grad()
def global_election(scores,tokens,max_steps=4096):
    """scores[i,t] is directional mean-NLL derivative for sequence i, tile t.

    Empirical sequence Fisher, normalized per token, is T/N * A.T A.
    This is an approximate task objective, not an exact loss Hessian.
    """
    a=scores.double();n,count=a.shape
    mean=a.mean(0);se=a.std(0,unbiased=True)/math.sqrt(n)
    eligible=mean+2*se<0
    v=a*math.sqrt(tokens/n);diag=v.square().sum(0)
    selected=torch.zeros(count,device=a.device,dtype=torch.bool)
    residual=torch.zeros(n,device=a.device,dtype=torch.float64)
    derivative=mean.clone();trace=[0.];steps=[]
    for step in range(max_steps):
        gains=torch.where(selected,-derivative,derivative)+.5*diag
        gains[~eligible]=torch.inf
        best=int(gains.argmin());change=float(gains[best])
        if change>=-1e-12:break
        sign=-1 if selected[best] else 1
        update=sign*v[:,best]
        residual+=update;derivative+=v.T@update;selected[best]^=True
        trace.append(trace[-1]+change);steps.append(best)
    exact=float(mean[selected].sum()+.5*residual.square().sum())
    assert abs(exact-trace[-1])<=1e-8*max(1.,abs(exact))
    diagonal=eligible & (mean+.5*diag<0)
    # Historical fixed-budget comparator, with exactly the same observations.
    order=torch.argsort(torch.where(eligible,mean,torch.inf))
    benefits=(-mean[order]).clamp_min(0);prefix=(benefits.cumsum(0)<=.1)&eligible[order]
    trust=torch.zeros_like(selected);trust[order[prefix]]=True
    return dict(global_fisher=selected,diagonal_fisher=diagonal,fixed_budget=trust),dict(
        eligible=int(eligible.sum()),steps=len(steps),converged=len(steps)<max_steps,
        selected=int(selected.sum()),objective=exact,trace=trace,
        predicted_mean_nll=float(mean[selected].sum()),
        sequence_predicted_nll=(a[:,selected].sum(1)).tolist())

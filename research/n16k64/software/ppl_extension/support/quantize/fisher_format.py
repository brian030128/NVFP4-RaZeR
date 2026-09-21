"""Fixed tile candidates with a block-diagonal output Fisher / KFAC objective."""
import torch
from quantize.conditional_format import fixed_scale_candidates
from quantize.interacting_format import apply_mask


def left_metric(x,g):
    return torch.einsum('bij,bjk->bik',g,x.reshape(-1,8,x.shape[-1])).reshape_as(x)


@torch.no_grad()
def select_fisher(w,h,g,max_passes=8):
    h=h.double();g=g.double()
    if g.shape!=(w.shape[0]//8,8,8):raise ValueError('Expected one 8x8 Fisher block per output tile')
    gs=(w.float().abs().amax()/(6*448)).clamp_min(torch.finfo(torch.float32).tiny)
    base,alt=fixed_scale_candidates(w,gs)
    delta=alt.double()-base.double();current=base.double().clone()
    mask=torch.zeros((w.shape[0]//8,w.shape[1]//64),device=w.device,dtype=torch.bool)
    error=current-w.double();gradient=left_metric(error@h,g)
    initial=float((gradient*error).sum());trace=[initial];flips=[]
    for sweep in range(max_passes):
        changed=0
        for j,start in enumerate(range(0,w.shape[1],64)):
            stop=start+64
            d=delta[:,start:stop]*torch.where(mask[:,j].repeat_interleave(8)[:,None],-1.,1.)
            linear=2*(gradient[:,start:stop]*d).sum(-1)
            quadratic=(left_metric(d@h[start:stop,start:stop],g)*d).sum(-1)
            gain=(linear+quadratic).reshape(-1,8).sum(-1)
            tol=1e-12*(linear.abs()+quadratic.abs()).reshape(-1,8).sum(-1)
            accept=gain < -tol
            update=d*accept.repeat_interleave(8)[:,None]
            gradient+=left_metric(update@h[start:stop,:],g)
            current[:,start:stop]+=update
            mask[:,j]^=accept;changed+=int(accept.sum())
        error=current-w.double();actual=float((left_metric(error@h,g)*error).sum())
        assert actual<=trace[-1]+1e-9*max(initial,1e-30)
        trace.append(actual);flips.append(changed)
        if changed==0:break
    q=apply_mask(base,alt,mask)
    assert torch.equal(current.bfloat16(),q)
    return q,mask,dict(trace=trace,flips=flips,converged=flips[-1]==0)

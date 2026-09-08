"""Binary tile coordinate descent on fully quantized layer reconstruction.

Candidates and scales are frozen at the original weights. No floating-point
compensation, task-loss feedback, or per-domain configuration selection.
"""
import torch
from quantize.conditional_format import fixed_scale_candidates, grouped_cost


def apply_mask(base, alt, mask):
    expanded=mask.repeat_interleave(8,0).repeat_interleave(64,1)
    return torch.where(expanded,alt,base)


@torch.no_grad()
def select_formats(w,h,max_passes=8):
    if w.ndim!=2 or w.shape[0]%8 or w.shape[1]%64:
        raise ValueError('Expected weights divisible by 8x64')
    h=h.double()
    gs=(w.float().abs().amax()/(6*448)).clamp_min(torch.finfo(torch.float32).tiny)
    base,alt=fixed_scale_candidates(w,gs)
    delta=alt.double()-base.double();error=base.double()-w.double()
    gradient=error@h
    masks={p:torch.zeros((w.shape[0]//8,w.shape[1]//64),dtype=torch.bool,device=w.device)
           for p in ('weight_mse','independent','interacting')}
    for j,start in enumerate(range(0,w.shape[1],64)):
        stop=start+64;d=delta[:,start:stop];e=error[:,start:stop]
        masks['weight_mse'][:,j]=grouped_cost(e+d)<grouped_cost(e)
        masks['independent'][:,j]=(2*(gradient[:,start:stop]*d).sum(-1)+(d@h[start:stop,start:stop]*d).sum(-1)).reshape(-1,8).sum(-1)<0
    current=base.double().clone();mask=masks['interacting']
    initial=float((gradient*error).sum());trace=[initial];flips=[]
    for sweep in range(max_passes):
        changed=0
        for j,start in enumerate(range(0,w.shape[1],64)):
            stop=start+64
            d=delta[:,start:stop]*torch.where(mask[:,j].repeat_interleave(8)[:,None],-1.,1.)
            linear=2*(gradient[:,start:stop]*d).sum(-1)
            quadratic=(d@h[start:stop,start:stop]*d).sum(-1)
            gain=(linear+quadratic).reshape(-1,8).sum(-1)
            # Roundoff guard, not an accuracy-dependent decision threshold.
            tol=1e-12*(linear.abs()+quadratic.abs()).reshape(-1,8).sum(-1)
            accept=gain < -tol
            update=d*accept.repeat_interleave(8)[:,None]
            gradient+=update@h[start:stop,:]
            current[:,start:stop]+=update
            mask[:,j]^=accept;changed+=int(accept.sum())
        err=current-w.double();actual=float(((err@h)*err).sum())
        assert actual<=trace[-1]+1e-9*max(initial,1e-30)
        assert torch.allclose(gradient,err@h,rtol=1e-8,atol=1e-9)
        trace.append(actual);flips.append(changed)
        if changed==0:break
    outputs={'four_over_six':base,**{p:apply_mask(base,alt,m) for p,m in masks.items()}}
    assert torch.equal(current.bfloat16(),outputs['interacting'])
    losses={}
    for p,q in outputs.items():
        err=q.double()-w.double();losses[p]=float(((err@h)*err).sum())
    return outputs,masks,dict(losses=losses,trace=trace,flips=flips,converged=flips[-1]==0)

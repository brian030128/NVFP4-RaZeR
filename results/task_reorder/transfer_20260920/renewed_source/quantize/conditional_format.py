"""Groupwise error compensation with a fixed, conditional format objective.

Research reference: full covariance, 64-column steps, 8-row shared formats.
Calibration second moments are collected once and shared by every comparator.
"""
import torch


@torch.no_grad()
def fixed_scale_candidates(w,global_scale):
    shape=w.shape
    x=w.float().reshape(-1,16)/global_scale
    maximum=x.abs().amax(-1,keepdim=True)
    levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=w.device)
    mid=(levels[:-1]+levels[1:])/2
    choices=[];errors=[]
    for qmax in (6.,4.):
        scale=(maximum/qmax).clamp(2**-9,448).to(torch.float8_e4m3fn).float()
        dq=levels[torch.bucketize((x/scale).abs().contiguous(),mid)]*x.sign()*scale
        choices.append(dq);errors.append((dq-x).square().sum(-1,keepdim=True))
    b=torch.where(errors[1]<errors[0],choices[1],choices[0])
    s0=(maximum/7).clamp(2**-9,448).to(torch.float8_e4m3fn).float()
    a=(x/s0).round().clamp(-7,7)*s0
    return ((b*global_scale).reshape(shape).bfloat16(),(a*global_scale).reshape(shape).bfloat16())


@torch.no_grad()
def compensation_plan(h):
    """Return conditional quadratic metrics and optimal future-weight updates."""
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or h.shape[0]%64:
        raise ValueError('Expected square second moment divisible by 64')
    h=h.double()
    damping=.01*float(h.diag().mean())
    if damping<=0 or not torch.isfinite(h).all():
        raise ValueError('Second moment must be finite and have positive trace')
    regularized=h+damping*torch.eye(h.shape[0],dtype=h.dtype,device=h.device)
    p=torch.cholesky_inverse(torch.linalg.cholesky(regularized))
    plan=[]
    for start in range(0,h.shape[0],64):
        t=torch.cholesky_inverse(torch.linalg.cholesky(p[:64,:64]))
        c=t@p[:64,64:]
        plan.append((t,c))
        if p.shape[0]>64:
            p=p[64:,64:]-p[64:,:64]@c
            p=(p+p.T)/2
    return plan,regularized


def grouped_cost(error,metric=None):
    e=error.double()
    costs=e.square().sum(-1) if metric is None else ((e@metric)*e).sum(-1)
    return costs.reshape(-1,8).sum(-1)


@torch.no_grad()
def quantize_compensated(w,plan,policy,global_scale=None):
    if w.ndim!=2 or w.shape[0]%8 or w.shape[1]%64:
        raise ValueError('Weight shape must be divisible by 8x64')
    if policy not in ('e2m1','static_mse','dynamic_mse','conditional'):
        raise ValueError(policy)
    if len(plan)!=w.shape[1]//64:
        raise ValueError('Plan does not match weight input dimension')
    gs=(w.float().abs().amax()/(6*448)).clamp_min(torch.finfo(torch.float32).tiny) if global_scale is None else global_scale
    work=w.double().clone();out=torch.empty_like(w,dtype=torch.bfloat16)
    masks=[];trace=[]
    for j,(metric,compensator) in enumerate(plan):
        start=j*64;current=work[:,start:start+64]
        base,alt=fixed_scale_candidates(current,gs)
        eb=base.double()-current;ea=alt.double()-current
        if policy=='e2m1':
            mask=torch.zeros(w.shape[0]//8,dtype=torch.bool,device=w.device)
        elif policy=='static_mse':
            original=w[:,start:start+64]
            ob,oa=fixed_scale_candidates(original,gs)
            mask=grouped_cost(oa.double()-original.double())<grouped_cost(ob.double()-original.double())
        else:
            scoring=metric if policy=='conditional' else None
            mask=grouped_cost(ea,scoring)<grouped_cost(eb,scoring)
        q=torch.where(mask.repeat_interleave(8)[:,None],alt,base)
        delta=q.double()-current
        out[:,start:start+64]=q
        work[:,start:start+64]=q.double()
        work[:,start+64:]+=delta@compensator
        trace.append(float(grouped_cost(delta,metric).sum()))
        masks.append(mask)
    return out,torch.stack(masks,dim=1),dict(conditional_costs=trace,conditional_cost_sum=sum(trace))


@torch.no_grad()
def asymmetric_transport(cross,regularized,damping):
    """A such that W A minimizes output mismatch plus damping ||W A-W||^2.

    cross=E[x xq.T], regularized=E[xq xq.T]+damping I, column-vector convention.
    This is ridge regression / asymmetric calibration, not a new theorem.
    """
    identity=torch.eye(regularized.shape[0],device=regularized.device,dtype=torch.float64)
    return torch.linalg.solve(regularized,(cross.double()+damping*identity).T).T

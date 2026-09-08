"""Calibration-free activation tile election against joint W4A4 output error."""
import torch
import torch.nn.functional as F
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6


@torch.no_grad()
def joint_sketch(wq,w):
    if w.shape[0]<192:raise ValueError('Output dimension must support the metadata budget')
    # Two rank-r factors and three diagonal vectors, <=1/8 packed W4 bytes.
    rank=max(0,min(16,(w.shape[0]//64-3)//2,w.shape[1]))
    wq=wq.float();error=wq-w.float()
    with torch.random.fork_rng(devices=[wq.device.index]):
        torch.manual_seed(20260918)
        if rank:
            u,_,_=torch.svd_lowrank(wq,q=rank,niter=2)
            factor=u.T@wq;offset=u.T@error
        else:factor=wq.new_empty((0,w.shape[1]));offset=factor.clone()
    d=(wq.square().sum(0)-factor.square().sum(0)).clamp_min(0)
    e=(error.square().sum(0)-offset.square().sum(0)).clamp_min(0)
    c=(wq*error).sum(0)-(factor*offset).sum(0)
    bound=(d*e).sqrt();c=c.clamp(-bound,bound)
    return factor.contiguous(),offset.contiguous(),d,c,e


@torch.no_grad()
def joint_quantize(x,wq=None,weight_error=None,sketch=None):
    if (wq is None)==(sketch is None):raise ValueError('Supply exact weights or sketch')
    original=x.float().reshape(-1,x.shape[-1]);m,k=original.shape
    if k%64:raise ValueError('Expected K divisible by64')
    if not torch.count_nonzero(original):
        return torch.zeros_like(x,dtype=torch.bfloat16),dict(tiles=((m+15)//16)*(k//64),e0m3=0,
            initial_objective=0.,final_objective=0.)
    base=quant_nvfp4_4over6(x,4,16).float().reshape(m,k)
    alt=quant_mix_4_6(x,4,16,type_block=(16,64),clip='a1',elect='always').float().reshape(m,k)
    pad=(-m)%16
    original=F.pad(original,(0,0,0,pad));base=F.pad(base,(0,0,0,pad));alt=F.pad(alt,(0,0,0,pad))
    err=base-original;delta=alt-base;out=base.clone()
    if sketch is None:
        factor=wq.float();offset=weight_error.float();d=c=e=None
    else:factor,offset,d,c,e=sketch
    residual=err@factor.T+original@offset.T
    def objective():
        value=residual.square().sum()
        if d is not None:value+=(err.square()*d+2*err*original*c+original.square()*e).sum()
        return value
    initial=objective();selected=0
    for start in range(0,k,64):
        stop=start+64;diff=delta[:,start:stop];change=diff@factor[:,start:stop].T
        linear=2*(residual*change).sum(-1);quad=change.square().sum(-1)
        if d is not None:
            linear+=2*(diff*(err[:,start:stop]*d[start:stop]+original[:,start:stop]*c[start:stop])).sum(-1)
            quad+=(diff.square()*d[start:stop]).sum(-1)
        cost=(linear+quad).reshape(-1,16).sum(-1)
        tolerance=8*torch.finfo(torch.float32).eps*(linear.abs()+quad.abs()).reshape(-1,16).sum(-1)
        mask=cost < -tolerance;expand=mask.repeat_interleave(16)[:,None]
        residual+=change*expand;err[:,start:stop]+=diff*expand;out[:,start:stop]+=diff*expand
        selected+=int(mask.sum())
    final=objective()
    assert float(final)<=float(initial)*(1+1e-4)+1e-15
    return out[:m].reshape_as(x).bfloat16(),dict(tiles=(m+pad)//16*(k//64),e0m3=selected,
        initial_objective=float(initial),final_objective=float(final))

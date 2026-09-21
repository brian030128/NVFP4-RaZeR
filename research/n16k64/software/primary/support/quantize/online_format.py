"""Calibration-free online format election with coupled output errors."""
import torch
import torch.nn.functional as F
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6


@torch.no_grad()
def sketch_weight(w):
    # Float32 metadata capped at 1/16 of the ideal packed W4 payload.
    rank=max(0,min(16,w.shape[0]//128-1,w.shape[1]))
    w=w.float()
    with torch.random.fork_rng(devices=[w.device.index]):
        torch.manual_seed(20260918)
        if rank:
            _,s,v=torch.svd_lowrank(w,q=rank,niter=2)
            factor=(v*s).T.contiguous()
        else:factor=w.new_empty((0,w.shape[1]))
    diagonal=(w.square().sum(0)-factor.square().sum(0)).clamp_min(0)
    return factor,diagonal


@torch.no_grad()
def online_quantize(x,w=None,sketch=None):
    if (w is None)==(sketch is None):raise ValueError('Supply exact weights or a sketch')
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
    factor=w.float() if w is not None else sketch[0]
    diagonal=None if w is not None else sketch[1]
    residual=err@factor.T
    initial=residual.square().sum()
    if diagonal is not None:initial+=(err.square()*diagonal).sum()
    selected=0
    for start in range(0,k,64):
        stop=start+64;d=delta[:,start:stop];change=d@factor[:,start:stop].T
        linear=2*(residual*change).sum(-1);quad=change.square().sum(-1)
        if diagonal is not None:
            linear+=2*(err[:,start:stop]*d*diagonal[start:stop]).sum(-1)
            quad+=(d.square()*diagonal[start:stop]).sum(-1)
        cost=(linear+quad).reshape(-1,16).sum(-1)
        tolerance=8*torch.finfo(torch.float32).eps*(linear.abs()+quad.abs()).reshape(-1,16).sum(-1)
        mask=cost < -tolerance;expand=mask.repeat_interleave(16)[:,None]
        residual+=change*expand;err[:,start:stop]+=d*expand;out[:,start:stop]+=d*expand
        selected+=int(mask.sum())
    final=residual.square().sum()
    if diagonal is not None:final+=(err.square()*diagonal).sum()
    assert float(final)<=float(initial)*(1+1e-4)+1e-15
    return out[:m].reshape_as(x).bfloat16(),dict(tiles=(m+pad)//16*(k//64),e0m3=selected,
        initial_objective=float(initial),final_objective=float(final))

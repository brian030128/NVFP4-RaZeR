"""Calibration-free activation format election using consuming-weight energy.

Diagonal layer-output geometry only; no per-input or task-loss guarantee.
"""
import torch
import torch.nn.functional as F
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6


@torch.no_grad()
def consumer_energy(weight):
    d=weight.float().square().sum(0)
    return d/d.mean().clamp_min(torch.finfo(torch.float32).tiny)


@torch.no_grad()
def quantize_input(x,importance=None):
    k=x.shape[-1]
    if k%64:raise ValueError('Expected input dimension divisible by 64')
    if not torch.count_nonzero(x):return torch.zeros_like(x,dtype=torch.bfloat16),dict(tiles=0,e0m3=0)
    b=quant_nvfp4_4over6(x,4,16)
    a=quant_mix_4_6(x,4,16,type_block=(16,64),clip='a1',elect='always')
    flat=x.reshape(-1,k).float();rows=len(flat);padding=(-rows)%16
    gain=(a.reshape_as(flat).float()-flat).square()-(b.reshape_as(flat).float()-flat).square()
    if importance is not None:
        if importance.shape!=(k,) or (importance<0).any() or not torch.isfinite(importance).all():
            raise ValueError('Expected finite nonnegative per-channel weights')
        gain*=importance
    gain=F.pad(gain,(0,0,0,padding)).reshape(-1,16,k//64,64).sum((1,3))
    mask=gain<0
    expanded=mask.repeat_interleave(16,0).repeat_interleave(64,1)[:rows]
    q=torch.where(expanded,a.reshape_as(flat),b.reshape_as(flat)).reshape_as(x)
    return q,dict(tiles=mask.numel(),e0m3=int(mask.sum()))

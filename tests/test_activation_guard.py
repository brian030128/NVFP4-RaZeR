"""Analytic-bound and map tests; execute only on allocated compute."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from quantize.activation_guard import (build_geometry,upper_change,joint_nuclear_difference,
    proxy,propose,guarded_map,apply_map,activation_candidates)
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6


def main():
    torch.set_num_threads(2)
    g=torch.Generator().manual_seed(514)
    w=torch.randn(80,128,generator=g,dtype=torch.float64)
    w[:,3]=0
    geo,info=build_geometry(w,rank=16)
    normalized=w/geo.scale
    actual_residual=normalized.T@normalized-geo.center*torch.eye(128,dtype=torch.float64)-(geo.factor@geo.factor.T)
    assert float(torch.linalg.eigvalsh(actual_residual).abs().max()) <= geo.epsilon+1e-10
    for _ in range(12):
        a=torch.randn(16,128,generator=g,dtype=torch.float64)
        b=torch.randn(16,128,generator=g,dtype=torch.float64)
        nuclear=torch.linalg.eigvalsh(a.T@a-b.T@b).abs().sum()
        assert torch.allclose(nuclear,joint_nuclear_difference(a,b),rtol=1e-10,atol=1e-8)
        upper=upper_change(a,b,geo)['upper_change']
        actual=float((a@w.T).square().sum()-(b@w.T).square().sum())
        assert actual <= upper+1e-8*max(abs(actual),1)
    # Tight deterministic case: isotropic geometry has essentially zero residual.
    iso,_=build_geometry(torch.eye(128,dtype=torch.float64),rank=16)
    b=torch.randn(16,128,generator=g,dtype=torch.float64)
    assert upper_change(.5*b,b,iso)['upper_change']<0
    assert abs(upper_change(b,b,iso)['upper_change'])<1e-8
    x=torch.randn(16,128,generator=g).bfloat16()
    base,alt=activation_candidates(x)
    assert torch.equal(base,quant_nvfp4_4over6(x,4,16))
    assert torch.equal(alt,quant_mix_4_6(x,4,16,type_block=(16,64),clip='a1',elect='always'))
    # Incremental sweep agrees with direct recomputation of the joint proxy.
    mask=propose(x,base,alt,geo)
    ref=torch.zeros(2,dtype=torch.bool)
    for i in range(2):
        trial=ref.clone();trial[i]=True
        if proxy(apply_map(base,alt,trial).double()-x.double(),geo) < proxy(apply_map(base,alt,ref).double()-x.double(),geo):
            ref=trial
    assert torch.equal(mask,ref)
    final,proposal,record=guarded_map(x,base,alt,geo)
    assert torch.equal(final,guarded_map(x,base,alt,geo)[0])
    error=apply_map(base,alt,final).double()-x.double()
    assert float((error@w.T).square().sum()) <= float(((base.double()-x.double())@w.T).square().sum())+1e-8
    zero=torch.zeros(16,128);zb,za=activation_candidates(zero)
    assert not guarded_map(zero,zb,za,geo)[0].any()
    # Joint error differs from summed isolated costs: aligned errors reinforce.
    W=torch.ones(1,2,dtype=torch.float64)
    e1=torch.tensor([[1.,0.]],dtype=torch.float64);e2=e1.flip(1)
    assert ((e1+e2)@W.T).square().sum() > (e1@W.T).square().sum()+(e2@W.T).square().sum()
    print('PASS: residual bound, nuclear identity, joint inequality, tight case, canonical candidates, incremental sweep, deterministic guard, zero, interactions')


if __name__=='__main__':
    main()

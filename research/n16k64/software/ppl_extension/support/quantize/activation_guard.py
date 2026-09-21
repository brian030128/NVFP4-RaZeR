"""Reference implementation of a weight-only joint activation-error bound.

The inequality is analytic; float64 implementation is not interval arithmetic.
The guard controls packet Frobenius output error, not every token or model NLL.
"""
from dataclasses import dataclass
import torch
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6


@dataclass
class Geometry:
    scale: torch.Tensor
    factor: torch.Tensor
    center: float
    epsilon: float


@torch.no_grad()
def build_geometry(weight, rank=16):
    w=weight.double()
    gram=w.T@w
    diagonal=gram.diag().clamp_min(0).sqrt()
    # A zero column has no output effect. A unit scale keeps its normalization defined.
    scale=torch.where(diagonal>0,diagonal,torch.ones_like(diagonal))
    correlation=gram/scale[:,None]/scale[None,:]
    values,vectors=torch.linalg.eigh(correlation)
    k=correlation.shape[0]
    if not 0<=rank<k:
        raise ValueError('rank must lie in [0, input_dimension)')
    tail=values[:k-rank]
    center=float((tail[0]+tail[-1])/2)
    radius=float((tail[-1]-tail[0])/2)
    if rank:
        factor=vectors[:,-rank:]*(values[-rank:]-center).clamp_min(0).sqrt()[None,:]
    else:
        factor=vectors[:,:0]
    # Conservative real-arithmetic bound even for an imperfect eigendecomposition.
    orth=float(torch.linalg.matrix_norm(vectors.T@vectors-torch.eye(k,device=w.device,dtype=w.dtype)))
    reconstruction=float(torch.linalg.matrix_norm(correlation-(vectors*values[None,:])@vectors.T))
    epsilon=radius*(1+orth)+abs(center)*orth+reconstruction
    # Numerical slack is not a formal floating-point proof.
    epsilon+=1e-10*max(float(correlation.abs().max()),1.)
    geometry=Geometry(scale,factor,center,epsilon)
    return geometry, dict(rank=rank,epsilon=epsilon,center=center,tail_radius=radius,
        eig_orthogonality_frobenius=orth,eig_reconstruction_frobenius=reconstruction,
        metadata_bytes=sum(t.numel()*t.element_size() for t in (scale,factor))+16,
        input_dimension=k,output_dimension=w.shape[0])


def proxy(error,geometry):
    z=error.double()*geometry.scale
    return geometry.center*z.square().sum()+(z@geometry.factor).square().sum()


def joint_nuclear_difference(a,b):
    """||a.T a-b.T b||_* from a matrix of dimension <=2*packet_tokens.

    Includes all input-channel and token interactions; no independent-tile gates.
    """
    both=torch.cat((a.T,b.T),dim=1)
    _,r=torch.linalg.qr(both,mode='reduced')
    signs=torch.cat((torch.ones(a.shape[0],device=a.device,dtype=a.dtype),
                     -torch.ones(b.shape[0],device=b.device,dtype=b.dtype)))
    small=(r*signs[None,:])@r.T
    return torch.linalg.eigvalsh(small).abs().sum()


def upper_change(candidate_error,baseline_error,geometry):
    a=candidate_error.double()*geometry.scale
    b=baseline_error.double()*geometry.scale
    delta=proxy(candidate_error,geometry)-proxy(baseline_error,geometry)
    penalty=geometry.epsilon*joint_nuclear_difference(a,b)
    return dict(proxy_change=float(delta),uncertainty_penalty=float(penalty),upper_change=float(delta+penalty))


@torch.no_grad()
def activation_candidates(x):
    if x.ndim!=2 or x.shape[0]!=16 or x.shape[1]%64:
        raise ValueError('Expected a 16-token packet and complete 64-channel tiles')
    if not torch.isfinite(x).all():
        raise ValueError('Activations must be finite')
    if not torch.count_nonzero(x):
        z=torch.zeros_like(x,dtype=torch.bfloat16)
        return z,z.clone()
    return (quant_nvfp4_4over6(x,4,16),
            quant_mix_4_6(x,4,16,type_block=(16,64),clip='a1',elect='always'))


def apply_map(base,alt,mask):
    return torch.where(mask.repeat_interleave(64)[None,:],alt,base)


@torch.no_grad()
def propose(x,base,alt,geometry):
    """One fixed left-to-right sweep minimizing the proxy by finite tile changes."""
    z=(base.double()-x.double())*geometry.scale
    dz=(alt.double()-base.double())*geometry.scale
    projected=z@geometry.factor
    mask=torch.zeros(x.shape[1]//64,dtype=torch.bool,device=x.device)
    for i in range(mask.numel()):
        sl=slice(i*64,(i+1)*64)
        d=dz[:,sl]; dp=d@geometry.factor[sl]
        change=geometry.center*(2*(z[:,sl]*d).sum()+d.square().sum())
        change+=2*(projected*dp).sum()+dp.square().sum()
        if float(change)<0:
            mask[i]=True
            z[:,sl]+=d
            projected+=dp
    return mask


@torch.no_grad()
def guarded_map(x,base,alt,geometry):
    proposal=propose(x,base,alt,geometry)
    q=apply_map(base,alt,proposal)
    record=upper_change(q.double()-x.double(),base.double()-x.double(),geometry)
    # A strict relative numerical margin, fixed before evaluating prompts.
    margin=1e-10*max(float(proxy(base.double()-x.double(),geometry).abs()),1e-30)
    accepted=bool(proposal.any()) and record['upper_change'] < -margin
    record.update(accepted=accepted,proposal_tiles=int(proposal.sum()),numerical_margin=margin)
    return proposal if accepted else torch.zeros_like(proposal),proposal,record

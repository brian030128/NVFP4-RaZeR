"""Legal tile formats with columnwise GPTQ compensation inside each branch."""
import torch


@torch.no_grad()
def inverse_factor(h):
    h=h.double();damping=.01*float(h.diag().mean())
    if damping<=0:raise ValueError('Positive input second moment required')
    regularized=h+damping*torch.eye(len(h),device=h.device,dtype=h.dtype)
    inverse=torch.cholesky_inverse(torch.linalg.cholesky(regularized))
    return torch.linalg.cholesky(inverse,upper=True),regularized


@torch.no_grad()
def quantize_branched(w,u):
    if w.ndim!=2 or w.shape[0]%8 or w.shape[1]%64:raise ValueError('Expected 8x64 aligned weights')
    policies=('e2m1','dynamic_mse','conditional')
    gs=(w.float().abs().amax()/(6*448)).clamp_min(torch.finfo(torch.float32).tiny)
    work=w.double().unsqueeze(0).repeat(3,1,1)
    result=torch.empty_like(work,dtype=torch.bfloat16);masks=[];costs=[]
    levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=w.device,dtype=torch.float64)
    mid=(levels[:-1]+levels[1:])/2
    for start in range(0,w.shape[1],64):
        stop=start+64;original=work[:,:,start:stop].clone()
        branch=original[:,None].repeat(1,2,1,1);q=torch.empty_like(branch);innov=torch.empty_like(branch)
        local=u[start:stop,start:stop]
        for j in range(64):
            if j%16==0:
                x=branch[:,0,:,j:j+16]/gs;maximum=x.abs().amax(-1)
                candidates=[];errors=[];scales=[]
                for qmax in (6.,4.):
                    scale=(maximum/qmax).clamp(2**-9,448).to(torch.float8_e4m3fn).double()
                    dq=levels[torch.bucketize((x/scale[...,None]).abs().contiguous(),mid)]*x.sign()*scale[...,None]
                    scales.append(scale);errors.append((dq-x).square().sum(-1))
                s2=torch.where(errors[1]<errors[0],scales[1],scales[0])*gs
                s0=(branch[:,1,:,j:j+16].abs().amax(-1)/gs/7).clamp(2**-9,448).to(torch.float8_e4m3fn).double()*gs
            x2=branch[:,0,:,j];x0=branch[:,1,:,j]
            q2=(levels[torch.bucketize((x2/s2).abs().contiguous(),mid)]*x2.sign()*s2).bfloat16().double()
            q0=((x0/s0).round().clamp(-7,7)*s0).bfloat16().double()
            current=torch.stack((q2,q0),1)
            err=(current-branch[:,:,:,j])/local[j,j]
            q[:,:,:,j]=current;innov[:,:,:,j]=err
            branch[:,:,:,j+1:]+=err[...,None]*local[j,j+1:]
        mse=(q-original[:,None]).square().reshape(3,2,-1,8,64).sum((-1,-2))
        conditional=innov.square().reshape(3,2,-1,8,64).sum((-1,-2))
        mask=torch.zeros((3,w.shape[0]//8),device=w.device,dtype=torch.bool)
        mask[1]=mse[1,1]<mse[1,0];mask[2]=conditional[2,1]<conditional[2,0]
        expand=mask.repeat_interleave(8,1)[...,None]
        chosen=torch.where(expand,q[:,1],q[:,0]);errors=torch.where(expand,innov[:,1],innov[:,0])
        result[:,:,start:stop]=chosen.bfloat16();work[:,:,start:stop]=chosen
        work[:,:,stop:]+=errors@u[start:stop,stop:]
        masks.append(mask);costs.append(errors.square().sum((1,2)))
    mask=torch.stack(masks,-1);trace=torch.stack(costs,-1)
    return {p:result[i] for i,p in enumerate(policies)}, {p:mask[i] for i,p in enumerate(policies)}, {
        p:dict(conditional_costs=trace[i].tolist(),conditional_cost_sum=float(trace[i].sum())) for i,p in enumerate(policies)}

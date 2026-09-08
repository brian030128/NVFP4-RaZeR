import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from quantize.conditional_format import fixed_scale_candidates,compensation_plan,quantize_compensated,asymmetric_transport
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6


def main():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(404)
    w=torch.randn(16,128,generator=g).bfloat16()
    gs=w.float().abs().max()/(6*448)
    b,a=fixed_scale_candidates(w,gs)
    assert torch.equal(b,quant_nvfp4_4over6(w,4,16))
    assert torch.equal(a,quant_mix_4_6(w,4,16,type_block=(8,64),clip='a1',elect='always'))
    x=torch.randn(256,128,generator=g,dtype=torch.float64)
    x[:,64:]+=.7*x[:,:64]
    h=x.T@x/len(x);plan,hr=compensation_plan(h)
    # Conditional cost is exactly the minimum cost with other coordinates free.
    delta=torch.randn(3,64,generator=g,dtype=torch.float64)
    t,c=plan[0];full=torch.cat((delta,delta@c),dim=1)
    direct=((full@hr)*full).sum();predicted=((delta@t)*delta).sum()
    assert torch.allclose(direct,predicted,rtol=1e-9,atol=1e-9)
    grad=full@hr
    assert torch.allclose(grad[:,64:],torch.zeros_like(grad[:,64:]),atol=1e-9)
    for policy in ('e2m1','static_mse','dynamic_mse','conditional'):
        q,m,tr=quantize_compensated(w,plan,policy)
        e=q.double()-w.double();actual=float(((e@hr)*e).sum())
        assert abs(actual-tr['conditional_cost_sum'])<1e-8*max(actual,1)
        assert q.dtype==torch.bfloat16 and tuple(m.shape)==(2,2)
        assert torch.equal(q,quantize_compensated(w,plan,policy)[0])
    # With independent inputs, the conditional metric is scalar identity;
    # dynamic MSE and conditional choice must coincide.
    ip,_=compensation_plan(torch.eye(128,dtype=torch.float64))
    assert torch.equal(quantize_compensated(w,ip,'dynamic_mse')[0],quantize_compensated(w,ip,'conditional')[0])
    xq=x+.15*torch.randn(x.shape,generator=g,dtype=torch.float64)
    hq=xq.T@xq/len(xq);cross=x.T@xq/len(x)
    qp,reg=compensation_plan(hq);damping=.01*float(hq.diag().mean())
    center=w.double()@asymmetric_transport(cross,reg,damping)
    assert torch.allclose(center@reg,w.double()@cross+damping*w.double(),rtol=1e-9,atol=1e-9)
    q,_,trace=quantize_compensated(center,qp,'conditional',global_scale=gs)
    def objective(v):
        return (xq@v.T-x@w.double().T).square().sum()/len(x)+damping*(v-w.double()).square().sum()
    excess=float(objective(q.double())-objective(center))
    assert abs(excess-trace['conditional_cost_sum'])<1e-8*max(excess,1)
    print('PASS: canonical fixed-scale candidates, exact conditional cost, optimal compensation, telescoping objective, legal maps, determinism, isotropic equivalence')
    print('PASS: asymmetric ridge stationarity and exact output-objective decomposition')


if __name__=='__main__':main()

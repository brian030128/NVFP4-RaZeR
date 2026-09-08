import torch
from quantize.joint_online_format import joint_sketch,joint_quantize
from quantize.online_format import online_quantize
from quantize.quantizer import quant_nvfp4_4over6


def main():
    torch.set_num_threads(4);torch.manual_seed(344)
    x=torch.randn(19,128,device='cuda').bfloat16();w=torch.randn(256,128,device='cuda').bfloat16()
    wq=quant_nvfp4_4over6(w,4,16);we=wq.float()-w.float()
    q,s=joint_quantize(x,wq=wq,weight_error=we)
    b=quant_nvfp4_4over6(x,4,16)
    loss=lambda a:float((a.double()@wq.double().T-x.double()@w.double().T).square().sum())
    assert abs(loss(q)-s['final_objective'])<1e-5*loss(q)
    assert loss(q)<=loss(b)*(1+1e-6)
    q0,_=joint_quantize(x,wq=wq,weight_error=torch.zeros_like(we));qa,_=online_quantize(x,w=wq)
    assert torch.equal(q0,qa)
    sketch=joint_sketch(wq,w)
    assert sum(t.numel()*t.element_size() for t in sketch)<=w.numel()/2/8
    a,b,d,c,e=sketch
    assert (c.square()<=d*e+1e-5).all()
    q,s=joint_quantize(x,sketch=sketch);err=q.float()-x.float();xx=x.float()
    actual=float((err@a.T+xx@b.T).square().sum()+(err.square()*d+2*err*xx*c+xx.square()*e).sum())
    assert abs(actual-s['final_objective'])<1e-5*actual
    print('PASS: actual joint W4A4 objective, monotonicity, zero-weight-error ablation, PSD diagonal remainder, metadata budget',flush=True)


if __name__=='__main__':main()

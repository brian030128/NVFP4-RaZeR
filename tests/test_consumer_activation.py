import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from quantize.consumer_activation import consumer_energy,quantize_input
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6


def main():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(981)
    x=torch.randn(1,19,128,generator=g).bfloat16()
    w=torch.randn(32,128,generator=g);w[:,::7]*=4
    d=consumer_energy(w)
    q,stats=quantize_input(x,d)
    b=quant_nvfp4_4over6(x,4,16)
    a=quant_mix_4_6(x,4,16,type_block=(16,64),clip='a1',elect='always')
    expected=b.clone()
    for row in (0,16):
        for col in (0,64):
            sl=(slice(None),slice(row,row+16),slice(col,col+64))
            e0=((a[sl].float()-x[sl].float()).square()*d[col:col+64]).sum()
            e2=((b[sl].float()-x[sl].float()).square()*d[col:col+64]).sum()
            if e0<e2:expected[sl]=a[sl]
    assert torch.equal(q,expected)
    assert stats['tiles']==4 and q.dtype==torch.bfloat16
    assert torch.equal(quantize_input(x,None)[0],quantize_input(x,torch.ones(128))[0])
    assert torch.equal(quantize_input(x,d)[0],quantize_input(x,5*d)[0])
    assert torch.allclose(consumer_energy(w),consumer_energy(2*w))
    assert not quantize_input(torch.zeros_like(x),d)[0].any()
    print('PASS: exact weighted tile election, padded rows, legal candidates, uniform equivalence, scale invariance, zero')


if __name__=='__main__':main()

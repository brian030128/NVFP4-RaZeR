"""Run on allocated compute, never the login node."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from quantize.spectral_approx import estimate, select_approx
from quantize.spectral_selector import candidates, spectral_squared


def main():
    torch.set_num_threads(2)
    g = torch.Generator().manual_seed(12)
    e = torch.randn(32,128,generator=g)
    result,v = estimate(e)
    exact = spectral_squared(e)
    assert abs(result['value']/exact-1)<1e-5
    # Vectorized finite directional scores agree with direct interventions.
    w = torch.randn(16,128,generator=g).bfloat16()
    b,a = candidates(w)
    delta = a.float()-b.float(); error=b.float()-w.float()
    _,v=estimate(error)
    dv=torch.einsum('rick,ck->rci',delta.reshape(2,8,2,64),v.reshape(2,64))
    scores=2*(dv*(error@v).reshape(2,1,8)).sum(-1)+dv.square().sum(-1)
    for row in range(2):
        for col in range(2):
            trial=error.clone()
            sl=(slice(row*8,row*8+8),slice(col*64,col*64+64))
            trial[sl]+=delta[sl]
            actual=(trial@v).square().sum()-(error@v).square().sum()
            assert torch.allclose(actual,scores[row,col],atol=2e-6,rtol=1e-4)
    mask,info=select_approx(w,b,a)
    hist=[x['value'] for x in info['history']]
    assert all(y<x for x,y in zip(hist,hist[1:]))
    assert torch.equal(mask,select_approx(w,b,a)[0])
    zero=torch.zeros(8,64)
    zm,zi=select_approx(zero,zero,zero)
    assert not zm.any() and zi['history'][0]['value']==0
    print('PASS: spectral estimate, finite directional scores, deterministic estimated descent, zero case')


if __name__=='__main__':
    main()

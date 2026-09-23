"""Observe unchanged primary per-sequence N8 scores; accumulate parent moments.

No candidate, loss, backward, score or scale definition is replaced. One hook
is inserted after the historical scorer already copies c8/k8 to CPU. This avoids
the old >=80 GiB full-score storage gate without pretending marginal SE suffices.
The adapter and original source hashes are saved; primary N8/N16 anchors must
be checked before any new parent map is used. Not a new selector.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
import torch

def main():
    import campaign.calibrate as historical
    from campaign import runtime
    path=Path(historical.__file__);original=path.read_text()
    needle='            c8c, k8c = c8.cpu(), k8.cpu()\n'
    assert original.count(needle)==1
    replacement=needle+'            stream_observer(n, c8c, k8c, shapes[n])\n'
    modified=original.replace(needle,replacement)
    out=runtime.out_dir('parent_moments');store={N:{} for N in [32,64,128,256]}
    def observer(name,ce,kl,shape):
        rows,cols=shape
        for N in store:
            assert rows%N==0 and cols%64==0,(name,shape,N)
            c=ce.double().reshape(rows//N,N//8,cols//64).sum(1).reshape(-1)
            k=kl.double().reshape(rows//N,N//8,cols//64).sum(1).reshape(-1)
            if name not in store[N]:
                store[N][name]=dict(n=0,ce_sum=torch.zeros_like(c),ce_sq=torch.zeros_like(c),kl_sum=torch.zeros_like(c),kl_sq=torch.zeros_like(c),cross=torch.zeros_like(c))
            st=store[N][name];st['n']+=1
            st['ce_sum']+=c;st['ce_sq']+=c*c;st['kl_sum']+=k;st['kl_sq']+=k*k;st['cross']+=c*k
    runtime.atomic_json(out/'adapter.json',dict(original_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrumented_source_sha256=hashlib.sha256(modified.encode()).hexdigest(),adapter_sha256=runtime.sha256_file(__file__),
        insertion='Observe existing c8c,k8c CPU score vectors; per-sequence FP64 nested row aggregation; no scientific expression replaced',
        storage='n/sum/sumsq/cross at N32/64/128/256; original N8/N16 persisted by historical main',status='running'))
    namespace=dict(__name__='__main__',__file__=str(path),stream_observer=observer)
    exec(compile(modified,str(path),'exec'),namespace)
    manifest={}
    for N,state in store.items():
        p=out/f'n{N}_moments.pt';assert not p.exists();torch.save(state,p)
        manifest[str(N)]=dict(path=str(p),sha256=runtime.sha256_file(p),sequences=sorted({s['n'] for s in state.values()}),modules=len(state))
    runtime.atomic_json(out/'manifest.json',manifest)

if __name__=='__main__':main()

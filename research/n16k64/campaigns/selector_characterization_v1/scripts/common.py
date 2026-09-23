"""CPU-only adapters; never change historical inputs or launch GPU jobs."""
import csv
import hashlib
import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[3]
PRIMARY = REPO / 'research_runs/mixfp4_n16k64_full_validation_20260911T065444Z'
FOLLOWUP = REPO / 'research_runs/mixfp4_n16k64_followup_20260917T090438Z'
MODELS = ['llama8b', 'qwen4b', 'mistral7b']
DRAWS = ['seed0', 'draw1', 'draw2', 'draw3', 'draw4']

def now(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def jsonout(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def checkpoint(task,status,evidence,blockers=(),next_action='Continue next pending task'):
    s=load(OUT/'TASK_STATUS.json')
    r=next(x for x in s['tasks'] if x['id']==task)
    r.update(status=status,evidence_paths=evidence,blockers=list(blockers))
    s['last_updated']=now();jsonout(OUT/'TASK_STATUS.json',s)
    (OUT/'NEXT_ACTION.md').write_text('# 下一步\n\n'+next_action+'\n')
def log(task,command,start,outputs,status='complete',error=None):
    row=dict(run_id=task+'-'+start,task=task,command=command,start_time=start,end_time=now(),
             exit_code=0 if status=='complete' else 1,status=status,source_sha256=sha(__file__),
             entrypoint_sha256=sha(sys.argv[0]) if Path(sys.argv[0]).is_file() else None,
             protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.yaml') if (OUT/'FROZEN_PROTOCOL.yaml').exists() else None,
             output_hashes={str(Path(p).relative_to(OUT)):sha(p) for p in outputs},error=error,gpu_hours=0)
    with (OUT/'RUN_LOG.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def mapread(p):
    b=Path(p).read_bytes();magic=b'MIXFP4MAP/1\n';assert b.startswith(magic)
    size=struct.unpack('<Q',b[len(magic):len(magic)+8])[0];off=len(magic)+8
    h=json.loads(b[off:off+size]);off+=size;masks={}
    for m in h['modules']:
        n=int(np.prod(m['grid_shape']));nb=(n+7)//8
        bits=np.unpackbits(np.frombuffer(b[off:off+nb],np.uint8),bitorder='little');off+=nb
        assert not bits[n:].any();a=bits[:n].astype(bool).reshape(m['grid_shape'])
        assert int(a.sum())==m['selected'];masks[m['name']]=a
    assert off==len(b)
    assert sum(int(x.sum()) for x in masks.values())==h['totals']['selected_tiles']
    return h,masks
def maskhash(masks):
    h=hashlib.sha256()
    for name,a in masks.items():h.update(name.encode());h.update(np.packbits(a.ravel(),bitorder='little').tobytes())
    return h.hexdigest()
def mapwrite(p,header,masks,policy,N=16):
    h=json.loads(json.dumps(header));h['type_block']=[N,64];h['policy']=policy
    h['protocol_id']='selector_characterization_v1';h['modules']=[]
    for m in header['modules']:
        a=masks[m['name']];h['modules'].append(dict(name=m['name'],weight_shape=m['weight_shape'],grid_shape=list(a.shape),selected=int(a.sum()),total=a.size))
    total=sum(a.size for a in masks.values());sel=sum(int(a.sum()) for a in masks.values())
    h['totals']=dict(total_tiles=total,selected_tiles=sel,total_weights=total*N*64,selected_weights=sel*N*64,selected_fraction=sel/total)
    hb=json.dumps(h,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    b=b'MIXFP4MAP/1\n'+struct.pack('<Q',len(hb))+hb+b''.join(np.packbits(a.ravel(),bitorder='little').tobytes() for a in masks.values())
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==b,'immutable map mismatch'
    else:p.write_bytes(b)
    return sha(p)
def moments(st,obj):
    n=int(st['n']);assert n>=2
    s=st[obj+'_sum'].numpy();q=st[obj+'_sq'].numpy()
    assert np.isfinite(s).all() and np.isfinite(q).all()
    return s/n,np.sqrt(np.maximum((q-s*s/n)/(n-1),0)/n)
def zscore(mu,se):
    assert np.isfinite(mu).all() and np.isfinite(se).all() and (se>=0).all()
    z=np.zeros_like(mu);np.divide(mu,se,out=z,where=se>0)
    z[(se==0)&(mu<0)]=-np.inf;z[(se==0)&(mu>0)]=np.inf
    return z
def topmask(score,k):
    assert not np.isnan(score).any();n=score.size;assert 0<=k<=n
    # Stable ordering implements canonical row-major identity tie-break.
    ids=np.argsort(score,kind='stable')[:k];a=np.zeros(n,bool);a[ids]=True
    tie=0 if k==0 else float(np.count_nonzero(score==score[ids[-1]])/n)
    return a,tie
def pair(a,b):
    ka=int(a.sum());kb=int(b.sum());i=int(np.count_nonzero(a&b));u=ka+kb-i
    return dict(K_a=ka,K_b=kb,intersection=i,jaccard=i/u if u else None,
                containment_a_to_b=i/ka if ka else None,containment_b_to_a=i/kb if kb else None,
                overlap_coefficient=i/min(ka,kb) if min(ka,kb) else None,empty_a=ka==0,empty_b=kb==0)
def null_pairs(universe,ka,kb,seed,reps=1000):
    # Hypergeometric draws are exactly the intersection law of independent
    # uniform fixed-size subsets per module, without materializing huge masks.
    rng=np.random.default_rng(seed);i=np.zeros(reps,dtype=np.int64)
    for u,a,b in zip(universe,ka,kb):
        if a and b:i+=rng.hypergeometric(int(a),int(u-a),int(b),size=reps)
    a=sum(ka);b=sum(kb);out={'expected_intersection_module':sum(x*y/u for x,y,u in zip(ka,kb,universe)),
                            'expected_intersection_global':a*b/sum(universe),'null_intersection_mean':float(i.mean())}
    for name,values in [('jaccard',i/(a+b-i) if a+b else None),('containment_a',i/a if a else None),('containment_b',i/b if b else None)]:
        for q,label in zip([.025,.5,.975],['lo','median','hi']):out['null_'+name+'_'+label]=float(np.quantile(values,q)) if values is not None else None
    return out

def aggregate_scores(x,rows,cols,N):
    assert N%8==0 and rows%N==0 and x.shape[1]==(rows//8)*(cols//64)
    return x.reshape(x.shape[0],rows//N,N//8,cols//64).sum(2).reshape(x.shape[0],-1)
def mean_se_array(x):
    return x.mean(0),x.std(0,ddof=1)/np.sqrt(len(x))
def regret(mu_children,mu_parent,se_parent,other_pass,k=3):
    fine=np.minimum(mu_children,0).sum(-1);coarse=np.minimum(mu_parent,0)
    own=coarse-fine;single=(mu_parent+k*se_parent<0)*mu_parent
    joint=(mu_parent+k*se_parent<0)*other_pass*mu_parent
    mass=np.abs(mu_children).sum(-1)
    c=np.full_like(mu_parent,np.nan);np.divide(2*own,mass,out=c,where=mass>0)
    assert (own>=-1e-10).all() and ((joint-single)>=-1e-10).all()
    return dict(ownership=own,threshold=single-coarse,veto=joint-single,total=joint-fine,cancellation=c)
def paired_bootstrap(loss,tokens,reps=2000,seed=20260921):
    loss=np.asarray(loss,dtype=np.float64);tokens=np.asarray(tokens,dtype=np.float64)
    assert loss.ndim==2 and loss.shape[1]==len(tokens) and (tokens>0).all()
    rng=np.random.default_rng(seed);counts=rng.multinomial(len(tokens),np.full(len(tokens),1/len(tokens)),size=reps)
    estimate=loss.sum(1)/tokens.sum();samples=(counts@loss.T)/(counts@tokens)[:,None]
    return estimate,samples

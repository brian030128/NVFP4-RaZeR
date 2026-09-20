"""Reuse the rejected confirmation as development; cache and audit last-MLP replay.

No fresh data, search, or PPL promotion. Eight fixed raw/arranged compositions.
"""
import json,os,time,tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from run_conditional_format import sha
from run_c4_frozen import digest_file
from run_math_code_calibration import load_model
from run_task_reorder_eval import load_layouts,raw256_masks,mix_coarse
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6
from quantize.task_reorder import original_order_weight_reference
from run_reorder_replay import paired_bounds

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def ce(logits,ids):
    lp=logits[:,:-1].float().reshape(-1,logits.shape[-1]).log_softmax(-1)
    return float(F.nll_loss(lp,ids[:,1:].reshape(-1)))

@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']));torch.backends.cuda.matmul.allow_tf32=False
    root=Path('/work/u4320956/task_reorder/transfer_20260920/llama8b')
    out=root/'suffix_diagnose';out.mkdir(exist_ok=False)
    prior=json.loads((root/'calibration/report.json').read_text())
    old=json.loads((root/'confirmation/report.json').read_text());assert old['status']=='complete' and old['passed'] is False
    records=torch.load(root/'confirmation/fresh.pt',map_location='cpu',weights_only=True)
    assert digest_file(root/'confirmation/fresh.pt')==old['fresh_sha256']
    panel,hashes=load_layouts([f'both={root / "compacted"}'],prior);layouts=panel['both']
    names=list(prior['reorder_modules']);assert [n.split('.')[-1] for n in names]==['gate_proj','up_proj','down_proj']
    raw=raw256_masks(root/'calibration',prior)
    plan=dict(status='frozen',development_only=True,new_fresh_required_before_ppl=True,
        records_from_rejected_confirmation=str(root/'confirmation/fresh.pt'),records_sha256=old['fresh_sha256'],
        compositions={str(i):[n for j,n in enumerate(names) if i&(1<<j)] for i in range(8)},
        prospective_screen='CE mean+1SE<0 vs raw on64 reused documents and nonpositive domain CE means; choose strongest pooled mean among eligible combinations',
        source_sha256=digest_file(__file__),layout_sha256=hashes)
    write(out/'plan.json',plan)
    report=dict(status='running',job_id=os.environ['SLURM_JOB_ID'],plan=plan,completed_sequences=0,
        metrics={str(i):[] for i in range(8)},audits=dict(raw_full=0,raw_suffix=0,both_suffix=0))
    write(out/'report.json',report)
    model,modules=load_model(prior,False);model.set_attn_implementation('sdpa')
    prepared={};pristine={}
    for name,module in modules.items():
        assert sha(module.weight)==prior['matrices'][name]['source_sha256'],name
        if name in layouts:pristine[name]=module.weight.cpu().clone()
        base=quant_nvfp4_4over6(module.weight,4,16)
        if name in layouts or raw[name].any():
            alt=quant_mix_4_6(module.weight,4,16,type_block=(8,64),clip='a1',elect='always')
            mixed=mix_coarse(base,alt,raw[name])
            if name in layouts:
                prepared[name]=dict(base=base,alt=alt,raw=mixed,both=original_order_weight_reference(base,alt,layouts[name]))
            base=mixed
        module.weight.copy_(base)
    last=model.model.layers[-1];head=model.get_output_embeddings();norm=model.model.norm
    handles=[m.register_forward_pre_hook(lambda module,inputs:(quant_nvfp4_4over6(inputs[0],4,16),*inputs[1:])) for m in modules.values()]
    captured={};capture=True
    def remember(key):
        def hook(module,inputs):
            if capture:captured[key]=inputs[0].detach().clone()
        return hook
    h1=last.post_attention_layernorm.register_forward_pre_hook(remember('residual'))
    h2=last.mlp.register_forward_pre_hook(remember('x'))
    cache=Path(tempfile.mkdtemp(prefix='llama_suffix_',dir=os.environ['TMPDIR']))
    report['cache']=str(cache);write(out/'report.json',report)
    captures=[];start=time.monotonic()
    for index,row in enumerate(records):
        ids=row['ids'].cuda()
        for name in names:modules[name].weight.copy_(prepared[name]['raw'])
        capture=True;logits=model(input_ids=ids,use_cache=False).logits;raw_ce=ce(logits,ids);del logits
        assert raw_ce==old['metrics']['raw256']['ce'][index],('raw full replay',index,raw_ce,old['metrics']['raw256']['ce'][index])
        report['audits']['raw_full']+=1;capture=False
        captures.append(dict(x=captured['x'].cpu(),residual=captured['residual'].cpu(),**row))
        for mask in range(8):
            for j,name in enumerate(names):modules[name].weight.copy_(prepared[name]['both' if mask&(1<<j) else 'raw'])
            logits=head(norm(captured['residual']+last.mlp(captured['x'])))
            value=ce(logits,ids);del logits
            if mask in (0,7):
                expected=old['metrics']['raw256' if mask==0 else 'candidate']['ce'][index]
                assert value==expected,('suffix replay',mask,index,value,expected)
                report['audits']['raw_suffix' if mask==0 else 'both_suffix']+=1
            report['metrics'][str(mask)].append(value)
        report['completed_sequences']=index+1;write(out/'report.json',report)
        if (index+1)%8==0:print('DIAGNOSE',index+1,'seconds',round(time.monotonic()-start,2),flush=True)
    h1.remove();h2.remove()
    for h in handles:h.remove()
    torch.save(captures,cache/'captures.pt')
    state=dict(pristine=pristine,weights={n:{k:v.cpu() for k,v in d.items()} for n,d in prepared.items()},
        head=head.weight.cpu(),norm=norm.weight.cpu(),config=model.config.to_dict(),names=names)
    torch.save(state,cache/'state.pt')
    report['paired']={};eligible=[]
    for mask in range(1,8):
        domains={}
        for domain in ('all','math','code'):
            idx=[i for i,r in enumerate(records) if domain=='all' or r['source']==domain]
            domains[domain]=paired_bounds([report['metrics'][str(mask)][i] for i in idx],[report['metrics']['0'][i] for i in idx])
        report['paired'][str(mask)]=domains
        if domains['all']['mean']+domains['all']['se']<0 and all(domains[d]['mean']<=0 for d in ('math','code')):eligible.append(mask)
    report.update(status='complete',eligible=eligible,selected=min(eligible,key=lambda m:report['paired'][str(m)]['all']['mean']) if eligible else None,
        cache_sha256={name:digest_file(cache/name) for name in ('captures.pt','state.pt')})
    write(out/'report.json',report);print('RESULT',json.dumps({k:report[k] for k in ('audits','paired','eligible','selected','cache')}),flush=True)
if __name__=='__main__':main()

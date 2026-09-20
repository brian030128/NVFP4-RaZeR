"""Frozen last-MLP arranging transfer; one fresh CE-primary gate before PPL."""
import argparse, copy, json, math, os, shutil, tempfile
from dataclasses import asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_conditional_format import sha, save
from run_c4_frozen import digest_file
from run_math_code_calibration import load_model
from run_task_reorder import run as search
from run_task_reorder_eval import raw256_masks, mix_coarse, load_layouts
from run_fine_row_research import fresh_data, sequence_metadata
from run_reorder_replay import paired_bounds
from quantize.task_reorder import SearchConfig, original_order_weight_reference
from quantize.compact_both import compact_both
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
from scripts.ce_confirmation_gate import passes


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


@torch.no_grad()
def run(root):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    torch.backends.cuda.matmul.allow_tf32=False
    prior=json.loads((root/'calibration/report.json').read_text())
    assert prior['status']=='complete' and prior['source_weights_verified']
    config=SearchConfig(tile_rows=256,tile_cols=64,atom_rows=1,atom_cols=16,k=3.,seed=0,
        starts=4,rounds=6,swap_samples=4096,swap_passes=2,scratch_mb=32,axes='both')
    summary={}
    for name,directory in prior['reorder_modules'].items():
        index=Path(directory).name;dest=root/'search/both'/index
        if (dest/'report.json').exists():
            record=json.loads((dest/'report.json').read_text())
            assert record['status']=='complete' and record['config']==asdict(config)
        else:record=search(Path(directory),dest,config)
        summary[name]=record
    layouts=root/'compacted';layouts.mkdir(exist_ok=True)
    for index in sorted((root/'search/both').iterdir()):
        old=torch.load(index/'layout.pt',map_location='cpu',weights_only=True)
        dest=layouts/index.name;dest.mkdir(exist_ok=True)
        if (dest/'layout.pt').exists():
            value=torch.load(dest/'layout.pt',map_location='cpu',weights_only=True)
            assert value['name']==old['name'] and value['config']==old['config']
        else:
            value=compact_both(old);torch.save(value,dest/'layout.pt')
        if not (dest/'report.json').exists():
            write(dest/'report.json',dict(status='complete',module=value['name'],
                elected_tiles=int(value['mask'].sum()),source_layout=str(index/'layout.pt'),
                source_layout_sha256=digest_file(index/'layout.pt'),
                layout_sha256=digest_file(dest/'layout.pt'),
                column_compaction=value.get('column_compaction'),row_compaction=value.get('row_compaction'),
                actual_quantized_weight_audit='performed during confirmation before any candidate loss'))
    out=root/'confirmation';out.mkdir(exist_ok=True)
    assert not any(out.iterdir()), 'Do not repeat a confirmation once its plan/data/report exists'
    panels,hashes=load_layouts([f'candidate={layouts}'],prior)
    controls=panels['candidate'];raw=raw256_masks(root/'calibration',prior)
    total=sum(int(mask.sum()) for name,mask in raw.items() if name not in controls)+sum(int(v['mask'].sum()) for v in controls.values())
    files=('run_llama_arrange_transfer.py','quantize/task_reorder.py','quantize/compact_both.py',
           'quantize/compact_rows.py','quantize/quantizer.py','run_fine_row_research.py','scripts/ce_confirmation_gate.py')
    plan=dict(protocol_family='ce_primary_after_positive_control',candidate='llama last-MLP both-axis transfer',
        criterion='Pooled CE mean+2SE<0 vs raw256 and identity; domain CE means <=0 vs raw; KL diagnostic',
        source_sha256={p:digest_file(p) for p in files},layout_sha256=hashes,
        calibration_report_sha256=digest_file(root/'calibration/report.json'),
        expected_modules=list(controls),expected_total_tiles=total,search=asdict(config))
    write(out/'plan.json',plan)
    write(root/'search_summary.json',summary)
    published=json.loads(Path('results/kse_paper/job_336566/llama8b/report.json').read_text())
    excluded=[r['document_sha256'] for r in published['data']['c4_paper']['documents']]
    exclusion_path=Path('results/task_reorder/transfer_20260920/exclude_manifests.json')
    exclusions=json.loads(exclusion_path.read_text())
    plan['exclude_manifest_list_sha256']=digest_file(exclusion_path)
    plan['exclude_manifests']={path:digest_file(path) for path in exclusions}
    old_tokens=set(sequence_metadata(prior)[0])
    for path in exclusions:
        records=json.loads(Path(path).read_text())['records']
        excluded.extend(r['document_sha256'] for r in records)
        old_tokens.update(r['token_sha256'] for r in records)
    # The accepted arrangement was developed on Qwen calibration documents too.
    qprior=json.loads(Path('/work/u4320956/task_reorder/pilot_20260919/qwen27b/calibration/report.json').read_text())
    excluded.extend(d['document_sha256'] for v in qprior['fit'].values() for d in v['documents'])
    old_tokens.update(sequence_metadata(qprior)[0])
    plan['excluded_document_count']=len(set(excluded))
    plan['excluded_token_count']=len(old_tokens)
    write(out/'plan.json',plan)
    tokenizer=AutoTokenizer.from_pretrained(prior['source'],revision=prior['revision'])
    fresh=fresh_data(tokenizer,prior,excluded)
    assert len(fresh)==64 and len({r['document_sha256'] for r in fresh})==64
    assert not {r['document_sha256'] for r in fresh}&set(excluded)
    assert not {r['token_sha256'] for r in fresh}&old_tokens
    torch.save(fresh,out/'fresh.pt')
    write(out/'fresh_manifest.json',dict(records=[{k:v for k,v in r.items() if k!='ids'} for r in fresh],
        windows=64,window_tokens=512,seed=20260919,excluded_calibration_and_published_c4=True))
    report=dict(status='running',job_id=os.environ['SLURM_JOB_ID'],plan_sha256=digest_file(out/'plan.json'),
        layout_sha256=hashes,fresh_sha256=digest_file(out/'fresh.pt'),fresh_manifest_sha256=digest_file(out/'fresh_manifest.json'),
        fixed_candidate_before_fresh_data=True,new_windows=True,candidates=1,completed_sequences=0,
        total_e0m3_tiles=total,raw_tiles=sum(int(v.sum()) for v in raw.values()),
        sequence_sources=[r['source'] for r in fresh],metrics={p:dict(ce=[],kl=[]) for p in ('raw256','identity','candidate')})
    save(out,report)
    model,modules=load_model(prior,False)
    # SDPA matches the accepted PPL reference; fresh comparisons share arithmetic.
    model.set_attn_implementation('sdpa')
    for name,module in modules.items():assert sha(module.weight)==prior['matrices'][name]['source_sha256'],name
    with tempfile.TemporaryDirectory(prefix='llama_confirm_',dir=os.environ['TMPDIR']) as tmp:
        tmp=Path(tmp)
        assert shutil.disk_usage(tmp).free>64*511*model.config.vocab_size*4+2*1024**3
        for i,row in enumerate(fresh):
            logits=model(input_ids=row['ids'].cuda(),use_cache=False).logits
            lp=logits[:,:-1].float().reshape(-1,logits.shape[-1]).log_softmax(-1)
            torch.save(lp.cpu(),tmp/f'{i}.pt');del logits,lp
            if (i+1)%16==0:print('TEACHER',i+1,flush=True)
        prepared={p:{} for p in report['metrics']}
        compact_verified={}
        for name,module in modules.items():
            base=quant_nvfp4_4over6(module.weight,4,16)
            if name in controls or raw[name].any():
                alt=quant_mix_4_6(module.weight,4,16,type_block=(8,64),clip='a1',elect='always')
                mixed=mix_coarse(base,alt,raw[name])
                if name in controls:
                    prepared['raw256'][name]=mixed
                    prepared['identity'][name]=original_order_weight_reference(base,alt,controls[name],'identity')
                    prepared['candidate'][name]=original_order_weight_reference(base,alt,controls[name])
                    original=torch.load(root/'search/both'/Path(prior['reorder_modules'][name]).name/'layout.pt',map_location='cpu',weights_only=True)
                    expected=original_order_weight_reference(base,alt,original)
                    assert torch.equal(expected,prepared['candidate'][name]);compact_verified[name]=True
                    del expected,original
                base=mixed;del mixed,alt
            module.weight.copy_(base)
        del base
        report['bitwise_compaction_verified']=compact_verified
        handles=[m.register_forward_pre_hook(lambda module,inputs:(quant_nvfp4_4over6(inputs[0],4,16),*inputs[1:])) for m in modules.values()]
        for i,row in enumerate(fresh):
            ids=row['ids'].cuda();teacher=torch.load(tmp/f'{i}.pt',map_location='cpu',weights_only=True).cuda()
            for label,weights in prepared.items():
                for name,weight in weights.items():modules[name].weight.copy_(weight)
                logits=model(input_ids=ids,use_cache=False).logits
                lp=logits[:,:-1].float().reshape(-1,logits.shape[-1]).log_softmax(-1)
                values=dict(ce=float(F.nll_loss(lp,ids[:,1:].reshape(-1))),kl=float(F.kl_div(lp,teacher,reduction='batchmean',log_target=True)))
                assert all(math.isfinite(v) for v in values.values())
                for key,v in values.items():report['metrics'][label][key].append(v)
                del logits,lp
            del teacher;(tmp/f'{i}.pt').unlink()
            report['completed_sequences']=i+1;save(out,report)
            if (i+1)%8==0:print('CONFIRM',i+1,flush=True)
        for h in handles:h.remove()
    report['paired']={}
    for ref in ('raw256','identity'):
        report['paired'][ref]={}
        for domain in ('all','math','code'):
            idx=[i for i,r in enumerate(fresh) if domain=='all' or r['source']==domain]
            report['paired'][ref][domain]={key:paired_bounds([report['metrics']['candidate'][key][i] for i in idx],[report['metrics'][ref][key][i] for i in idx]) for key in ('ce','kl')}
    report['passed']=passes(report['paired']);report['status']='complete'
    for path,h in plan['source_sha256'].items():assert digest_file(path)==h
    for path,h in hashes.items():assert digest_file(path)==h
    for path,h in plan['exclude_manifests'].items():assert digest_file(path)==h
    save(out,report);print('RESULT',json.dumps(dict(passed=report['passed'],tiles=total,paired=report['paired'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)

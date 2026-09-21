#!/usr/bin/env python3
"""Cheap read-only source audit and paired-array arithmetic; no GPU/bootstrap.

Only review metadata is written. The optional workspace supplies local immutable
maps for hash/header verification; public arrays suffice for policy accounting.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
s=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def header(p):
 data=p.read_bytes();magic=b'MIXFP4MAP/1\n';assert data.startswith(magic)
 n=struct.unpack('<Q',data[len(magic):len(magic)+8])[0]
 return json.loads(data[len(magic)+8:len(magic)+8+n])
def payload_hash(p):
 data=p.read_bytes();magic=b'MIXFP4MAP/1\n';n=struct.unpack('<Q',data[len(magic):len(magic)+8])[0]
 h=json.loads(data[len(magic)+8:len(magic)+8+n]);off=len(magic)+8+n;digest=hashlib.sha256()
 for m in h['modules']:
  count=m['grid_shape'][0]*m['grid_shape'][1];size=(count+7)//8;chunk=data[off:off+size]
  assert len(chunk)==size
  bits=np.unpackbits(np.frombuffer(chunk,dtype=np.uint8),bitorder='little')
  assert int(bits[:count].sum())==m['selected'] and not bits[count:].any()
  digest.update(m['name'].encode());digest.update(chunk);off+=size
 assert off==len(data)
 return digest.hexdigest()
inventory=load(s/'ARTIFACT_INVENTORY.json')
roots={x['public_snapshot'].split('/')[1]:a.workspace/'research_runs'/x['id'] for x in inventory['campaigns']}
errors=[];report={'checked_utc':datetime.now(timezone.utc).isoformat(),'new_gpu_jobs':0,'bootstrap_rerun':False}
seals=[];seal_entries={}
for row in inventory['campaigns']:
 key=row['public_snapshot'].split('/')[1]
 rel={'primary':'runs/V83_validate_artifacts_attempt7/artifact_validation/ARTIFACT_MANIFEST.sha256',
      'ppl_improvement':'runs/P80_final_analysis_attempt1/final/ARTIFACT_MANIFEST.sha256'}.get(key,'ARTIFACT_MANIFEST.sha256')
 f=roots[key]/rel;digest=sha(f)
 if digest!=row['artifact_manifest_sha256']:errors.append('campaign seal '+key)
 seals.append({'campaign':key,'manifest':rel,'sha256':digest,'matched':digest==row['artifact_manifest_sha256']})
 seal_entries[key]={line.split(maxsplit=1)[1].strip().lstrip('*'):line.split(maxsplit=1)[0] for line in f.read_text().splitlines() if line.strip()}
report['campaign_manifest_seals']=seals
# Verify original and public lineage, including additions with direct logical paths.
from prepare_public_snapshot import origin_for
index=load(s/'SNAPSHOT_SOURCE_INDEX.json');sources=[]
for row in index['entries']:
 logical=row['logical_source'];prefix,tail=logical.split('://',1);key,rel=tail.split('/',1)
 if prefix=='campaign-artifact':src=roots[key]/rel
 else:_,src=origin_for(Path(row['public_path']),roots)
 ok=src.is_file() and sha(src)==row['source_sha256'] and sha(s/row['public_path'])==row['public_sha256']
 sources.append({'public_path':row['public_path'],'matched':ok})
 if not ok:errors.append('lineage '+row['public_path'])
 if key in ['mechanism','followup','boundary']:
  source_rel=src.relative_to(roots[key]).as_posix()
  expected=seal_entries[key].get(source_rel)
  if expected is not None and expected!=row['source_sha256']:errors.append('sealed member '+source_rel)
report['source_lineage']={'checked':len(sources),'mismatches':[x for x in sources if not x['matched']]}
# Confirm the two earlier packages are exact subsets of the final archived package.
versions={}
for key in ['mechanism','followup']:
 src=roots[key]/'source/NVFP4-RaZeR-main/campaign';dst=roots['boundary']/'source/NVFP4-RaZeR-main/campaign'
 files=[x for x in src.rglob('*.py') if '__pycache__' not in x.parts]
 bad=[x.relative_to(src).as_posix() for x in files if not (dst/x.relative_to(src)).is_file() or sha(x)!=sha(dst/x.relative_to(src))]
 versions[key]={'files':len(files),'different_or_missing':bad}
 errors.extend('source subset '+x for x in bad)
report['historical_subset_equivalence']=versions
# Decode headers and match every frozen map hash. No model weights loaded.
maps=load(roots['boundary']/'MAP_MANIFEST.json');counts=Counter();maprows=[]
for r in maps:
 path=Path(r['path'])
 if not path.is_file():path=roots['boundary']/r['path'].split(roots['boundary'].name+'/',1)[1]
 h=header(path);assert sha(path)==r['sha256'];assert h['type_block']==[16,64]
 actual_payload=payload_hash(path);assert actual_payload==r['mask_payload_sha256']
 for m in h['modules']:
  assert m['grid_shape']==[m['weight_shape'][0]//16,m['weight_shape'][1]//64]
 counts[r['model']]+=1
 maprows.append({'model':r['model'],'policy':r['policy'],'sha256':r['sha256'],'payload_sha256':actual_payload,'selected':h['totals']['selected_tiles']})
report['boundary_map_counts']=dict(counts);report['boundary_map_hashes_checked']=len(maps)
report['p100_distinct_payload_counts']={model:len({r['payload_sha256'] for r in maprows if r['model']==model and r['policy'].endswith('_p100')}) for model in counts}
assert all(n==4 for n in report['p100_distinct_payload_counts'].values())
pools=load(s/'campaigns/boundary/CORRUPTION_POOL_DEFINITIONS.json')
pool_checks={}
for model,row in pools['models'].items():
 sets=[{(name,i) for name,values in pool['tiles'].items() for i in values} for pool in row['pools']]
 assert len(sets)==4 and all(len(x)==row['selected_tiles'] for x in sets)
 assert not row['shortages'] and row['corruption_eligible_coverage']==1
 assert all(not (sets[i]&sets[j]) for i in range(4) for j in range(i+1,4))
 for pool in range(1,5):
  prev_r={};prev_a={}
  for m in sorted([m for m in row['near_boundary_maps'] if m['pool']==pool],key=lambda m:m['target_p']):
   for name,vals in m['removed'].items():
    assert set(prev_r.get(name,[]))<=set(vals)
    assert len(vals)==len(m['added'][name])
    assert set(prev_a.get(name,[]))<=set(m['added'][name])
   prev_r=m['removed'];prev_a=m['added']
 pool_checks[model]={'disjoint':True,'pool_count':4,'tiles_per_pool':row['selected_tiles'],'coverage':1.0,'nested_quotas':True}
report['corruption_pool_checks_recomputed']=pool_checks
# Compact arrays are small enough to include verbatim and recompute means cheaply.
panels=[]
for f in sorted((s/'campaigns/boundary/arrays').glob('*.npz')):
 with np.load(f,allow_pickle=False) as z:
  names=z['policies'].tolist();tokens=z['cluster_tokens'];nll=z['cluster_nll_sum'].sum(1)/tokens.sum()
  assert len(names)==58 and np.isfinite(nll).all()
  assert len(set(z['cluster_ids'].tolist()))==len(tokens)
  orig=[i for i,n in enumerate(names) if n=='corruption_p000_anchor'];assert len(orig)==1,names
  p1=[i for i,n in enumerate(names) if n.startswith('corruption_near_') and n.endswith('_p100')];assert len(p1)==4
  delta=float(nll[p1].mean()-nll[orig[0]])
  panels.append({'array':f.relative_to(s).as_posix(),'sha256':sha(f),'policies':len(names),'clusters':len(tokens),'p100_minus_p0':delta})
report['boundary_arrays']=panels
conf=load(s/'campaigns/primary/analysis_ppl/CONFIRMATORY_PPL.json')
n16=sum(v['contrasts']['n16_k3-four_over_six'][d]['estimate'] for v in conf['models'].values() for d in ['wiki','c4'])
n8=sum(v['contrasts']['n8_k3-four_over_six'][d]['estimate'] for v in conf['models'].values() for d in ['wiki','c4'])
report['retained_gain']={'numerator_sum_delta_nll_n16':n16,'denominator_sum_delta_nll_n8':n8,'ratio':n16/n8,'endpoints':6,'meaning':'ratio of summed log-PPL gains, not selected-tile fraction'}
assert abs(n16/n8-0.7798406578406878)<1e-14
# Resolve all five draws by evaluated conjunction SHA, never choose the newest attempt blindly.
stability=load(s/'campaigns/primary/analysis_ppl/CALIBRATION_SEED_STABILITY_PPL.json')
legacy=load(s/'campaigns/primary/analysis_ppl/LEGACY_PANEL_PPL.json')
acc=load(s/'campaigns/primary/analysis_accuracy/CALIBRATION_SEED_STABILITY_ACCURACY.json')
draws=[]
for model in ['llama8b','qwen4b','mistral7b']:
 for draw in ['seed0','draw1','draw2','draw3','draw4']:
  panel=(conf if model=='mistral7b' else legacy)['models'][model] if draw=='seed0' else stability['models'][model]
  policy='n16_k3'+('' if draw=='seed0' else '_'+draw)
  expected=panel['map_sha256'][policy]
  candidates=list((roots['primary']/'runs').glob(f'*calib_{model}_{draw}_attempt*/maps/{model}_{draw}_n16_k3.mixfp4map'))
  good=[f for f in candidates if sha(f)==expected];assert len(good)==1,(model,draw,len(good))
  f=good[0];run=f.parent.parent;cm=run/'calibration/calibration_manifest.json'
  natural=[]
  for suffix in ['', '_ce_only','_kl_only']:
   mf=f.with_name(f.stem+suffix+f.suffix);h=header(mf)
   natural.append({'path':mf.relative_to(roots['primary']).as_posix(),'sha256':sha(mf),'selected_tiles':h['totals']['selected_tiles'],'rule':h['policy']})
  pair=panel['contrasts'][policy+'-four_over_six']
  macro=acc['models'][model][policy]['macro']
  draws.append({'model':model,'draw':draw,'calibration_manifest':cm.relative_to(roots['primary']).as_posix(),'calibration_manifest_sha256':sha(cm),
    'conjunction_map_sha256':expected,'natural_maps':natural,'delta_nll':{d:pair[d]['estimate'] for d in ['c4','wiki']},
    'ppl':{d:pair[d]['ppl_a'] for d in ['c4','wiki']},'accuracy_macro_diff':macro['diff'],'accuracy_tasks':macro['tasks'],
    'ppl_bootstrap_replicates':{d:pair[d].get('B') for d in ['c4','wiki']}})
report['five_draw_inputs']=draws
report['five_draw_favorable_ppl_points']=sum(r['delta_nll'][d]<0 for r in draws for d in ['c4','wiki'])
report['five_draw_accuracy_macro_range_by_model']={m:[min(r['accuracy_macro_diff'] for r in draws if r['model']==m),max(r['accuracy_macro_diff'] for r in draws if r['model']==m)] for m in ['llama8b','qwen4b','mistral7b']}
questions=[
 ('primary quality / N8 non-inferiority','primary','PROTOCOL_FREEZE.json','decision_gate.json','strong pass (quality component only)','six fixed confirmatory endpoints; not N16 superiority'),
 ('post-hoc k2 extension','ppl_improvement','PROTOCOL_EXTENSION.json','CLAIM_REGISTER.json','mixed; post-hoc support with failed/stopped promotion gates','not a replacement primary or universal downstream claim'),
 ('individual effect / group ranking','mechanism','MECHANISM_PROTOCOL.json','MECHANISM_VERDICT.md','ranking supported; strict veto/interaction not supported','no individual-tile calibration'),
 ('dose response / objective ablation','followup','FOLLOWUP_PROTOCOL.json','FOLLOWUP_VERDICT.md','dose response supported; no universal conjunction superiority','seed0 only, not across-draw objective robustness'),
 ('boundary / full-map corruption','boundary','PROTOCOL.json','BOUNDARY_CORRUPTION_VERDICT.md','power_limited_support','late retries are protocol-deviating; strict admissibility unresolved'),
 ('five-draw PPL / four-task accuracy','primary','PROTOCOL_FREEZE.json','analysis_accuracy/CALIBRATION_SEED_STABILITY_ACCURACY.json','evaluations complete; full multi-map frequency analysis pending','30 favorable PPL point estimates, not universally favorable accuracy'),
 ('calibration size / composition','primary','PROTOCOL_FREEZE.json','analysis_ppl/CALIBRATION_SIZE_DOMAIN_PPL.json','existing one-draw controls complete','not the proposed five-block fixed-budget composition experiment'),
 ('GSM8K / PG19 supplementary','primary','PROTOCOL_FREEZE.json','analysis_ppl/LONG_CONTEXT.json','primary evaluations complete; secondary evidence','limited book clusters; extension downstream promotion stopped'),
]
report['research_status']=[]
for question,key,proto,result,label,limit in questions:
 pp=s/'campaigns'/key/proto;rr=s/'campaigns'/key/result
 report['research_status'].append({'question':question,'campaign':key,'execution':'completed for existing matrix',
  'analysis_status':'existing terminal analysis complete; proposed extensions are separate',
  'terminal_or_review_classification':label,'unsupported_or_unresolved':limit,
  'protocol':str(pp.relative_to(s)),'public_protocol_sha256':sha(pp),'outcome':str(rr.relative_to(s)),
  'public_outcome_sha256':sha(rr),'code_version':'per-file original/public hashes in SNAPSHOT_SOURCE_INDEX.json; frozen protocol and run source manifests',
  'map_versions':'MAP_MANIFEST.json where available; primary actual map hashes in five_draw_inputs and compact result JSON',
  'missing_public_artifacts':'large raw maps/moments/evaluation arrays (except six boundary paired arrays)',
  'new_gpu_needed_for_completed_question':False,'verification_level':'compact evidence/source hash audit; not a new full raw inference campaign'})
report['errors']=errors;report['passed']=not errors
(s/'validation/CURRENT_EVIDENCE_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['source_lineage','historical_subset_equivalence','boundary_map_counts','boundary_map_hashes_checked','boundary_arrays','retained_gain','five_draw_favorable_ppl_points','five_draw_accuracy_macro_range_by_model','errors','passed']},indent=2))
raise SystemExit(0 if report['passed'] else 1)

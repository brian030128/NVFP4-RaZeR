"""Actual joint CE optimization on 192 explicitly reused development documents."""
import copy,json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP,LlamaRMSNorm
from run_c4_frozen import digest_file
from run_reorder_replay import paired_bounds
from quantize.task_reorder import original_order_weight_reference
from quantize.quantizer import quant_nvfp4_4over6
from scripts.ce_confirmation_gate import passes

def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
@torch.no_grad()
def main():
 assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 root=Path('/work/u4320956/task_reorder/transfer_20260920/llama8b');out=root/'joint192';out.mkdir(exist_ok=False)
 parent=json.loads((root/'suffix192/report.json').read_text());assert parent['status']=='complete'
 cache=Path(parent['cache']);assert digest_file(cache/'captures.pt')==parent['cache_sha256'];assert digest_file(parent['state_path'])==parent['state_sha256']
 records=torch.load(cache/'captures.pt',map_location='cpu',weights_only=True);state=torch.load(parent['state_path'],map_location='cpu',weights_only=True)
 names=state['names'];cfg=LlamaConfig(**state['config']);mlp=LlamaMLP(cfg).cuda().bfloat16().eval();norm=LlamaRMSNorm(cfg.hidden_size,eps=cfg.rms_norm_eps).cuda().bfloat16().eval();norm.weight.copy_(state['norm']);head=state['head'].cuda()
 weights={n:{k:v.cuda() for k,v in state['weights'][n].items()} for n in names};del state
 hooks=[m.register_forward_pre_hook(lambda mod,inputs:(quant_nvfp4_4over6(inputs[0],4,16),)) for m in (mlp.gate_proj,mlp.up_proj,mlp.down_proj)]
 # Match the cached log_softmax+NLL implementation exactly.
 def losses():
  values=[]
  for r in records:
   ids=r['ids'].cuda();logits=F.linear(norm(r['residual'].cuda()+mlp(r['x'].cuda())),head);lp=logits[:,:-1].float().reshape(-1,logits.shape[-1]).log_softmax(-1)
   values.append(float(F.nll_loss(lp,ids[:,1:].reshape(-1))))
  return values
 for n in names:getattr(mlp,n.split('.')[-1]).weight.copy_(weights[n]['raw'])
 assert losses()==parent['raw_ce'],'192 raw suffix audit'
 layouts=[torch.load(root/f'tile_refine/layouts/{j:03d}/layout.pt',map_location='cpu',weights_only=True) for j in range(3)]
 elected=[torch.load(root/f'compacted/{j:03d}/layout.pt',map_location='cpu',weights_only=True)['mask'].nonzero().tolist() for j in range(3)]
 assert [len(c) for c in elected]==[4,7,39]
 initial=[l['mask'].clone() for l in layouts]
 def install(j,l,policy='reordered'):
  getattr(mlp,names[j].split('.')[-1]).weight.copy_(original_order_weight_reference(weights[names[j]]['base'],weights[names[j]]['alt'],l,policy))
 for j,l in enumerate(layouts):install(j,l,'identity')
 identity=losses()
 def paired(v,ref):
  return {d:{'ce':paired_bounds([v[i] for i,r in enumerate(records) if d=='all' or r['source']==d],[ref[i] for i,r in enumerate(records) if d=='all' or r['source']==d])} for d in ('all','math','code')}
 def objective(v):
  q=paired(v,parent['raw_ce']);scores=[q['all']['ce']['mean']+2*q['all']['ce']['se']]+[q[d]['ce']['mean']+q[d]['ce']['se'] for d in ('math','code')]
  for k in range(3):
   z=paired_bounds(v[k*64:(k+1)*64],parent['raw_ce'][k*64:(k+1)*64]);scores.append(z['mean'])
  return max(scores)
 plan=dict(development_only=True,documents=192,source_sha256=digest_file(__file__),cache_report_sha256=digest_file(root/'suffix192/report.json'),algorithm='At most4 exact single-tile flips among all50 originally k3-elected gate/up/down tiles; minmax pooled CE2SE,domain CE1SE,and each observed64set mean vsraw; actual map change and full original raw+identity CE-primary screen before NEWfresh64.',no_new_validation_data=True,max_seconds=420)
 write(out/'plan.json',plan)
 for j,l in enumerate(layouts):install(j,l)
 current_values=losses();current=objective(current_values);trace=[];start=time.monotonic()
 report=dict(status='running',job_id=os.environ['SLURM_JOB_ID'],plan=plan,audits=192,initial_ce=current_values,identity_ce=identity,trace=trace)
 write(out/'report.json',report)
 for step in range(4):
  options=[]
  for j,coords in enumerate(elected):
   for row,col in coords:
    l=copy.deepcopy(layouts[j]);l['mask'][row,col]=~l['mask'][row,col];install(j,l);v=losses();options.append((objective(v),j,row,col,v));install(j,layouts[j])
   print('SCAN',step+1,names[j],round(time.monotonic()-start,1),flush=True)
  score,j,row,col,values=min(options,key=lambda x:x[0])
  if score>=current:break
  layouts[j]['mask'][row,col]=~layouts[j]['mask'][row,col];install(j,layouts[j]);current=score;current_values=values
  trace.append(dict(step=step,module=names[j],row=row,col=col,enabled=bool(layouts[j]['mask'][row,col]),objective=score,ce=values));write(out/'report.json',report)
  print('STEP',step+1,names[j],row,col,'objective',score,flush=True)
  if time.monotonic()-start>plan['max_seconds']:break
 frozen=out/'layouts';frozen.mkdir()
 for j,l in enumerate(layouts):
  p=frozen/f'{j:03d}';p.mkdir();torch.save(l,p/'layout.pt');write(p/'report.json',dict(status='complete',module=l['name']))
 final=losses();assert final==current_values
 comparisons={ref:paired(final,v) for ref,v in [('raw256',parent['raw_ce']),('identity',identity)]}
 changed=any(not torch.equal(l['mask'],v) for l,v in zip(layouts,initial))
 report.update(status='complete',map_changed=changed,final_ce=final,paired=comparisons,passed_development=changed and passes(comparisons),tiles=[int(l['mask'].sum()) for l in layouts])
 write(out/'report.json',report);print('RESULT',json.dumps({k:report[k] for k in ['map_changed','paired','passed_development','tiles']}),flush=True)
if __name__=='__main__':main()

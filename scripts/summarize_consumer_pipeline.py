"""Summarize measured per-projection GEMM+consumer pipeline timings."""
from pathlib import Path
import json,statistics,sys,hashlib
p=Path(sys.argv[1]);out=[]
for projection in ('up_proj','down_proj'):
 for m in (1,128,512,2048):
  f=p/f'{projection}_m{m}.txt';lines=f.read_text().splitlines();assert any(s.startswith('PIPELINE_CORRECT') for s in lines)
  rows=[json.loads(s.split(' ',1)[1]) for s in lines if s.startswith('PIPELINE_RESULT ')];assert len(rows)==5
  r=dict(projection=projection,m=m)
  for k in ('base','fused','separate'):r[k+'_us']=statistics.median(v[k+'_ms'] for v in rows)*1000
  r['overhead_percent']=(r['fused_us']/r['base_us']-1)*100;r['speedup_vs_separate']=r['separate_us']/r['fused_us'];r['raw_sha256']=hashlib.sha256(f.read_bytes()).hexdigest();out.append(r)
full=p.name.startswith('full_pipeline_')
source=(p/'consumer_pipeline.cu').read_text()
report=dict(status='complete',global_scale_applied=('host_maximum' in source) if full else None,scope=('Matched FourOverSix producer+GEMM+BF16consumer; fused pipeline has no separate permutation pass. Common tensoramax computed once and excluded; prototype quantizer, not optimal throughput claim; fixed E2M1 GEMM format; no fullMLP/model timing.' if full else 'Same GEMM and BF16 consumer for each pipeline; fused/separate include packed input gather. Up consumer also produces down-column order. Excludes activation quantization, global-factor selection, full MLP/full model. Fixed E2M1 GEMM format isolates permutation cost.'),measurements=out)
(p/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(out,indent=2))

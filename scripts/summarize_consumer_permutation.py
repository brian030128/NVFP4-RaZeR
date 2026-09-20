"""Summarize measured GB200 consumer fusion; no inferred end-to-end speedup."""
import json,statistics,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1]);lines=(p/'results.txt').read_text().splitlines()
rows=[json.loads(s.split(' ',1)[1]) for s in lines if s.startswith('RESULT ')]
checks=[s for s in lines if s.startswith('CORRECT ')];assert len(checks)==8
summary=[]
for op in ('silu_mul','residual'):
 for m in (1,128,512,2048):
  batch=[r for r in rows if r['op']==op and r['m']==m];assert len(batch)==5
  med={k:statistics.median(r[k] for r in batch)*1000 for k in ('base_ms','fused_ms','separate_ms')}
  summary.append(dict(op=op,m=m,n=batch[0]['n'],base_us=med['base_ms'],fused_us=med['fused_ms'],separate_us=med['separate_ms'],increment_us=med['fused_ms']-med['base_ms'],speedup_vs_separate=med['separate_ms']/med['fused_ms']))
report=dict(status='complete',checks=checks,scope='Consumer-only microbenchmark; exact accepted Qwen compact maps. BF16 SiLU-round-multiply-round and residual-add. No GEMM, activation quantization, RMSNorm, or end-to-end timing claimed.',measurements=summary,raw_sha256=hashlib.sha256((p/'results.txt').read_bytes()).hexdigest())
(p/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))

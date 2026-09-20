"""Summarize verified native timing logs; stdlib only."""
import argparse, hashlib, json, statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--sparse-output',action='store_true');a=p.parse_args()
rows=[]
for f in sorted(a.directory.glob('*_graph*.txt')):
    content=f.read_text();samples=[json.loads(line.split(' ',1)[1]) for line in content.splitlines() if line.startswith('PERM_RESULT ')]
    assert len(samples)==5 and content.count('PASSED')==3,f
    row=dict(projection=f.name.split('_m')[0],graph='_graph1' in f.name,m=samples[0]['m'],n=samples[0]['n'],k=samples[0]['k'])
    for key in ('gemm_ms','pipeline_ms','input_ms','output_ms'):row[key]=statistics.median(r[key] for r in samples)
    row['overhead_us']=1000*(row['pipeline_ms']-row['gemm_ms'])
    row['overhead_percent']=100*(row['pipeline_ms']/row['gemm_ms']-1)
    row['paired_overhead_us']=[1000*(r['pipeline_ms']-r['gemm_ms']) for r in samples]
    row['source']=str(f);rows.append(row)
assert len(rows)==20
assert 'PASSED' in (a.directory/'gemm_check.txt').read_text()
report=dict(status='complete',job_id=a.directory.name.split('_')[-1],device='NVIDIA GB200',
    method='Exact compacted Qwen permutations; full out-of-place16-element FP4 code/scale gather, existing SM100 mixed GEMM, '+('two-pass sparse in-place BF16 output restoration' if a.sparse_output else 'full out-of-place BF16 inverse output gather'),
    format_mode='Existing mixed GEMM E2M1 path; isolates permutation cost, does not validate native212-tile format map',
    correctness='Nonconstant FP4 codes/scales gathered byte-exact; BF16 restoration bitwise; small GEMM host-reference check passed',
    iterations=100,repetitions=5,alternating_baseline_pipeline_order=True,includes_quantization=False,includes_weight_preparation=False,native_fusion=False,rows=rows)
(a.directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['| Projection | Tokens | Graph | GEMM µs | Pipeline µs | Overhead µs | Overhead % |','|---|---:|---|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:(not r['graph'],r['projection'],r['m'])):
    lines.append(f"| {r['projection']} | {r['m']} | {r['graph']} | {r['gemm_ms']*1000:.2f} | {r['pipeline_ms']*1000:.2f} | {r['overhead_us']:.2f} | {r['overhead_percent']:.1f} |")
(a.directory/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))

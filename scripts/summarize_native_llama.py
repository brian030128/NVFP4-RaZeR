"""Summarize completed native full-model timings without rerunning GPU work."""
import argparse,json,statistics,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
r=json.loads((args.directory/'report.json').read_text());assert r['status']=='complete'
repetitions=r.get("timing_repetitions",5)
rows=[]
for length in (128,2048):
 arms={p:sorted((x for x in r['timings'] if x['policy']==p and x['prompt_tokens']==length),key=lambda x:x['rep']) for p in ('base','raw','arranged')}
 assert all(len(v)==repetitions for v in arms.values())
 for policy,samples in arms.items():
  row=dict(policy=policy,prompt_tokens=length,decode_tokens=32,repetitions=repetitions)
  for key,get in [('prefill_ms',lambda x:x['prefill']['cuda_ms']),('decode_ms_per_token',lambda x:x['decode']['cuda_ms']/32),('request_ms',lambda x:x['request_cuda_ms']),('request_wall_ms',lambda x:x['request_wall_ms'])]:
   values=[get(x) for x in samples];pairs=[100*(get(x)/get(b)-1) for x,b in zip(samples,arms['base'])]
   row[key]=statistics.median(values);row[key+'_min']=min(values);row[key+'_max']=max(values)
   row[key+'_paired_overhead_mean_pct']=statistics.mean(pairs);row[key+'_paired_overhead_2se_pct']=2*statistics.stdev(pairs)/math.sqrt(repetitions)
  rows.append(row)
gemms=[]
for name in sorted({x['name'] for x in r['gemm_only']}):
 for m in (1,128,2048):
  values=[x for x in r['gemm_only'] if x['name']==name and x['m']==m];assert len(values)==5
  a=statistics.median(x['base_us'] for x in values);b=statistics.median(x['arranged_us'] for x in values)
  gemms.append(dict(name=name,m=m,base_us=a,arranged_us=b,overhead_pct=100*(b/a-1)))
summary=dict(job=r['job'],performance_diagnostic=r.get('performance_diagnostic',False),native_accuracy_established=r.get('native_accuracy_established',False),full_output_equivalence_gate_passed=r.get('full_output_equivalence_gate_passed',False),full_model=rows,gemm_only=gemms)
(args.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
text='''### Full-model native Llama-3.1-8B latency on GB200 (diagnostic)

This measures the complete model using the sibling SM100 GEMM with a tested
per-256×64 format map. All 224 transformer projections use native packed FP4;
embeddings, normalization, attention, residuals, and the vocabulary head retain
their normal model implementation. Every packed weight decodes to its frozen
fake-quantized BF16 reference bitwise. All 672 projection checks across three
policies passed (<0.005 relative error against a decoded FP64 dot reference),
activation codes/scales matched exactly, and native repeated forwards were
bitwise reproducible. The warmed follow-up reuses these audits after hash/AST checks
and repeats every source-weight and packed-weight audit.

**This is a diagnostic latency measurement, not validated native model quality.**
The free-running full-model reference-equivalence gate failed for mixed policies.
Small GEMM rounding differences amplify through subsequent FP4 quantization;
the first divergent layer has relative error 4.586e-6, while arranged final
logits differ by about 10.8% from the FP32 epilogue-order reference. The published
PPL figures remain the prior BF16 fake-quantized quality measurements. Native
PPL is unmeasured, and previous full-output gate failures remain failures.

Batch size is one. Each request performs prefill and 32 actual KV-cached decode
steps using identical fixed tokens across policies. The head computes last-token
logits, as in generation. Results are medians of REPS paired runs. WARMUP Loading, offline packing, tokenizer work, and sampling are excluded.
Activation amax, FourOverSix activation quantization, attention, normalization,
cache updates, and permutations are included. These are measurements of this
matched eager Transformers/native backend, not an optimized serving engine.

Columns are fused into the packed activation store. This first full-model
integration restores permuted output rows in a separate kernel; it does not
claim the fully consumer-fused microbenchmark implementation.

| Prompt tokens | Policy | Prefill ms | Decode ms/token | Request ms (prefill + 32 tokens) | Paired request overhead, mean ±2SE |
|---:|---|---:|---:|---:|---:|
'''
text=text.replace('REPS',str(repetitions)).replace('WARMUP',
 'Two complete requests per policy/context warmed all 32 decode positions; all six policy orders were measured and Python GC was disabled.' if r.get('full_request_warmups_per_policy')==2 else
 'This first pilot warmed only one decode position; cold initialization affected its baseline samples, so it must not establish a speedup.')
labels={'base' :'FourOverSix NVFP4','raw':'Raw MixFP4 256×64 (187 tiles)','arranged':'Arranged MixFP4 256×64 (147 tiles)'}
for x in rows:
 over='—' if x['policy']=='base' else f"{x['request_ms_paired_overhead_mean_pct']:+.2f}% ± {x['request_ms_paired_overhead_2se_pct']:.2f}%"
 text+=f"| {x['prompt_tokens']} | {labels[x['policy']]} | {x['prefill_ms']:.3f} | {x['decode_ms_per_token']:.3f} | {x['request_ms']:.3f} | {over} |\n"
text+='''
The ±2SE values describe paired run variation across these paired repetitions;
small negative overhead does not establish a speedup. CUDA elapsed time is shown;
synchronized host-wall measurements and each repetition are retained in the report.

For warmed job 406828, the 128-token paired request overhead is informative only
for this eager backend. At 2,048 tokens, decode time varies roughly 27–53 ms/token
across all policies. **The 2,048-token overhead is inconclusive**; retain every
sample and do not interpret its point estimate as an algorithmic penalty.
The earlier job 406751 is retained as a cold-initialization pilot, not the primary
request comparison. No post-hoc samples are removed.

The following **GEMM-only** measurements use the actual frozen final-MLP weights
and heterogeneous format masks. Activation quantization and output restoration
are outside this timing. Each sample replays a graph containing 100 GEMM calls;
values are medians of five paired repetitions. These short graph measurements
also vary between jobs: for example, gate M=1 changes from +0.32% in pilot 406751
to -12.91% in 406828. Negative point estimates do not establish a speedup.
This measures the complete arranged-versus-baseline GEMM difference, not an
isolated format-branch cost. The uniform-format 8192-cubed control above has a
different scope.

| Projection | M | NVFP4 µs | Arranged MixFP4 µs | GEMM-only overhead |
|---|---:|---:|---:|---:|
'''
for x in gemms:text+=f"| {x['name'].split('.')[-1]} | {x['m']} | {x['base_us']:.3f} | {x['arranged_us']:.3f} | {x['overhead_pct']:+.2f}% |\n"
text+=f"\n[Complete measurements]({args.directory}/report.json), [summary]({args.directory}/summary.json), and [frozen protocol](results/task_reorder/full_model_20260920/plan.json).\n"
(args.directory/'report_section.md').write_text(text)
print(json.dumps(summary,indent=2))

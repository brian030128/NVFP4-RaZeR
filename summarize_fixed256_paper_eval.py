"""Publish every fixed-256 map under one matched paper evaluation protocol."""
import argparse
import csv
import json
import os
from pathlib import Path
from run_conditional_model import paired


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); args=ap.parse_args()
    cases=json.loads((args.root/'cases.json').read_text()); reports={}; rows=[]
    assert len(cases)==33
    for c in cases:
        r=json.loads((args.root/c['id']/'report.json').read_text())
        assert r['status']=='complete', (c['id'],r.get('error'))
        assert r['maps_unchanged'] and r['source_weights_verified'] and not r['recalibration']
        assert r['length']==2048 and r['qwen_o_proj_quantized']
        assert r['wiki_use_cache'] and not r['c4_use_cache']
        assert r['selected_e0m3_blocks']==(0 if c['policy']=='four_over_six' else 256)
        reports.setdefault(c['model'],{})[c['policy']]=r
    names={'qwen4b':'Qwen3-4B','llama8b':'Llama-3.1-8B','qwen27b':'Qwen3.8-27B'}
    body=['## Math/code-only calibration: fixed-256 W4A4, aligned 2048-token benchmark', '',
          'Calibration uses **OpenWebMath and CodeParrot only; neither WikiText nor C4 is used for calibration**. '
          'One 128-sequence causal scoring pass per model (64 math + 64 code, 512 tokens each) '
          'supplies the ten frozen source/sample-count settings below. The numeric suffix denotes '
          'the total calibration sequence count; math_code settings use equal numbers from each source.', '',
          'This evaluation replays all ten frozen math/code-only fixed-256 maps for each of three models. '
          '**No adaptive maps, weight-MSE controls, recalibration, or best-setting selection are included.** '
          'The same map is used for both datasets. All maps contain exactly **256 E0M3 type blocks** '
          'of 8×64 weights; all other targeted weights use FourOverSix E2M1.', '',
          '**Evaluation:** WikiText-2 raw test text concatenated with double newlines, nonoverlapping '
          '2048-token windows; C4 validation shard 00000, 256 seed-0 random 2048-token crops using the '
          'released sampling rule. As in the release, WikiText uses a fresh cache per window and C4 '
          'disables it; no cache is carried between windows. '
          'Incomplete WikiText tails are omitted. PPL uses the released float32 '
          'NLL aggregation. WikiText and C4 token hashes match the released-code reproduction exactly '
          'for Llama-3.1-8B and Qwen3-4B.', '',
          '**W4A4 definition:** tensor-wide FourOverSix activation factors, 16-element scale blocks, '
          'BF16 simulation, SDPA attention, and unquantized KV cache. Every targeted linear input is '
          'quantized, including Qwen o_proj. The small-model baseline is checked against the corrected '
          'repository wrapper. Qwen3.8-27B uses native Transformers 5.16.1 text-linear modules; its '
          'vision, convolution, recurrent state operations, normalization, and LM head are outside that '
          'linear quantization scope. It has no corresponding RaZeR Table 3 entry.', '',
          '**Alignment means matching the evaluation protocol, not reproducing every published Table 3 value.** '
          'Baselines below are freshly measured with matching inputs and quantization scope. '
          'In particular, quantizing Qwen o_proj corrects an omission in the archived release. '
          'All improvements reported here are relative to **FourOverSix**, not NVFP4.', '',
          'The older 512-token tables remain historical measurements under different context, C4 '
          'sampling, and per-token activation scales. These new tensor-wide factors can depend on '
          'future tokens, as in the paper-style simulation; these results are not a causal deployment '
          'claim. Maps were originally fitted with causal 512-token math/code calibration and are '
          'transferred here unchanged.', '',
          'ΔPPL = fixed-256 PPL minus matched FourOverSix PPL; **negative is better**. '
          'Average is the unweighted arithmetic mean of the separate Wiki/C4 PPL values.', '']
    gains=0; comparisons=0
    for model in ('qwen4b','llama8b','qwen27b'):
        group=reports[model]; base=group['four_over_six']
        assert len(group)==11
        if model!='qwen27b': assert base['corrected_wrapper_equivalence']['equal']
        bw=base['evaluation']['wiki']['ppl']; bc=base['evaluation']['c4']['ppl']
        body += [f"### {names[model]}", '',
                 '| Calibration | E0M3 blocks | Wiki PPL | C4 PPL | Average | Δ Wiki | Δ C4 | Δ average |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|',
                 f'| FourOverSix baseline | 0 | {bw:.6f} | {bc:.6f} | {(bw+bc)/2:.6f} | +0.000000 | +0.000000 | +0.000000 |']
        for policy,r in group.items():
            for d in ('wiki','c4_paper'):
                assert r['data'][d]['token_sha256']==base['data'][d]['token_sha256']
            assert r['activation_quantized_modules']==base['activation_quantized_modules']
            assert r['map_sha256']==base['map_sha256']
            if policy=='four_over_six': continue
            w=r['evaluation']['wiki']['ppl']; c=r['evaluation']['c4']['ppl']
            row=dict(model=model,policy=policy,selected_e0m3_blocks=256,wiki_ppl=w,c4_ppl=c,
                     average=(w+c)/2,wiki_delta=w-bw,c4_delta=c-bc,average_delta=(w+c-bw-bc)/2,
                     baseline_wiki=bw,baseline_c4=bc,
                     paired={d:paired(r['evaluation'][d]['nll'],base['evaluation'][d]['nll']) for d in ('wiki','c4')})
            for d in ('wiki','c4'):
                row['paired'][d]['ppl_delta']=r['evaluation'][d]['ppl']-base['evaluation'][d]['ppl']
            rows.append(row); gains+=(w<bw)+(c<bc); comparisons+=2
            body.append(f'| {policy.removeprefix("fixed256_")} | 256 | {w:.6f} | {c:.6f} | {(w+c)/2:.6f} | {w-bw:+.6f} | {c-bc:+.6f} | {(w+c-bw-bc)/2:+.6f} |')
        body+=['']
    job_ids=sorted({r['job_id'] for g in reports.values() for r in g.values()})
    body += [f'Fixed-256 improves point PPL in **{gains}/{comparisons}** matched model/setting/dataset cells. '
             'Qwen3-4B and Llama-3.1-8B improve in all 20 cells each; Qwen3.8-27B improves in '
             '14/20, with six WikiText regressions and ten C4 improvements. '
             'These settings share calibration data and evaluation windows; this count is descriptive, '
             'not a count of independent confirmations. Per-window losses and paired NLL diagnostics '
             'are retained in the machine-readable summary; intervals do not account for WikiText '
             'article dependence or calibration-draw variability.', '',
             f'Execution jobs: {", ".join(job_ids)}; consolidated results: `{args.root.name}`. '
             'Each job uses `gov113008/taide_h200`, an eight-H200 allocation, and independent '
             'one-GPU Slurm steps. Cache-only repairs reuse C4 after exact identity checks and retain '
             'the pre-fix reports. Every dataset comparison uses identical inputs '
             'and activation quantization scope within its model.', '']
    summary=dict(status='complete',cases=33,ppl_cells=66,point_gains=gains,comparisons=comparisons,
                 rows=rows,baselines={m:g['four_over_six']['evaluation'] for m,g in reports.items()})
    (args.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    columns=['model','policy','selected_e0m3_blocks','baseline_wiki','wiki_ppl','wiki_delta',
             'baseline_c4','c4_ppl','c4_delta','average','average_delta']
    with (args.root/'fixed256_cells.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    text='\n'.join(body); (args.root/'REPORT.md').write_text(text)
    project=Path(__file__).resolve().parent; main_report=project/'MIXFP4_REPORT.md'
    start,end='<!-- FIXED256_PAPER_EVAL_START -->','<!-- FIXED256_PAPER_EVAL_END -->'
    link=(args.root.resolve()/'REPORT.md').relative_to(project)
    block=start+'\n'+text+f'\n[Full results]({link}) · [Protocol](results/fixed256_paper_eval/PROTOCOL.md).\n\n'+end
    old=main_report.read_text()
    if start in old:
        assert old.count(start)==old.count(end)==1
        before,tail=old.split(start,1); _,after=tail.split(end,1); old=before+after
    history_start,history_end='<!-- MATH_CODE_ADAPTIVE_START -->','<!-- MATH_CODE_ADAPTIVE_END -->'
    history=(history_start+'\n**Historical 512-token adaptive study:** '
             '[Full tables and diagnostics](results/math_code_adaptive/summary_333786_333788/REPORT.md). '
             'Its adaptive counts, fixed-256 controls, and weight-MSE controls use a different '
             'evaluation protocol and are not part of the aligned benchmark above.\n'+history_end)
    if history_start in old:
        assert old.count(history_start)==old.count(history_end)==1
        before,tail=old.split(history_start,1); _,after=tail.split(history_end,1)
        old=before+block+'\n\n'+history+after
    else:
        anchor='<!-- END POOLED PERFORMANCE TABLES -->'; assert anchor in old
        old=old.replace(anchor,anchor+'\n\n'+block+'\n\n'+history,1)
    old=old.replace('The math/code-only study above uses **512-token evaluation windows**.',
                    'The historical math/code-only study used **512-token evaluation windows**.')
    marker='<!-- FIXED256_CURRENT_POINTER -->'
    if marker not in old:
        heading='**Motivation, method, experiments, and limitations**'
        notice=(f'{marker}\n**Paper-protocol fixed-256 results (2048 tokens):** '
                f'[matched WikiText/C4 measurements]({link}). '
                'The earlier 512-token tables below use a different evaluation protocol.\n')
        old=old.replace(heading,heading+'\n\n'+notice,1)
    old=old.replace('Updated September 8, 2026 ·','Updated September 9, 2026 ·',1)
    old=old.replace('The earlier 512-token tables below use a different evaluation protocol.',
                    'The Math/code-only section below contains these results; other 512-token tables are historical.')
    main_report.write_text(old)
    print(f'VALIDATED 33 cases / 66 PPL cells; fixed256 point gains {gains}/{comparisons}',flush=True)


if __name__=='__main__': main()

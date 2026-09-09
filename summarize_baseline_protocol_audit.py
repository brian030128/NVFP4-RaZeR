"""Summarize completed matched baseline diagnostics without changing study maps."""
import argparse
import json
import math
import os
from pathlib import Path
from run_c4_frozen import digest_file

LABELS = {'qwen4b':'Qwen3-4B', 'llama8b':'Llama-3.1-8B'}
PAPER = {'qwen4b': {'bf16':(13.66,16.65), 'four_over_six_tensor':(13.88,17.21), 'nvfp4_tensor':(13.88,17.21)},
         'llama8b': {'bf16':(6.24,8.96), 'four_over_six_tensor':(6.88,9.83), 'nvfp4_tensor':(6.95,9.94)}}


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap=argparse.ArgumentParser(); ap.add_argument('--job',required=True)
    ap.add_argument('--historical'); args=ap.parse_args()
    root=Path('results/baseline_protocol_audit'); reports={}; paths={}
    for task in range(4):
        p=root/f'job_{args.job}_{task}/report.json'; r=json.loads(p.read_text())
        assert r['status']=='complete' and not r['calibration']
        for ds in r['evaluation'].values():
            for d,v in ds.items():
                assert len(v['nll'])==r['data'][d]['windows']
                assert math.isclose(v['ppl'],math.exp(sum(v['nll'])/len(v['nll'])),rel_tol=1e-12)
                assert v['scored_tokens']==len(v['nll'])*(r['length']-1)
        reports[r['model'],r['length']]=r; paths[str(p)]=digest_file(p)
    for model in LABELS:
        short=reports[model,512]; long=reports[model,2048]
        for key in ('source','revision','torch_version','transformers_version','dtype','attention'):
            assert short[key]==long[key], key
        for key in ('quantize/quantizer.py','quantize/causal_four_over_six.py'):
            assert short['source_sha256'][key]==long['source_sha256'][key]
        assert short['data']['wiki']['used_tokens']==long['data']['wiki']['used_tokens']
        assert short['data']['c4_paper']['documents']==long['data']['c4_paper']['documents']
        for d in ('wiki','c4_paper'):
            assert short['data'][d]['windows']==4*long['data'][d]['windows']
    lines=['# Audit of the discrepancy from RaZeR Table 3','',
        'No calibration or E0M3 selection was performed. All models use pinned original weights, '
        'BF16 execution, eager attention, and unquantized KV tensors. The released evaluator also '
        'loads BF16 despite its FP16 label. FourOverSix quantizes the same nonhead linear weights '
        'and inputs as the math/code study.','',
        'Wiki uses identical complete 2048-token spans, split into four windows for the 512 condition. '
        'C4 paper-protocol crops use shard 00000 and the released seed-0 sampling procedure; '
        'their 512 condition splits the same parent crops into four windows. More window boundaries '
        'also change which first tokens are excluded from loss. The historical C4 study uses different '
        'documents from shard 00001.','',
        '| Model | Context | Method | Wiki PPL | C4 paper-sample PPL | C4 historical-sample PPL |',
        '|---|---:|---|---:|---:|---:|']
    for model in LABELS:
        for length in (512,2048):
            r=reports[model,length]
            for policy,ds in r['evaluation'].items():
                lines.append(f'| {LABELS[model]} | {length} | {policy} | {ds["wiki"]["ppl"]:.6f} | '
                    f'{ds["c4_paper"]["ppl"]:.6f} | '+(f'{ds["c4_study"]["ppl"]:.6f}' if 'c4_study' in ds else '—')+' |')
    lines+=['','## Direct comparison with the published values','',
        'The audit column uses 2048 tokens and tensor-wide activation factors. Paper values are rounded '
        'to two decimals. Δ is audit minus paper; the comparison does not assert identical checkpoint '
        'revisions or attention backends to the original experiment.','',
        '| Model | Method | Paper Wiki | Audit Wiki | ΔWiki | Paper C4 | Audit C4 | ΔC4 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in LABELS:
        for policy,(pw,pc) in PAPER[model].items():
            ds=reports[model,2048]['evaluation'][policy]; w=ds['wiki']['ppl']; c=ds['c4_paper']['ppl']
            lines.append(f'| {LABELS[model]} | {policy} | {pw:.2f} | {w:.6f} | {w-pw:+.6f} | {pc:.2f} | {c:.6f} | {c-pc:+.6f} |')
    if args.historical:
        path=Path(args.historical); h=json.loads(path.read_text()); current=reports['qwen4b',2048]
        assert h['status']=='complete' and h['historical_qwen_unquantized_o_proj_input']
        assert h['model']=='qwen4b' and h['length']==2048
        for k in ('source','revision','torch_version','transformers_version','dtype','attention','data'):
            assert h[k]==current[k], k
        paths[str(path)]=digest_file(path)
        lines+=['','## Historical Qwen attention-output activation bug','',
            'Repository commit [abab3c6](qwen_o_proj_fix.patch) '
            '(2026-06-17) changed o_proj(attn_output) to o_proj(attn_output_quant). The earlier Qwen '
            'wrapper computed but discarded quantized attention-output activations. The following '
            'diagnostic reproduces that behavior by leaving only those projection inputs unquantized; '
            'their weights remain quantized. It uses exactly the same checkpoints, input hashes and '
            '2048-token windows as the corrected audit.','',
            '| Method | Paper Wiki | Historical-behavior Wiki | Corrected Wiki | Paper C4 | Historical-behavior C4 | Corrected C4 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for policy in ('four_over_six_tensor','nvfp4_tensor'):
            pw,pc=PAPER['qwen4b'][policy]; ds=h['evaluation'][policy]; ds2=current['evaluation'][policy]
            for d,v in ds.items():
                assert len(v['nll'])==h['data'][d]['windows']
                assert math.isclose(v['ppl'],math.exp(sum(v['nll'])/len(v['nll'])),rel_tol=1e-12)
            lines.append(f'| {policy} | {pw:.2f} | {ds["wiki"]["ppl"]:.6f} | {ds2["wiki"]["ppl"]:.6f} | '
                f'{pc:.2f} | {ds["c4_paper"]["ppl"]:.6f} | {ds2["c4_paper"]["ppl"]:.6f} |')
        lines+=['','This comparison measures the effect of the historical behavior. Any remaining '
            'gap to the paper is still unresolved; the code history does not independently prove '
            'which exact revision generated its table. The corrected W4A4 baseline remains the '
            'appropriate comparison for a method that quantizes all these inputs.']
    lines+=['','## Sequential protocol differences for FourOverSix','',
        'These are changes along a fixed comparison path, not independently additive causal effects. '
        'The residual includes paper rounding and remaining implementation/checkpoint differences.','',
        '| Model | Dataset | Historical 512 PPL | Aligned-data 512 PPL | Per-token 2048 PPL | Tensor-wide 2048 PPL | Paper PPL |',
        '|---|---|---:|---:|---:|---:|---:|']
    for model in LABELS:
        old=json.loads(Path(f'results/math_code_adaptive/evaluation_333786_{model}/report.json').read_text())
        for i,domain in enumerate(('wiki','c4')):
            d='wiki' if domain=='wiki' else 'c4_paper'
            values=[old['evaluation']['four_over_six'][domain]['ppl'],
                reports[model,512]['evaluation']['four_over_six_row'][d]['ppl'],
                reports[model,2048]['evaluation']['four_over_six_row'][d]['ppl'],
                reports[model,2048]['evaluation']['four_over_six_tensor'][d]['ppl'],PAPER[model]['four_over_six_tensor'][i]]
            lines.append(f'| {LABELS[model]} | {domain} | '+' | '.join(f'{v:.6f}' for v in values)+' |')
        assert all(v['equal'] for v in reports[model,2048]['wrapper_equivalence'].values())
        assert reports[model,512]['evaluation']['four_over_six_row']['c4_study']['previous_max_nll_error']<=1e-6
        assert reports[model,512]['evaluation']['four_over_six_row']['wiki']['previous_max_nll_error']<=1e-6
    lines+=['','The native hook implementation and released quantized model wrappers produced exactly '
        'equal logits on the first 2048-token Wiki window for BF16 and tensor-wide FourOverSix at '
        'matched eager attention. Every historical C4 baseline loss and every shared historical Wiki '
        'baseline loss replayed within 1e-6. These checks constrain, but do not exhaust, possible bugs.','',
        'Tensor-wide activation factors may depend on later tokens within the evaluated window; '
        'matching the released evaluator is a reproduction condition, not a causal deployment guarantee. '
        'The historical sparse-map gains remain specific to their 512-token protocol; this audit '
        'does not evaluate those maps at 2048 or validate a paper-comparable improvement.','',
        '[Protocol](PROTOCOL.md). [RaZeR Table 3](https://arxiv.org/html/2501.04052v2#S4.T3).']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (root/'source_reports.json').write_text(json.dumps(paths,indent=2)+'\n')
    note=['## Baseline audit against RaZeR Table 3','',
        'The historical math/code-only study used **512-token evaluation windows**. A separate baseline-only '
        'audit at **2048 tokens**, with the released C4 sampling procedure and tensor-wide activation '
        'factors, gives the values below. No calibration or E0M3 map selection occurs in this audit. '
        'The unquantized baselines match the paper to its displayed precision; shorter context accounts '
        'for most of the large gap in the original study.','',
        '| Model | W4A4 method | Paper Wiki PPL | Audit Wiki PPL | Paper C4 PPL | Audit C4 PPL |',
        '|---|---|---:|---:|---:|---:|']
    for model in LABELS:
        for policy,label in [('nvfp4_tensor','NVFP4'),('four_over_six_tensor','FourOverSix')]:
            pw,pc=PAPER[model][policy]; ds=reports[model,2048]['evaluation'][policy]
            note.append(f'| {LABELS[model]} | {label} | {pw:.2f} | {ds["wiki"]["ppl"]:.6f} | {pc:.2f} | {ds["c4_paper"]["ppl"]:.6f} |')
    if args.historical:
        note+=['','The Qwen wrapper formerly computed quantized attention-output activations but passed '
            'the unquantized tensor into o_proj; repository commit abab3c6 fixed this on 2026-06-17. '
            'A separate historical-behavior diagnostic leaves only those projection inputs unquantized '
            'at the same 2048-token context. Its results are:','',
            '| Qwen3-4B historical behavior | Wiki PPL | C4 PPL |','|---|---:|---:|']
        for policy,label in [('nvfp4_tensor','NVFP4'),('four_over_six_tensor','FourOverSix')]:
            ds=h['evaluation'][policy]
            note.append(f'| {label} | {ds["wiki"]["ppl"]:.6f} | {ds["c4_paper"]["ppl"]:.6f} |')
    note+=['','The corrected full-W4A4 measurements remain the appropriate baseline for methods '
        'quantizing all these inputs. Residual differences from the paper remain visible above; '
        'a historical-behavior match does not independently establish the exact code that generated '
        'the published table. **This baseline-only audit did not re-evaluate E0M3 maps. The subsequent '
        'Math/code-only fixed-256 benchmark above reports their aligned 2048-token results.**','',
        '[Full matched-context audit and validation](results/baseline_protocol_audit/REPORT.md) · '
        '[RaZeR Table 3](https://arxiv.org/html/2501.04052v2#S4.T3).','']
    start='<!-- BASELINE_PROTOCOL_AUDIT_START -->'; end='<!-- BASELINE_PROTOCOL_AUDIT_END -->'
    block=start+'\n'+'\n'.join(note)+'\n'+end
    path=Path('MIXFP4_REPORT.md'); text=path.read_text()
    if start in text:
        assert text.count(start)==text.count(end)==1
        text=text[:text.index(start)]+block+text[text.index(end)+len(end):]
    else:
        anchor='<!-- MATH_CODE_ADAPTIVE_END -->'; pos=text.index(anchor)+len(anchor)
        text=text[:pos]+'\n\n'+block+text[pos:]
    path.write_text(text)
    print('\n'.join(lines),flush=True)


if __name__=='__main__': main()

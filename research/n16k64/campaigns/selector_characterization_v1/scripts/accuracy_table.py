"""CPU-only secondary full8 point tables, independently checked against arrays."""
from common import *
from secondary_accuracy import TASKS

def main():
    start=now();source=OUT/'results/secondary_accuracy_current.csv'
    rows=list(csv.DictReader(source.open()));assert len(rows)==96
    lookup={(r['model'],r['policy'],r['task']):r for r in rows}
    assert len(lookup)==96
    inputs={str(source.relative_to(OUT)):sha(source)};summary=[]
    lines=['# Secondary accuracy: fixed seed0, full eight tasks','',
        'Equal-task macro points; MMLU is question-weighted within its task. '
        'No confidence interval, non-inferiority, population-risk or five-draw accuracy claim. '
        'Only admitted complete runs enter these tables.','',
        '| Model | Policy | Valid tasks | Macro accuracy (%) | Change vs baseline (pp) | Negative task changes |',
        '|---|---|---:|---:|---:|---:|']
    for model in MODELS:
        for policy in ['baseline','joint','ce_matched','kl_matched']:
            valid=[lookup[model,policy,t] for t in TASKS if lookup[model,policy,t]['status']!='coverage_gap']
            assert len(valid) in [0,8], 'Partial policies must not be silently macro-averaged'
            if not valid:
                lines.append(f'| {model} | {policy} | 0/8 | missing | missing | missing |')
                summary.append(dict(model=model,policy=policy,status='coverage_gap'));continue
            values=[];deltas=[]
            for r in valid:
                task=r['task'];new=policy in ['ce_matched','kl_matched']
                path=OUT/'results'/('accuracy_new_arrays' if new else 'accuracy_arrays')/f'{model}_{task}.npz'
                inputs[str(path.relative_to(OUT))]=sha(path)
                with np.load(path,allow_pickle=False) as a:
                    labels=a['policies'].tolist();x=a['correct'][labels.index(policy)];b=a['correct'][labels.index('baseline')]
                    assert np.isin(x,[0,1]).all() and x.shape==b.shape
                    assert abs(float(x.mean())-float(r['value']))<1e-12
                    assert abs(float(b.mean())-float(lookup[model,'baseline',task]['value']))<1e-12
                    delta=float((x-b).mean())
                    if new:assert abs(delta-float(r['delta_vs_baseline']))<1e-12
                    values.append(float(x.mean()));deltas.append(delta)
            macro=float(np.mean(values));difference=float(np.mean(deltas));negative=sum(d<0 for d in deltas)
            summary.append(dict(model=model,policy=policy,status='array_checked',macro_accuracy=macro,
                macro_delta_vs_baseline=difference,negative_tasks=negative,tasks=8))
            lines.append(f'| {model} | {policy} | 8/8 | {100*macro:.4f} | {100*difference:+.4f} | {negative}/8 |')
    table=OUT/'results/SECONDARY_ACCURACY_TABLE.md';table.write_text('\n'.join(lines)+'\n')
    audit=OUT/'results/SECONDARY_ACCURACY_TABLE_AUDIT.json'
    jsonout(audit,dict(checked_utc=start,source_hashes=inputs,script_sha256=sha(Path(__file__)),results=summary,
        qualification='Raw paired binary correctness arrays independently recomputed; points only, no new inference.'))
    log('accuracy-table','python scripts/accuracy_table.py',start,[table,audit])
    print('Array-checked complete policy rows',sum(r['status']=='array_checked' for r in summary))

if __name__=='__main__':main()

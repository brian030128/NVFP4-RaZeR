"""Audit two bounded Ada resource probes without changing scientific gates."""
from common import *
from accuracy_new_results import require_gpu_record

def main():
    start=now();plan=load(OUT/'results/ADA_ACCURACY_RESOURCE_PROBE.json')
    table=list(csv.DictReader((OUT/'results/anchor_windows.csv').open()));result=[]
    runtime=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'
    for model,run in zip(plan['models'],plan['runs']):
        root=runtime/'runs'/run;lp=root/'launch_record.json'
        if not lp.exists() or load(lp)['status']=='running':
            result.append(dict(model=model,run=run,status='waiting_or_running',accuracy_promotion=False));continue
        try:
            launch=require_gpu_record(root)
            rows=[r for r in table if r['run']==run]
            assert len(rows)==8 and all(r['model']==model for r in rows)
            identities=all(r['tokens_equal']=='True' and r['weights_equal']=='True' for r in rows)
            maximum=max(abs(float(r['delta_nll'])) for r in rows)
            passed=identities and maximum<=1e-10
            result.append(dict(model=model,run=run,status='prefix_passed' if passed else 'numerical_anchor_failed',
                identities_match=identities,max_abs_window_delta_nll=maximum,absolute_tolerance=1e-10,
                gpu_hours=launch['gpu_hours'],launch_sha256=sha(lp),report_sha256=sha(root/'ppl/ppl_report.json'),
                accuracy_promotion=False,
                next_action='Separate CPU-verified Ada accuracy adapter plus complete baseline pairing required before any acceptance' if passed else 'Keep existing A6000 wait. No tolerance relaxation or automatic numerical retry.'))
        except (AssertionError,KeyError,OSError,ValueError) as exc:
            result.append(dict(model=model,run=run,status='operationally_invalid_or_failed',reason=str(exc),accuracy_promotion=False))
    dest=OUT/'results/ADA_ACCURACY_RESOURCE_RESULT.json'
    jsonout(dest,dict(checked_utc=start,plan_sha256=sha(OUT/'results/ADA_ACCURACY_RESOURCE_PROBE.json'),models=result,
        qualification='Resource-eligibility diagnosis only. Does not isolate physical cause, measure native execution, add a new quality endpoint, or override exhausted T3 retries. Original A6000 anchors are unchanged.'))
    log('ada-resource-probe-audit','python scripts/ada_resource_audit.py',start,[dest]);print(result)

if __name__=='__main__':main()

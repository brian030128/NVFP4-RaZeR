"""Validate completed model reports while the rest of the fixed study runs."""
import argparse
import json
import math
import os
from pathlib import Path
from quantize.adaptive_prefix import source_subsets
from run_c4_frozen import digest_file
from run_conditional_model import paired
from summarize_math_code_adaptive import DOMAINS, LABELS, performance_table


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap=argparse.ArgumentParser(); ap.add_argument('--reports',nargs='+',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    lines=['# Completed smaller-model results — full study still pending','',
           'Calibration: OpenWebMath and CodeParrot only; evaluation: WikiText-2 and C4 only. '
           'One shared causal score pass per model; no calibration-seed replication. '
           'All ten settings and both count rules are retained. This file does not represent '
           'completion of the three-model study.','']
    summaries={}
    for source in args.reports:
        path=Path(source); r=json.loads(path.read_text()); model=r['model']
        assert r['status']=='complete' and r['maps_unchanged'] and r['source_weights_verified']
        assert not r['uses_c4_calibration'] and not r['uses_wiki_calibration']
        cal=Path(r['calibration']); assert digest_file(cal/'report.json')==r['calibration_report_sha256']
        assert digest_file(cal/'maps.json')==r['map_sha256']
        c=json.loads((cal/'report.json').read_text())
        assert set(c['fit'])=={'math','code'} and c['subsets']==source_subsets()
        assert c['block_statistics']==r['block_statistics']
        for policy,domains in r['evaluation'].items():
            assert set(domains)==set(DOMAINS) and r['suffix_intervention'][policy]['equal']
            for domain,e in domains.items():
                n=r['data'][domain]['windows']
                assert len(e['nll'])==n and e['scored_tokens']==511*n
                assert all(math.isfinite(v) for v in e['nll'])
                assert abs(math.exp(sum(e['nll'])/n)-e['ppl'])<1e-12
                if policy!='four_over_six': assert paired(e['nll'],r['evaluation']['four_over_six'][domain]['nll'])==r['contrasts'][policy][domain]
                if e['origin']['reused']:
                    assert digest_file(e['origin']['report'])==e['origin']['report_sha256']
                    assert e['origin']['first_window_absolute_error']<=1e-6
        lines += [f'{LABELS[model]} baseline: Wiki {r["evaluation"]["four_over_six"]["wiki"]["ppl"]:.6f}; '
                  f'C4 {r["evaluation"]["four_over_six"]["c4"]["ppl"]:.6f}.','']
        lines += performance_table(model,r)+['']
        adaptive=[f'adaptive_{s}' for s in source_subsets()]
        cs=[r['contrasts'][p][d] for p in adaptive for d in DOMAINS]
        matched=[paired(r['evaluation'][p][d]['nll'],r['evaluation'][p.replace('adaptive_','fixed256_',1)][d]['nll'])
                 for p in adaptive for d in DOMAINS]
        summaries[model]=dict(source_report=source,adaptive_cells=len(cs),
            zero_maps=sum(r['block_statistics'][p]['selected_blocks']==0 for p in adaptive),
            point_gains=sum(v['mean_nll']<0 for v in cs),
            supported_gains=sum(v['mean_nll']+v['two_se']<0 for v in cs),
            supported_harms=sum(v['mean_nll']-v['two_se']>0 for v in cs),
            adaptive_beats_fixed_point=sum(v['mean_nll']<0 for v in matched),
            mixed128={p:{d:r['evaluation'][p][d]['ppl'] for d in DOMAINS}
                      for p in ('four_over_six','adaptive_math_code128','fixed256_math_code128')})
    lines += ['The source/sample-count cells share data and often selected blocks. Gain counts '
              'are descriptive, not independent replications. Two-SE intervals do not account '
              'for WikiText article dependence or multiple comparisons. The adaptive objective '
              'is an estimated-curvature surrogate, not a true finite-switch loss bound.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__': main()

"""Summarize the frozen panel without selecting policies or modifying maps."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory'); args=ap.parse_args()
    out=Path(args.directory); report=json.loads((out/'report.json').read_text())
    assert report['status']=='complete'
    matrices=report['matrices']; crops=[c for m in matrices for c in m['crops']]
    def ratio(c,policy):
        return c['policies'][policy]['spectral_squared']/c['policies']['baseline']['spectral_squared']
    stats=dict(
        matrices=len(matrices), changed_matrices=sum(m['selected_tiles']>0 for m in matrices),
        selected_tiles=sum(m['selected_tiles'] for m in matrices),
        candidate_tiles=sum(m['tile_count'] for m in matrices),
        crops=len(crops),
        reference_crop_improvements=sum(ratio(c,'reference')<1-1e-8 for c in crops),
        approximate_crop_improvements=sum(ratio(c,'approximate')<1-1e-8 for c in crops),
        approximate_crop_regressions=sum(ratio(c,'approximate')>1+1e-8 for c in crops),
        approximation_worse_than_reference=sum(ratio(c,'approximate')>ratio(c,'reference')+1e-8 for c in crops),
        max_crop_estimation_relative_error=max(abs(p['relative_estimation_error']) for c in crops for p in c['policies'].values()),
        max_full_audit_relative_residual=max(p['relative_residual'] for m in matrices for p in m['policies'].values()),
        max_selection_audit_relative_difference=max(abs(m['selection']['history'][-1]['value']/m['policies']['spectral_approximate']['value']-1) for m in matrices),
        total_selection_seconds=sum(m['selection_seconds'] for m in matrices),
        total_candidate_seconds=sum(m['candidate_seconds'] for m in matrices),
        elapsed_seconds=report['elapsed_seconds'],
        max_peak_allocated_gib=max(m['peak_allocated_bytes'] for m in matrices)/2**30,
    )
    (out/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    lines=['# Real-weight mechanism diagnostics','',
           'This report summarizes a fixed weight-only experiment. No maps were changed after audit.','',
           '```json',json.dumps(stats,indent=2),'```','',
           '| Matrix / crop | Reference spectral / baseline | Approximate spectral / baseline | MSE spectral / baseline |',
           '|---|---:|---:|---:|']
    for m in matrices:
        for c in m['crops']:
            lines.append(f'| {m["model"]} / {m["tensor"]} / {c["position"]} | {ratio(c,"reference"):.8f} | {ratio(c,"approximate"):.8f} | {ratio(c,"mse"):.8f} |')
    lines+=['','Full matrices: lower-budget selector versus higher-budget audit. Neither is a certificate.','',
            '| Matrix | Selector / audit − 1 | Audit relative eigen residual | Baseline estimated stable rank | MSE audit spectral / baseline |',
            '|---|---:|---:|---:|---:|']
    for m in matrices:
        p=m['policies']; a=p['spectral_approximate']; b=p['baseline']
        lines.append(f'| {m["model"]} / {m["tensor"]} | {m["selection"]["history"][-1]["value"]/a["value"]-1:.8f} | {a["relative_residual"]:.8f} | {b["stable_rank_estimate"]:.2f} | {p["mse"]["value"]/b["value"]:.8f} |')
    (out/'DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()

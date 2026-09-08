"""Numerical summary only; never changes policies or thresholds."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args()
    out=Path(args.directory);report=json.loads((out/'report.json').read_text())
    assert report['status']=='complete'
    packets=report['packets'];metadata=report['metadata']
    gains=[c for c in packets if c['policies']['proposal']['ratio_to_baseline']<1-1e-8]
    summary=dict(report['summary'],
        max_proposal_output_error_reduction_percent=max(100*(1-c['policies']['proposal']['ratio_to_baseline']) for c in packets),
        metadata_total_bytes=sum(c['metadata_bytes'] for c in metadata.values()),
        metadata_build_total_seconds=sum(c['build_seconds'] for c in metadata.values()),
        metadata_percent_of_sampled_nvfp4_weight_bytes=100*sum(c['metadata_bytes'] for c in metadata.values())/sum(c['nvfp4_weight_bytes_excluding_tensor_scale'] for c in metadata.values()),
        mean_unfused_guard_milliseconds=1000*sum(c['reference_guard_seconds'] for c in packets)/len(packets),
        min_uncertainty_to_proxy_gain_on_useful_proposals=min((c['guard']['uncertainty_penalty']/(-c['guard']['proxy_change']) for c in gains),default=None),
        max_uncertainty_to_proxy_gain_on_useful_proposals=max((c['guard']['uncertainty_penalty']/(-c['guard']['proxy_change']) for c in gains),default=None))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Activation guard: bound and cost diagnostics','',
           'Unfused float64 reference implementation; timing is not a production-kernel prediction.','',
           '```json',json.dumps(summary,indent=2),'```','',
           '| Module | Residual epsilon | Metadata bytes | Weight-relative metadata | Build seconds |',
           '|---|---:|---:|---:|---:|']
    for name,m in metadata.items():
        lines.append(f'| {name} | {m["epsilon"]:.6f} | {m["metadata_bytes"]} | {100*m["metadata_bytes"]/m["nvfp4_weight_bytes_excluding_tensor_scale"]:.2f}% | {m["build_seconds"]:.3f} |')
    lines+=['','| Prompt / module | Proxy change / baseline exact loss | Bound penalty / baseline exact loss | Exact change / baseline exact loss | Guard ms |',
            '|---|---:|---:|---:|---:|']
    for c in packets:
        base=c['policies']['baseline']['output_error'];g=c['guard']
        lines.append(f'| {c["prompt"]} / {c["module"]} | {g["proxy_change"]/base:.6f} | {g["uncertainty_penalty"]/base:.6f} | {c["policies"]["proposal"]["ratio_to_baseline"]-1:.6f} | {1000*c["reference_guard_seconds"]:.3f} |')
    (out/'DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

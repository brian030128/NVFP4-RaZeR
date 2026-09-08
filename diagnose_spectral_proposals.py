"""Finite-intervention diagnosis of the frozen directional proposal rule."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from quantize.spectral_selector import candidates
from run_spectral_weights import MODELS, sha


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory'); args=ap.parse_args()
    out=Path(args.directory)
    original=json.loads((out/'report.json').read_text())
    torch.set_num_threads(4)
    records=[]
    for matrix in original['matrices']:
        path=Path(MODELS[matrix['model']]); index=path/'model.safetensors.index.json'
        shard=json.loads(index.read_text())['weight_map'][matrix['tensor']] if index.exists() else 'model.safetensors'
        with safe_open(str(path/shard),framework='pt',device='cpu') as f:
            w=f.get_tensor(matrix['tensor']).cuda()
        assert sha(w)==matrix['weight_sha256']
        b,a=candidates(w)
        assert sha(b)==matrix['candidate_sha256']['baseline']
        assert sha(a)==matrix['candidate_sha256']['alternative']
        for crop in matrix['crops']:
            row,col=crop['offset']; sl=(slice(row,row+32),slice(col,col+128))
            e=(b[sl].double()-w[sl].double()).cpu()
            delta=(a[sl].double()-b[sl].double()).cpu()
            _,s,vh=torch.linalg.svd(e,full_matrices=False)
            v=vh[0]; baseline=float(s[0]**2)
            trials=[]
            for idx in range(8):
                r,c=divmod(idx,2); ts=(slice(r*8,r*8+8),slice(c*64,c*64+64))
                trial=e.clone(); trial[ts]+=delta[ts]
                _,st,vt=torch.linalg.svd(trial,full_matrices=False)
                directional=float((trial@v).square().sum())-baseline
                actual=float(st[0]**2)-baseline
                trials.append(dict(index=idx,directional_relative_delta=directional/baseline,
                    actual_relative_delta=actual/baseline,
                    top_direction_abs_cosine=float(torch.abs(v@vt[0]))))
            best=min(trials,key=lambda t:t['directional_relative_delta'])
            records.append(dict(model=matrix['model'],tensor=matrix['tensor'],crop=crop['position'],
                baseline_spectral_squared=baseline,chosen=best,trials=trials,
                useful_toggles=sum(t['actual_relative_delta'] < -1e-8 for t in trials)))
        del w,b,a
    contradicted=[r for r in records if r['chosen']['directional_relative_delta'] < -1e-8
                 and r['chosen']['actual_relative_delta'] > 1e-8]
    summary=dict(crops=len(records),
        best_directional_proposal_harmful=len(contradicted),
        harmful_proposals_with_other_useful_toggle=sum(r['useful_toggles']>0 for r in contradicted),
        any_useful_initial_toggle=sum(r['useful_toggles']>0 for r in records),
        harmful_proposal_max_abs_cosine=max((r['chosen']['top_direction_abs_cosine'] for r in contradicted),default=None))
    (out/'proposal_diagnostics.json').write_text(json.dumps(dict(summary=summary,crops=records),indent=2)+'\n')
    lines=['# Exact finite-switch diagnosis','',
           'Diagnostic only: no maps or selection parameters were changed. Candidate hashes match the original run.','',
           'For each original crop at the all-E2M1 baseline, evaluate every single-tile switch with float64 SVD.',
           'A negative change along the original worst direction and a positive spectral change show that a new worst direction defeats the proposal.','',
           '```json',json.dumps(summary,indent=2),'```','',
           '| Model / tensor / crop | Predicted directional change | Actual spectral change | Direction cosine | Any useful toggle exists |',
           '|---|---:|---:|---:|---|']
    for r in records:
        t=r['chosen']
        lines.append(f'| {r["model"]} / {r["tensor"]} / {r["crop"]} | {100*t["directional_relative_delta"]:.4f}% | {100*t["actual_relative_delta"]:.4f}% | {t["top_direction_abs_cosine"]:.4f} | {r["useful_toggles"]>0} |')
    (out/'PROPOSAL_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

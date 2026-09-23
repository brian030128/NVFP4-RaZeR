"""CPU-only diagnosis of the rejected Mistral parent regeneration.

No acceptance tolerance or historical source is changed. A reversible source
transformation is accepted as recovered historical text ONLY on full SHA match.
"""
from common import *
import torch

OLD_QUANT_SHA = '210d182478d9bab9ce6c9b8e6d8ff831420ca83f31a2ecf170998c13ca641aca'
CURRENT_QUANT_SHA = 'c19623d141e6fb4f473efceccbda72b83cb6524b1997e834e7dbc5ae19838451'

def historical_quant_text(source):
    assert hashlib.sha256(source.encode()).hexdigest() == CURRENT_QUANT_SHA
    start = source.index('ROW_WISE =')
    end = source.index('class ActivationQuant:')
    result = source[:start] + source[end:]
    replacements = {
        ', ste=False, max_rows=4096)': ', ste=False)',
        '        self.max_rows = max_rows if kind in ROW_WISE else None\n': '',
        'chunked_rows(self.fn, x.detach(), self.max_rows) if self.max_rows else self.fn(x.detach())': 'self.fn(x.detach())',
    }
    for old, new in replacements.items():
        assert result.count(old) == 1
        result = result.replace(old, new)
    assert hashlib.sha256(result.encode()).hexdigest() == OLD_QUANT_SHA
    return result

def main():
    start = now(); torch.set_num_threads(1)
    original = PRIMARY/'runs/V61_calib_mistral7b_seed0_attempt2'
    regenerated = REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z/runs/calibration_mistral7b_attempt2_reassigned'
    a, b = [load(p/'calibration/calibration_report.json') for p in (original, regenerated)]
    source = PRIMARY/'source/NVFP4-RaZeR-main/campaign/quant.py'
    recovered = historical_quant_text(source.read_text())
    manifest = (original/'source_manifest.txt').read_text()
    assert OLD_QUANT_SHA in manifest
    losses = {}
    for objective in ['ce', 'kl']:
        delta = np.array([y[objective]-x[objective] for x,y in zip(a['fit_losses'],b['fit_losses'])])
        losses[objective] = dict(count=len(delta), equal=int((delta==0).sum()), max_abs=float(abs(delta).max()))
    rows=[]
    for N in [8,16]:
        pa=original/f'maps/mistral7b_seed0_n{N}_k3.mixfp4map'
        pb=regenerated/f'maps/mistral7b_seed0_n{N}_k3.mixfp4map'
        _,ma=mapread(pa);_,mb=mapread(pb)
        x=np.concatenate([v.ravel() for v in ma.values()]);y=np.concatenate([mb[n].ravel() for n in ma])
        rows.append(dict(N=N,**pair(x,y),xor_tiles=int((x!=y).sum()),historical_sha256=sha(pa),regenerated_sha256=sha(pb)))
    matched = [n for n,v in a['score_stream_sha256'].items() if b['score_stream_sha256'][n]==v]
    result=dict(status='rejected_historical_score_identity', checked_utc=start,
        reports=[dict(path=str(p.relative_to(REPO)),sha256=sha(p)) for p in [original/'calibration/calibration_report.json',regenerated/'calibration/calibration_report.json']],
        bf16_teacher_losses_exact=a['bf16_fit_nll']==b['bf16_fit_nll'], student_fit_losses=losses,
        matched_score_modules=matched,total_modules=len(a['score_stream_sha256']),maps=rows,
        quant_source=dict(current_sha256=sha(source),recovered_historical_sha256=hashlib.sha256(recovered.encode()).hexdigest(),
            historical_manifest_sha256=sha(original/'source_manifest.txt'),
            finding='Only token-row chunking added. At 512 rows, fn(x) is used; original STE expression and dtype guard are unchanged. No evidence of changed STE arithmetic.'),
        known_launch_differences=['Historical --raw sample --subset-moments; regeneration --raw none without subset moments.',
            'Regeneration adds CPU parent-moment observer; historical launch was containerized, current launch is direct using the same pinned environment.'],
        inference='Identical forward losses but differing backward score streams localize the mismatch beyond the checked forward losses. Allocation/backend backward nondeterminism is a hypothesis, not an established cause.',
        acceptance='No regenerated parents or maps admitted. Do not relax score identity, report new moments as historical, or set missing covariance to zero.',
        final_retry_requirement='Restore hash-verified historical quant text and original raw/subset allocation options in a new isolated adapter; retain exact score/N8/N16 gate. At most one final reasoned calibration retry remains.')
    jsonout(OUT/'results/MISTRAL_CALIBRATION_IDENTITY_DIAGNOSTIC.json',result)
    log('T3-calibration-diagnostic','python scripts/calibration_identity_diagnostic.py',start,[OUT/'results/MISTRAL_CALIBRATION_IDENTITY_DIAGNOSTIC.json'])
    print({k:result[k] for k in ['status','bf16_teacher_losses_exact','student_fit_losses','matched_score_modules','maps']})

if __name__=='__main__':main()

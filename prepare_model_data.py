"""Local calibration and development data for run_multiround.py on the N16K64 models, verified by hash.

Writes, per model, the layout prepare_multiround_data.py writes for Llama-3.1-8B:

    <out>/<model>/calibration/report.json
    <out>/<model>/<development name>/{report.json, fresh.pt, fresh_manifest.json}   (three sets of 64)

Fit set: 128 = 64 OpenWebMath + 64 CodeParrot windows of 512 tokens, run_math_code_calibration.math_code_data's
spec (per source: repo, revision, path, documents with offsets, token_sha256).
  * qwen4b, qwen27b: the archived calibration record (results/math_code_adaptive/calibration_*). Its windows
    are reproduced with math_code_data, which checks every token hash; so are its per-matrix weight hashes,
    when run_multiround.py loads the model.
  * mistral7b: the N16K64 campaign's calibration record (research/n16k64/campaigns/primary/
    calibration_manifests/mistral7b_seed0.json), reproduced the same way.
  * phi4: no record exists. It is drawn with the rule that produced Llama's fit set. That rule is
    run_pooled_scale.shared_data's math/code part: file order, skip documents shorter than 512 tokens,
    offset random.Random(20260926).randrange, at the pinned dataset files. The rule runs through
    campaign.data.builder_seed0, which must first reproduce the Llama, Qwen3-4B and Mistral records
    exactly (document, offset and token hash).
Development set: 192 = three draws of 32 math + 32 code windows of 512 tokens.
  * All models except qwen27b: run_fine_row_research.fresh_data, the code that drew Llama's three
    development sets. Its rule: seed 20260919, file order, skip documents in the fit set or in
    `excluded` or shorter than 512 tokens, skip windows whose tokens equal a fit window. The three
    draws run one after another. Each excludes this model's C4 evaluation documents (Llama's excluded
    its published C4 documents) and the documents of the draws before it.
  * qwen27b: main's three development sets are reproduced instead. Their fresh.pt sha256 is recorded in
    results/mixfp4_potential/qwen27b/multiround_kl_8x64_final.json. The pilot drew six sets in sequence
    with the same code. The first (fine_rows_v2) excluded the published C4 documents; each later one
    also excluded every set before it (fisher_subset_validate_v2_confirm lists the five before it).
    The six draws are regenerated in that order, and each recorded hash must be matched.
Every fresh.pt is accepted only by its hash: recorded (qwen27b), or computed and recorded here.

python prepare_model_data.py --out DATA_ROOT --model qwen4b [--model ...]   (CPU; needs the datasets)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from run_baseline_protocol_audit import data as evaluation_data
from run_c4_frozen import digest_file
from run_conditional_format import sha
from run_fine_row_research import fresh_data
from run_math_code_calibration import math_code_data

REPO = Path(__file__).resolve().parent
MODELS = {
    'llama8b': dict(repo='meta-llama/Llama-3.1-8B', revision='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b',
                    record='results/math_code_adaptive/calibration_333779_llama8b/report.json'),
    'qwen4b': dict(repo='Qwen/Qwen3-4B', revision='1cfa9a7208912126459214e8b04321603b3df60c',
                   record='results/math_code_adaptive/calibration_333779_qwen4b/report.json'),
    'mistral7b': dict(repo='mistralai/Mistral-7B-v0.3', revision='caa1feb0e54d415e2df31207e5f4e273e33509b1',
                      record='research/n16k64/campaigns/primary/calibration_manifests/mistral7b_seed0.json'),
    'phi4': dict(repo='microsoft/phi-4', revision='2db69c1c3e91a05d2c64a3185acfbaf36f744e25', record=None),
    'qwen27b': dict(repo='Qwen/Qwen3.8-27B', revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
                    record='results/math_code_adaptive/calibration_333787_qwen27b/report.json'),
}
DEVELOPMENT = ('fresh_dev1', 'fresh_dev2', 'fresh_dev3')
QWEN27B_PUBLISHED = 'results/kse_paper/job_336969/qwen27b/report.json'
QWEN27B_RECORD = 'results/mixfp4_potential/qwen27b/multiround_kl_8x64_final.json'
QWEN27B_ORDER_AFTER_BASE = 5          # fisher_subset_validate_v2_confirm excluded five earlier draws


def snapshot(key):
    spec = MODELS[key]
    hub = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ['HF_HOME']) / 'hub'))
    path = hub / ('models--' + spec['repo'].replace('/', '--')) / 'snapshots' / spec['revision']
    assert path.is_dir(), path
    return path


def fit_spec(meta):
    return {s: {k: meta[s][k] for k in ('repo', 'revision', 'path', 'documents', 'token_sha256')}
            for s in ('math', 'code')}


def same_fit(a, b):
    return all([(d['document_sha256'], d['offset']) for d in a[s]['documents']] ==
               [(d['document_sha256'], d['offset']) for d in b[s]['documents']]
               and a[s]['token_sha256'] == b[s]['token_sha256']
               and all(a[s][k] == b[s][k] for k in ('repo', 'revision', 'path')) for s in ('math', 'code'))


def builder_rule_checks(builder):
    """campaign.data.builder_seed0 must reproduce every recorded fit set drawn with this rule."""
    out = {}
    for key in ('llama8b', 'qwen4b', 'mistral7b'):
        record = json.loads((REPO / MODELS[key]['record']).read_text())
        recorded = record['fit'] if 'fit' in record else record
        _, built = builder(AutoTokenizer.from_pretrained(str(snapshot(key))))
        out[key] = same_fit(fit_spec(built), fit_spec(recorded))
        print(f'BUILDER RULE {key}: {"reproduces the record" if out[key] else "DIFFERS"}', flush=True)
    assert all(out.values()), out
    return out


def matrices_of(key, source):
    """Per quantized matrix (every text nn.Linear except the output head): shape and weight sha256."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16, device_map='cpu')
    head = model.get_output_embeddings()
    out = {n: dict(shape=list(m.weight.shape), source_sha256=sha(m.weight))
           for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not head}
    del model
    return out


def write_draw(target, records, rule, extra):
    target.mkdir(parents=True, exist_ok=True)
    torch.save(records, target / 'fresh.pt')
    manifest = dict(records=[{k: v for k, v in r.items() if k != 'ids'} for r in records], windows=len(records),
                    window_tokens=512, seed=20260919, excluded_calibration_and_published_c4=True, **extra)
    (target / 'fresh_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = dict(status='complete', fresh_sha256=digest_file(target / 'fresh.pt'),
                  fresh_manifest_sha256=digest_file(target / 'fresh_manifest.json'), rule=rule,
                  sequence_sources=[r['source'] for r in records])
    (target / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def prepare(key, out, builder):
    spec = MODELS[key]
    source = str(snapshot(key))
    tok = AutoTokenizer.from_pretrained(source)
    summary = dict(model=key, source=source, revision=spec['revision'])
    if spec['record'] is not None:
        record = json.loads((REPO / spec['record']).read_text())
        recorded = fit_spec(record['fit'] if 'fit' in record else record)
        summary['fit_from'] = dict(path=spec['record'], sha256=digest_file(REPO / spec['record']))
    else:
        _, built = builder(tok)
        recorded = fit_spec(built)
        summary['fit_from'] = dict(rule='campaign.data.builder_seed0 (run_pooled_scale.shared_data math/code rule)')
    # Reproduce the 128 windows exactly as run_multiround.py will: every token hash is asserted.
    _, fit_meta = math_code_data(tok, recorded)
    assert same_fit(fit_spec(fit_meta), recorded)
    summary['fit_windows_reproduced'] = 128
    if spec['record'] is not None and 'matrices' in record:
        matrices, versions = record['matrices'], dict(torch_version=record['torch_version'],
                                                        transformers_version=record['transformers_version'])
        summary['matrices_from'] = spec['record']
    else:
        matrices = matrices_of(key, source)
        versions = dict(torch_version=torch.__version__, transformers_version=transformers.__version__)
        summary['matrices_from'] = 'computed from the pinned snapshot (sha256 of each loaded bf16 weight)'
    prior = dict(model=key, source=source, revision=spec['revision'], **versions, fit=recorded, matrices=matrices,
                 local_preparation=dict(script='prepare_model_data.py', fit_from=summary['fit_from'],
                                        matrices_from=summary['matrices_from']))
    calibration = out / key / 'calibration'
    calibration.mkdir(parents=True, exist_ok=True)
    (calibration / 'report.json').write_text(json.dumps(prior, indent=2) + '\n')
    summary['calibration_report_sha256'] = digest_file(calibration / 'report.json')
    summary['matrices'] = len(matrices)
    summary['development'] = {}
    if key == 'qwen27b':
        published = json.loads((REPO / QWEN27B_PUBLISHED).read_text())
        c4 = [d['document_sha256'] for d in published['data']['c4_paper']['documents']]
        wanted = {d['name'].rsplit('/', 1)[1]: d['sha256']
                  for d in json.loads((REPO / QWEN27B_RECORD).read_text())['development']}
        drawn, found = [], {}
        for index in range(QWEN27B_ORDER_AFTER_BASE + 1):         # the base draw, then five confirmations
            excluded = c4 + [r['document_sha256'] for draw in drawn for r in draw]
            records = fresh_data(tok, prior, excluded)
            drawn.append(records)
            scratch = out / key / f'_draw{index}'
            report = write_draw(scratch, records, 'fresh_data; published C4 + all earlier draws excluded',
                                dict(draw_index=index, excluded_earlier_draws=index))
            match = [name for name, digest in wanted.items() if digest == report['fresh_sha256']]
            print(f'QWEN27B draw {index}: fresh.pt {report["fresh_sha256"][:12]} '
                  f'{"= " + match[0] if match else "(no recorded development set)"}', flush=True)
            for name in match:
                found[name] = index
                (out / key / name).mkdir(parents=True, exist_ok=True)
                for f in ('fresh.pt', 'fresh_manifest.json', 'report.json'):
                    (out / key / name / f).write_bytes((scratch / f).read_bytes())
        summary['development'] = {name: dict(expected=digest, draw_index=found.get(name),
                                             match=name in found) for name, digest in wanted.items()}
        if set(found) != set(wanted):
            raise SystemExit(f'qwen27b development sets not reproduced: {sorted(set(wanted) - set(found))}')
    else:
        _, evaluation = evaluation_data(tok, prior, 2048)
        c4 = [d['document_sha256'] for d in evaluation['c4_paper']['documents']]
        summary['c4_evaluation_documents_excluded'] = len(set(c4))
        drawn = []
        for index, name in enumerate(DEVELOPMENT):
            excluded = c4 + [r['document_sha256'] for draw in drawn for r in draw]
            records = fresh_data(tok, prior, excluded)
            assert len(records) == 64 and not {r['document_sha256'] for r in records} & set(excluded)
            drawn.append(records)
            report = write_draw(out / key / name, records,
                                'run_fine_row_research.fresh_data (Llama development rule); this model\'s C4 '
                                'evaluation documents and all earlier draws excluded',
                                dict(draw_index=index, excluded_earlier_draws=list(DEVELOPMENT[:index])))
            summary['development'][name] = dict(fresh_sha256=report['fresh_sha256'],
                                                fresh_manifest_sha256=report['fresh_manifest_sha256'])
            print(f'{key} {name}: fresh.pt {report["fresh_sha256"]}', flush=True)
        documents = [r['document_sha256'] for draw in drawn for r in draw]
        assert len(set(documents)) == 192
    (out / key / 'prepare_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', action='append', choices=[k for k in MODELS if k != 'llama8b'], required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    sys.path.insert(0, str(REPO / 'research/n16k64/software/primary'))
    from campaign.data import builder_seed0
    checks = builder_rule_checks(builder_seed0) if 'phi4' in args.model else None
    for key in args.model:
        summary = prepare(key, args.out, builder_seed0)
        if checks is not None:
            summary['builder_rule_reproduces'] = checks
            (args.out / key / 'prepare_summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()

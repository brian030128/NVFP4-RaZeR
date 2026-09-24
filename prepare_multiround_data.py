"""Rebuild the Llama-3.1-8B inputs of run_multiround.py outside the cluster, verified by hash.

run_multiround.py reads a calibration report (fit-set spec, model source/revision,
per-matrix weight hashes) and three development directories (report.json plus
fresh.pt, 192 math/code documents), all under /work on the cluster. This script
writes local copies with the same layout under --out:

    <out>/llama8b/calibration/report.json
    <out>/llama8b/{confirmation,gate_up_confirm,tile_refine_confirm}/{report.json,fresh.pt,fresh_manifest.json}

fresh.pt is regenerated from the published fresh_manifest.json (document hash,
token offset, token hash of every 512-token window): the same dataset files are
streamed, the recorded window is cut from the tokenized document, and the records
are built exactly as run_fine_row_research.fresh_data builds them before
torch.save. The file is accepted only if its sha256 equals the fresh_sha256 in the
published report.json, which run_multiround.py checks again when it loads it.

The calibration report itself (sha256 47e0d54c...) is not in the repository. The
fields run_multiround.py reads are taken from the historical calibration report
results/math_code_adaptive/calibration_333779_llama8b/report.json, whose fit set
is the one the transfer calibration used (checked below against the fit/election
records of results/task_reorder/llama_diagnosis_20260920/data.json, which cite
calibration sha256 47e0d54c...). run_multiround.py re-verifies every fit token
hash and every weight hash on use. The only field that changes is `source`, the
local path of the same pinned Hugging Face snapshot.

CPU only. Needs the two fit data files (OpenWebMath, CodeParrot) in the HF cache
or network access.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer

from run_c4_frozen import digest_file
from run_conditional_format import sha

HISTORICAL = Path('results/math_code_adaptive/calibration_333779_llama8b/report.json')
DIAGNOSIS = Path('results/task_reorder/llama_diagnosis_20260920/data.json')
TRANSFER_CALIBRATION = dict(path='/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration/report.json',
                            sha256='47e0d54c6ce647f68e076e46356b9d812ceb59094db8e92c935d925117583d32')
# Local directory name -> published record of the cluster directory of the same name.
DEVELOPMENT = {'confirmation': Path('results/task_reorder/transfer_20260920/llama_confirmation'),
               'gate_up_confirm': Path('results/task_reorder/transfer_20260920/renewed_llama/gate_up_confirm'),
               'tile_refine_confirm': Path('results/task_reorder/transfer_20260920/renewed_llama/tile_refine_confirm')}
REPO_ID = 'meta-llama/Llama-3.1-8B'


def calibration_prior(historical):
    """The fields of the transfer calibration report that run_multiround.py reads."""
    diagnosis = json.loads(DIAGNOSIS.read_text())
    fit = {}
    for source, meta in historical['fit'].items():
        for doc, token in zip(meta['documents'], meta['token_sha256']):
            fit[doc['document_sha256']] = (source, doc['offset'], token)
    checked = 0
    for r in diagnosis['records']:
        if r['split'] in ('fit', 'election'):
            assert fit[r['document_sha256']] == (r['source'], r['offset'], r['token_sha256']), r
            checked += 1
    assert checked == 64
    return checked


def stream_windows(meta, source, wanted, tok):
    """Tokenize the wanted documents of one fit source exactly as fresh_data does."""
    stream = load_dataset(meta['repo'], revision=meta['revision'],
                          data_files={'train': meta['path']}, split='train', streaming=True)
    found, scanned = {}, 0
    for row in stream:
        scanned += 1
        content = row['text' if source == 'math' else 'content']
        digest = hashlib.sha256(content.encode()).hexdigest()
        if digest not in wanted or digest in found:
            continue
        found[digest] = tok(content, return_tensors='pt').input_ids
        if len(found) == len(wanted):
            break
    assert set(found) == set(wanted), f'{source}: {len(set(wanted) - set(found))} documents not found'
    return found, scanned


def rebuild_fresh(manifest, prior, tok):
    """Records in fresh_data's order and object layout: all math, then all code."""
    records, scanned = [], {}
    for source in ('math', 'code'):
        rows = [r for r in manifest['records'] if r['source'] == source]
        found, scanned[source] = stream_windows(prior['fit'][source], source,
                                                {r['document_sha256'] for r in rows}, tok)
        for r in rows:
            digest, offset = r['document_sha256'], r['offset']
            ids = found[digest]
            assert ids.shape[1] >= 512 and 0 <= offset <= ids.shape[1] - 512
            ids = ids[:, offset:offset + 512].clone()
            token_hash = sha(ids)
            assert token_hash == r['token_sha256'], (source, digest)
            records.append(dict(source=source, document_sha256=digest, offset=offset,
                                token_sha256=token_hash, ids=ids))
    assert [(r['source'], r['document_sha256']) for r in records] == \
        [(r['source'], r['document_sha256']) for r in manifest['records']], 'manifest is not math-then-code'
    return records, scanned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True, help='Data root passed to run_multiround.py --data-root')
    args = ap.parse_args()
    historical = json.loads(HISTORICAL.read_text())
    revision = historical['revision']
    checked = calibration_prior(historical)
    # The cached snapshot holds only the files the model needs, so locate it through one of them.
    config = try_to_load_from_cache(REPO_ID, 'config.json', revision=revision)
    assert isinstance(config, str), f'{REPO_ID}@{revision} is not in the Hugging Face cache'
    source = str(Path(config).parent)
    assert Path(source).name == revision
    prior = dict(model='llama8b', source=source, revision=revision,
                 torch_version=historical['torch_version'],
                 transformers_version=historical['transformers_version'],
                 fit=historical['fit'], matrices=historical['matrices'],
                 local_reconstruction=dict(
                     replaces=TRANSFER_CALIBRATION,
                     original_available=False,
                     fields_from=str(HISTORICAL), fields_from_sha256=digest_file(HISTORICAL),
                     fields=['revision', 'torch_version', 'transformers_version', 'fit', 'matrices'],
                     changed_fields={'source': dict(original=historical['source'], local=source)},
                     fit_evidence=dict(file=str(DIAGNOSIS), sha256=digest_file(DIAGNOSIS), matching_fit_records=checked,
                                       note='fit/election records drawn from the transfer calibration (sha256 '
                                            + TRANSFER_CALIBRATION['sha256'][:8] + '...) match the historical fit set'),
                     verified_on_use_by_run_multiround=['128 fit token_sha256 (math_code_data)',
                                                        '224 matrix source_sha256']))
    calibration = args.out / 'llama8b' / 'calibration'
    calibration.mkdir(parents=True, exist_ok=True)
    (calibration / 'report.json').write_text(json.dumps(prior, indent=2) + '\n')
    tok = AutoTokenizer.from_pretrained(source, revision=revision)
    summary = dict(calibration=dict(path=str(calibration / 'report.json'),
                                    sha256=digest_file(calibration / 'report.json')), development={})
    for name, published in DEVELOPMENT.items():
        report = json.loads((published / 'report.json').read_text())
        manifest = json.loads((published / 'fresh_manifest.json').read_text())
        assert report['status'] == 'complete' and len(manifest['records']) == 64
        assert digest_file(published / 'fresh_manifest.json') == report['fresh_manifest_sha256'], name
        records, scanned = rebuild_fresh(manifest, prior, tok)
        target = args.out / 'llama8b' / name
        target.mkdir(parents=True, exist_ok=True)
        torch.save(records, target / 'fresh.pt')
        digest = digest_file(target / 'fresh.pt')
        shutil.copyfile(published / 'report.json', target / 'report.json')
        shutil.copyfile(published / 'fresh_manifest.json', target / 'fresh_manifest.json')
        summary['development'][name] = dict(published=str(published), expected=report['fresh_sha256'], actual=digest,
                                             match=digest == report['fresh_sha256'], rows_scanned=scanned)
        print(name, 'fresh.pt', digest, 'MATCH' if digest == report['fresh_sha256'] else 'MISMATCH', flush=True)
    (args.out / 'prepare_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not all(v['match'] for v in summary['development'].values()):
        raise SystemExit('fresh.pt hash mismatch: do not use these files')


if __name__ == '__main__':
    main()

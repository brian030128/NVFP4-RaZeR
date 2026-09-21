"""V53 calibration/evaluation data-overlap audit (CPU).

Exact: SHA-256 of NFKC/lower/whitespace-normalized full documents.
Near-duplicate: word 13-gram shingles (GPT-3 style) and 8-gram shingles; a calibration document is flagged
when it shares any 13-gram with an evaluation text; overlap fractions are reported per pair of corpora.
Also reports C4 repeated-document structure and WikiText window/article structure.
"""
import gzip
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from campaign import data as D
from campaign import models as MOD
from campaign import runtime

WORD = re.compile(r"\w+", re.UNICODE)


def norm(t):
    return ' '.join(unicodedata.normalize('NFKC', t).lower().split())


def shingles(t, n):
    w = WORD.findall(unicodedata.normalize('NFKC', t).lower())
    return {hashlib.blake2b(' '.join(w[i:i + n]).encode(), digest_size=8).digest() for i in range(max(0, len(w) - n + 1))}


def main():
    out = runtime.out_dir('overlap')
    rep = dict(method=__doc__, corpora={}, pairs={}, flagged={})
    # ---------- calibration documents (all draws and panels) from their manifests
    texts_by_hash = {}
    for dom in ('math', 'code'):
        for t in D.source_texts(dom):
            texts_by_hash[D.text_sha(t)] = t
    for dom in ('arxiv', 'govreport'):
        for t in D.heldout_texts(dom):
            texts_by_hash[D.text_sha(t)] = t
    calib = defaultdict(set)
    runs = Path(os.environ['CAMPAIGN_ROOT']) / 'runs'
    for mf in list(runs.glob('*/calibration/calibration_manifest.json')) + list(runs.glob('V14_smoke_*/smoke/calibration_manifest_seed0.json')):
        meta = json.loads(mf.read_text())
        for dom, m in meta.items():
            if isinstance(m, dict) and 'documents' in m:
                for d in m['documents']:
                    calib[f'{dom}'].add(d['document_sha256'])
    for key in ('llama8b', 'qwen4b', 'qwen27b'):
        fit, _ = D.archived_manifest(key)
        for dom in ('math', 'code'):
            calib[dom] |= {d['document_sha256'] for d in fit[dom]['documents']}
    calib_docs = {h: texts_by_hash[h] for s in calib.values() for h in s if h in texts_by_hash}
    rep['corpora']['calibration'] = dict(documents=len(calib_docs), by_slot={k: len(v) for k, v in calib.items()},
                                         missing_text=[h for s in calib.values() for h in s if h not in texts_by_hash])
    # ---------- evaluation corpora
    import pyarrow.parquet as pq
    wiki_lines = D.parquet_column(D.hub_file(*D.WIKI), 'text')
    articles, cur = [], []
    for ln in wiki_lines:
        if D._ARTICLE.match(ln) and cur:
            articles.append(''.join(cur)); cur = []
        cur.append(ln)
    if cur:
        articles.append(''.join(cur))
    c4_texts = D.jsonl_gz_column(D.hub_file(*D.C4), 'text')
    tok = MOD.load_tokenizer('llama8b')
    _, cmeta = D.c4_windows(tok)
    c4_used = sorted({d['index'] for d in cmeta['documents']})
    mult = Counter(d['document_sha256'] for d in cmeta['documents'])
    rep['c4_repeated_documents'] = dict(draws=len(cmeta['documents']), unique=len(mult), multiplicity_histogram=dict(Counter(mult.values())),
                                        note='tokenizer-dependent draw; llama8b tokenizer shown, others reported in PPL window manifests')
    pg = []
    for f in sorted(D.hub_snapshot(*D.PG19).rglob('*.parquet')):
        pg.extend(D.parquet_column(f, 'text'))
    pg = pg[:40]
    evals = dict(wikitext_articles=articles, c4_used_documents=[c4_texts[i] for i in c4_used], pg19_first_books=pg)
    for dom in ('math_eval', 'code_eval'):  # A03 held-out in-domain evaluation documents (V52), by document hash
        want = set()
        for wf in runs.glob(f'V52_ppl_crossdomain_*/ppl/windows_{dom}.json'):
            want |= {d['document_sha256'] for d in json.loads(wf.read_text())['documents']}
        if want:
            evals[f'heldout_{dom}_documents'] = [texts_by_hash[h] for h in sorted(want) if h in texts_by_hash]
            rep.setdefault('heldout_domain_eval', {})[dom] = dict(documents=len(want), exact_hash_in_calibration=len(want & set(calib_docs)))
    task_root = Path(os.environ['HF_DATASETS_CACHE'])
    try:
        from lm_eval.tasks import TaskManager, get_task_dict
        tm = TaskManager()
        for t in ('arc_easy', 'arc_challenge', 'hellaswag', 'openbookqa', 'boolq', 'winogrande', 'piqa', 'mmlu', 'gsm8k'):
            td = get_task_dict([t], tm)
            docs = []
            def walk(o):
                if isinstance(o, dict):
                    for v in o.values():
                        walk(v)
                elif hasattr(o, 'config'):
                    ds = list(o.test_docs() if o.has_test_docs() else o.validation_docs())
                    docs.extend(json.dumps(d, sort_keys=True, default=str) for d in ds)
            walk(td)
            evals[f'task_{t}'] = docs
    except Exception as exc:
        rep['task_load_error'] = repr(exc)
    for name, texts in evals.items():
        rep['corpora'][name] = dict(documents=len(texts))
    # ---------- exact normalized hash overlap
    calib_norm = {hashlib.sha256(norm(t).encode()).hexdigest(): h for h, t in calib_docs.items()}
    for name, texts in evals.items():
        en = {hashlib.sha256(norm(t).encode()).hexdigest() for t in texts}
        rep['pairs'].setdefault(name, {})['exact_normalized_overlap'] = len(en & set(calib_norm))
    # ---------- n-gram overlap
    for n in (13, 8):
        cal_sh = {h: shingles(t, n) for h, t in calib_docs.items()}
        for name, texts in evals.items():
            ev = set()
            for t in texts:
                ev |= shingles(t, n)
            flagged = {h: len(s & ev) for h, s in cal_sh.items() if s & ev}
            total = sum(len(s) for s in cal_sh.values())
            hits = sum(flagged.values())
            rep['pairs'][name][f'{n}gram'] = dict(calibration_docs_with_overlap=len(flagged), shared_shingles=hits,
                                                  calibration_shingle_fraction=(hits / total if total else 0.0), eval_shingles=len(ev))
            if n == 13 and flagged:
                rep['flagged'].setdefault(name, dict(sorted(flagged.items(), key=lambda kv: -kv[1])[:20]))
    rep['wikitext_structure'] = dict(articles=len(articles), note='windows are 2048 tokens of the joined test split and may span articles; clusters use the first article')
    rep['limits'] = ['pretraining-data overlap of the evaluated models is not measurable here (closed or unindexed corpora)',
                     'n-gram screening uses word shingles on full documents; paraphrase-level contamination is not detected']
    runtime.atomic_json(out / 'DATA_OVERLAP_REPORT.json', rep)
    src = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id='data-audit', protocol_freeze_sha256=os.environ.get('FREEZE_SHA256', ''),
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=rep['pairs']),
        policies=[], results=dict(raw_outputs=[str(out / 'DATA_OVERLAP_REPORT.json')], summary=rep['pairs'], uncertainty={},
                                  attempted_endpoints=list(evals), missing_endpoints=([] if 'task_load_error' not in rep else ['lm-eval tasks'])),
        logs=[], failures=([] if 'task_load_error' not in rep else [rep['task_load_error']])))
    print(json.dumps(rep['pairs'], indent=1)[:4000])


if __name__ == '__main__':
    main()

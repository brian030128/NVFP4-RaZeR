"""Calibration record + development set for NVIDIA-Nemotron-Nano-9B-v2 (hybrid Mamba-2/attention).

Nemotron has its own tokenizer, so the Llama/Qwen token windows do not apply. This
re-derives them with the rule used for the existing math/code calibration sets:
stream the same pinned OpenWebMath / CodeParrot shards, skip documents shorter than
512 tokens, one seeded random 512-token window per document, the first 64 per source.
The next 96 + 96 qualifying documents (disjoint from calibration) form the 192
development windows (monitor only). Also records the linear-layer weight hashes and
two diagnostics that decide the training configuration: whether a batched forward
equals batch 1, and the forward+backward time per 512-token sequence.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SOURCE = 'nvidia/NVIDIA-Nemotron-Nano-9B-v2'
REVISION = '6533e8de2c68e4536bf7c411d7a3ce5734111476'
# Same pinned shards as the Qwen3.8-27B calibration record.
STREAMS = {'math': dict(repo='open-web-math/open-web-math', revision='fde8ef8de2300f5e778f56261843dab89f230815',
                        path='data/train-00000-of-00114-5a023365406cb9c4.parquet', field='text'),
           'code': dict(repo='codeparrot/codeparrot-clean', revision='35a59fb025bc0a102f7d96eac09d145b896d487b',
                        path='file-000000000001.json.gz', field='content')}
SEED = 20260928


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def windows(tok, source, count, skip, seed):
    s = STREAMS[source]
    stream = load_dataset(s['repo'], revision=s['revision'], data_files={'train': s['path']}, split='train', streaming=True)
    rng, seen, out = random.Random(seed), set(skip), []
    for row in stream:
        text = row[s['field']]
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        ids = tok(text, return_tensors='pt').input_ids
        if ids.shape[1] < 512:
            continue
        offset = rng.randrange(ids.shape[1] - 512 + 1)
        out.append(dict(document_sha256=digest, offset=offset, ids=ids[:, offset:offset + 512].clone(), source=source))
        if len(out) == count:
            return out
    raise RuntimeError(f'{source}: only {len(out)} documents')


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('/work/u4320956/mixfp4_potential'))
    args = ap.parse_args()
    cal_dir, dev_dir = args.out / 'nemotron9b_calibration', args.out / 'nemotron9b_development'
    cal_dir.mkdir(parents=True, exist_ok=False)
    dev_dir.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(SOURCE, revision=REVISION)
    fit, dev = {}, []
    for source in ('math', 'code'):
        cal = windows(tok, source, 64, (), SEED)
        s = STREAMS[source]
        fit[source] = dict(repo=s['repo'], revision=s['revision'], path=s['path'],
                           documents=[dict(document_sha256=r['document_sha256'], offset=r['offset']) for r in cal],
                           token_sha256=[sha(r['ids']) for r in cal])
        dev.extend(windows(tok, source, 96, {r['document_sha256'] for r in cal}, SEED + 1))
    assert len({r['document_sha256'] for r in dev} | {d['document_sha256'] for f in fit.values() for d in f['documents']}) == 320
    torch.save(dev, dev_dir / 'fresh.pt')
    digest = hashlib.sha256((dev_dir / 'fresh.pt').read_bytes()).hexdigest()
    (dev_dir / 'report.json').write_text(json.dumps(dict(
        status='complete', job_id=os.environ['SLURM_JOB_ID'], source=SOURCE, revision=REVISION, seed=SEED + 1,
        windows=len(dev), window_tokens=512, fresh_sha256=digest,
        excluded='the 128 calibration documents', streams=STREAMS), indent=2) + '\n')

    model = AutoModelForCausalLM.from_pretrained(SOURCE, revision=REVISION, dtype=torch.bfloat16, device_map='cuda')
    model.eval().requires_grad_(False)
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    kinds = {}
    for n in modules:
        key = n.rsplit('.', 1)[-1]
        kinds[key] = kinds.get(key, 0) + 1
    print('LINEARS', len(modules), kinds, flush=True)
    print('PARAMS sample', [n for n, _ in model.named_parameters()][:12], flush=True)
    print('MIXER impl', {type(m).__name__ for m in model.modules() if 'Mixer' in type(m).__name__}, flush=True)

    # Diagnostics: batch equivalence and speed on 8 calibration windows.
    ids = torch.cat([fit_ids for fit_ids in (r['ids'] for r in windows(tok, 'math', 8, (), SEED))]).cuda()
    single = torch.cat([model(input_ids=ids[i:i + 1], use_cache=False).logits.float().log_softmax(-1) for i in range(8)])
    batched = model(input_ids=ids, use_cache=False).logits.float().log_softmax(-1)
    diag = dict(batch8_vs_batch1_max_abs_logprob=float((single - batched).abs().max()),
                batch8_vs_batch1_mean_abs_logprob=float((single - batched).abs().mean()),
                batch8_equal_bitwise=bool(torch.equal(single, batched)))
    del single, batched
    # Batch 1 only: the pure-torch Mamba-2 scan (no mamba_ssm kernels) OOMs on a batch-8 backward.
    for b in (1,):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.enable_grad():
            for i in range(0, 8, b):
                e = model.get_input_embeddings()(ids[i:i + b]).detach().requires_grad_()
                model(inputs_embeds=e, use_cache=False).logits.float().log_softmax(-1).mean().backward()
        torch.cuda.synchronize()
        diag[f'fwd_bwd_seconds_per_seq_batch{b}'] = (time.time() - t0) / 8
    diag['gpu_peak_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    print('DIAG ' + json.dumps(diag), flush=True)

    prior = dict(status='complete', model='nemotron9b', source=SOURCE, revision=REVISION,
                 job_id=os.environ['SLURM_JOB_ID'], transformers_version=transformers.__version__,
                 torch_version=torch.__version__, uses_c4_calibration=False, uses_wiki_calibration=False,
                 selection_rule='first 64 documents >= 512 tokens per stream, one random 512-token window each',
                 seed=SEED, fit=fit, diagnostics=diag,
                 matrices={n: dict(shape=list(m.weight.shape), source_sha256=sha(m.weight)) for n, m in modules.items()})
    (cal_dir / 'report.json').write_text(json.dumps(prior, indent=2) + '\n')
    print(f'PRIOR {cal_dir} matrices={len(modules)} dev={len(dev)}', flush=True)


if __name__ == '__main__':
    main()

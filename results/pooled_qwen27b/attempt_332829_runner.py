"""Unchanged pooled tile rule on native hybrid Qwen3.8-27B; run via Slurm."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit all compute through Slurm')
for key, sub in [('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch'),
                 ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
    os.environ.setdefault(key, os.path.join(os.environ['HF_HOME'], sub))
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from probe_qwen38 import text_modules
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
from quantize.causal_four_over_six import quantize_rows
from quantize.modulewise_selection import modulewise_maps
from quantize.interacting_format import apply_mask
from run_c4_frozen import heldout_data, digest_file
from run_conditional_format import sha, save
from run_conditional_model import paired


def calibration(tok):
    prior = json.loads(Path('results/pooled_scale/model_332389_qwen4b/report.json').read_text())
    batches, metadata, seen = {}, {}, set()
    for source in ('web', 'math', 'code'):
        meta = prior['fit'][source]
        repo = meta.get('repo', 'allenai/c4')
        stream = load_dataset(repo, revision=meta['revision'], data_files={'train': meta['path']},
                              split='train', streaming=True)
        rng = random.Random(20260920 if source == 'web' else 20260926)
        bs, docs = [], []
        for row in stream:
            text = row['content' if source == 'code' else 'text']
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen: continue
            seen.add(digest)
            ids = tok(text, return_tensors='pt').input_ids
            if ids.shape[1] < 512: continue
            offset = rng.randrange(ids.shape[1] - 512 + 1)
            bs.append(ids[:, offset:offset + 512].clone())
            docs.append(dict(document_sha256=digest, offset=offset))
            if len(bs) == 64: break
        assert len(bs) == 64
        batches[source] = bs
        metadata[source] = dict(repo=repo, revision=meta['revision'], path=meta['path'],
                                documents=docs, token_sha256=[sha(b) for b in bs])
    return batches, metadata


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); args = ap.parse_args()
    torch.set_num_threads(12); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260928)
    root = Path(__file__).resolve().parent; out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    probe = json.loads((root / 'results/task_sensitivity_qwen38_probe/report.json').read_text())
    assert probe['complete'] and transformers.__version__ == probe['transformers']
    files = ['run_pooled_qwen27b.py', 'quantize/modulewise_selection.py', 'quantize/relinearized_format.py',
             'quantize/causal_four_over_six.py', 'quantize/quantizer.py', 'quantize/interacting_format.py',
             'run_c4_frozen.py', 'probe_qwen38.py', 'results/pooled_qwen27b/PROTOCOL.md']
    r = dict(status='running', model='qwen27b', source=probe['model'], revision=probe['model_commit'],
             job_id=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__, transformers_version=transformers.__version__,
             source_sha256={f: digest_file(root / f) for f in files}, matrices={}, evaluation={}, suffix_intervention={})
    save(out, r)
    start = time.perf_counter()
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        r['source'], revision=r['revision'], dtype=torch.bfloat16, attn_implementation='eager',
        device_map='balanced', max_memory={0: '65GiB', 1: '65GiB'}, output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
    model.eval().requires_grad_(False)
    assert all(m.norm.activation in ('silu', 'swish') for m in model.modules()
               if m.__class__.__name__ == 'Qwen3_5GatedDeltaNet')
    r['device_map'] = model.hf_device_map; r['load_seconds'] = time.perf_counter() - start
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    modules = text_modules(model); names = list(modules)
    assert {n: list(m.weight.shape) for n, m in modules.items()} == probe['quantized_shapes']
    fit, r['fit'] = calibration(tok)
    batches = [b for group in fit.values() for b in group]
    input_device = model.get_input_embeddings().weight.device
    teacher_logs = []
    r['bf16_fit_losses'] = []
    with torch.no_grad():
        for i, batch in enumerate(batches):
            ids = batch.to(input_device)
            logits = model(input_ids=ids, use_cache=False).logits
            logprob = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
            assert torch.isfinite(logprob).all()
            teacher_logs.append(logprob.cpu())
            r['bf16_fit_losses'].append(float(F.nll_loss(logprob, ids[:, 1:].reshape(-1).to(logprob.device))))
            if (i + 1) % 16 == 0: print(f'TEACHER {i+1}/192', flush=True)
        del logits, logprob
    save(out, r)
    base, alt, mse = {}, {}, {}
    with torch.no_grad():
        for i, (n, m) in enumerate(modules.items()):
            w = m.weight.detach(); o, k = w.shape
            r['matrices'][n] = dict(shape=list(w.shape), source_sha256=sha(w))
            b = quant_nvfp4_4over6(w, 4, 16)
            a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            assert torch.isfinite(b).all() and torch.isfinite(a).all()
            diff = (a.float() - w.float()).square() - (b.float() - w.float()).square()
            mse[n] = (diff.reshape(o//8, 8, k//64, 64).sum((1, 3)) < 0).cpu()
            base[n], alt[n] = b.cpu(), a.cpu()
            m.weight.copy_(b)
            if (i + 1) % 64 == 0: print(f'CANDIDATES {i+1}/{len(modules)}', flush=True)
        del a, b, w, diff
    scoring = True
    def act(module, inputs):
        x = inputs[0]
        q = quant_nvfp4_4over6(x.detach(), 4, 16) if scoring else quantize_rows(x.detach())
        return (q + (x - x.detach()) if scoring else q, *inputs[1:])
    ah = [m.register_forward_pre_hook(act) for m in modules.values()]
    tables = {n: (torch.empty(192, base[n].numel()//512), torch.empty(192, base[n].numel()//512)) for n in names}
    hits = {n: [0, 0] for n in names}; phase = 0; sequence = 0
    def make_hook(n):
        def forward(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            def backward(dy):
                grad = dy.detach().reshape(-1, dy.shape[-1]).float().T @ x.float()
                direction = alt[n].to(grad.device).float() - base[n].to(grad.device).float()
                o, k = grad.shape
                value = (grad * direction).reshape(o//8, 8, k//64, 64).sum((1, 3)).flatten()
                assert torch.isfinite(value).all(), n
                tables[n][phase][sequence].copy_(value.cpu())
                hits[n][phase] += 1
            output.register_hook(backward)
        return forward
    handles = [m.register_forward_hook(make_hook(n)) for n, m in modules.items()]
    start = time.perf_counter(); r['initial_fit_losses'] = []
    for sequence, batch in enumerate(batches):
        ids = batch.to(input_device)
        embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
        logits = model(inputs_embeds=embeds, use_cache=False).logits
        logprob = logits[:, :-1].float().reshape(-1, logits.shape[-1]).log_softmax(-1)
        teacher_log = teacher_logs[sequence].to(logprob.device)
        ce = F.nll_loss(logprob, ids[:, 1:].reshape(-1).to(logprob.device))
        kl = F.kl_div(logprob, teacher_log, reduction='batchmean', log_target=True)
        phase = 0; ce.backward(retain_graph=True)
        phase = 1; kl.backward()
        assert math.isfinite(float(ce)) and math.isfinite(float(kl))
        r['initial_fit_losses'].append(dict(ce=float(ce), kl=float(kl)))
        del embeds, logits, logprob, teacher_log, ce, kl
        teacher_logs[sequence] = None
        print(f'SCORED {sequence+1}/192 {time.perf_counter()-start:.1f}s', flush=True)
        if (sequence + 1) % 16 == 0: save(out, r)
    assert all(v == [192, 192] for v in hits.values())
    for h in handles: h.remove()
    del teacher_logs
    r['score_seconds'] = time.perf_counter() - start
    flat = modulewise_maps(tables, names)
    maps = {'weight_mse': mse, **{p: {n: v.reshape(base[n].shape[0]//8, base[n].shape[1]//64)
                                    for n, v in ms.items()} for p, ms in flat.items()}}
    score_dir = out / 'scores'; score_dir.mkdir()
    for i, n in enumerate(names):
        torch.save(dict(name=n, ce=tables[n][0], kl=tables[n][1]), score_dir / f'{i:03d}.pt')
    del tables, flat
    torch.save(dict(source=r['source'], revision=r['revision'], maps=maps, baseline='FourOverSix',
                    alternative='E0M3 alpha1', type_block=(8, 64), scope='qwen3_5_text_linear'), out / 'maps.pt')
    r['maps_frozen'] = True; r['map_file_sha256'] = digest_file(out / 'maps.pt')
    r['selected_tiles'] = {p: sum(int(m.sum()) for m in ms.values()) for p, ms in maps.items()}
    assert r['selected_tiles']['pooled192'] <= 256
    sparse = {n: m.flatten().nonzero().flatten().tolist() for n, m in maps['pooled192'].items() if m.any()}
    (out / 'pooled192_indices.json').write_text(json.dumps(dict(source=r['source'], revision=r['revision'],
        scope='qwen3_5_text_linear', baseline='FourOverSix', alternative='E0M3 alpha1', type_block=[8, 64], modules=sparse), indent=2)+'\n')
    scoring = False; save(out, r); print('FROZEN '+json.dumps(r['selected_tiles']), flush=True)
    excluded = {d['document_sha256'] for meta in r['fit'].values() for d in meta['documents']}
    assert len(excluded) == 192
    eval_batches, r['c4_data'] = heldout_data(tok, excluded)
    policies = ('four_over_six', 'pooled192', 'c4_64', 'mixed64', 'weight_mse')
    def install(p):
        for n, m in modules.items():
            b = base[n].to(m.weight.device)
            m.weight.copy_(b if p == 'four_over_six' else apply_mask(b, alt[n].to(m.weight.device), maps[p][n].to(m.weight.device)))
    with torch.no_grad():
        ids = eval_batches[0].to(input_device); changed = ids.clone()
        replacement = tok.eos_token_id if tok.eos_token_id is not None else 0
        changed[:, 128:] = replacement
        for p in ('four_over_six', 'pooled192'):
            install(p)
            a = model(input_ids=ids, use_cache=False).logits[:, :128].clone()
            b = model(input_ids=changed, use_cache=False).logits[:, :128].clone()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            equal = torch.equal(a, b)
            r['suffix_intervention'][p] = dict(equal=equal, max_logit_difference=float((a.float()-b.float()).abs().max()),
                                               prefix_tokens=128, replacement_token_id=replacement)
            assert equal, p
        del a, b
        save(out, r)
        for p in policies:
            install(p); values = []
            for i, batch in enumerate(eval_batches):
                ids = batch.to(input_device)
                logits = model(input_ids=ids, use_cache=False).logits
                loss = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                              ids[:, 1:].reshape(-1).to(logits.device)))
                assert math.isfinite(loss); values.append(loss)
                if (i+1) % 32 == 0: print(f'EVAL {p} {i+1}/256', flush=True)
            r['evaluation'][p] = dict(nll=values, ppl=math.exp(sum(values)/len(values)), scored_tokens=256*511)
            save(out, r); print(f'PPL {p} {r["evaluation"][p]["ppl"]:.6f}', flush=True)
    for h in ah: h.remove()
    r['contrasts'] = {p: paired(r['evaluation']['pooled192']['nll'], r['evaluation'][p]['nll']) for p in policies if p != 'pooled192'}
    assert digest_file(out / 'maps.pt') == r['map_file_sha256']
    r['frozen_map_unchanged'] = True; r['status'] = 'complete'; save(out, r)
    c = r['contrasts']['four_over_six']
    lines = ['# Qwen3.8-27B: unchanged pooled rule, held-out C4', '',
             'One shared 192-sequence score table; same 256-tile cap; no backtracking or acceptance gate.', '',
             '| Policy | C4 PPL |', '|---|---:|']
    lines += [f'| {p} | {r["evaluation"][p]["ppl"]:.6f} |' for p in policies]
    lines += ['', f'Pooled192 minus FourOverSix: ΔPPL {c["ppl_delta"]:+.6f}; '
              f'ΔNLL {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} (descriptive paired 2SE).', '',
              'C4 is a calibration source; this is held-out within-source evaluation. '
              '256 validation documents, 512 tokens each; zero calibration hash overlap. '
              'Both policies pass exact prefix independence with causal activation factors. '
              'Native Transformers 5.16.1 hybrid text pathway; simulated nonhead-linear W4A4. '
              'One calibration pool; no claim of simultaneous confidence, native speed or a universal guarantee.']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()

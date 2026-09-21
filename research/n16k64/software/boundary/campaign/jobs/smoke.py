"""Pre-freeze compatibility / memory / cost smoke test (no quantized quality value is stored or printed).

Checks: tokenizer + calibration/evaluation input reconstruction (and archived token-hash equality
for development models), offline model load, quantization scope, 16x64 divisibility and audited
tile totals, archived weight hashes, BF16/W4A4 finiteness, a 2-sequence STE scoring pass, KV-cache
causality of per-token activation quantization, lm-eval HFLM compatibility (metrics discarded),
long-context memory, and timings/peak memory for cost planning.
"""
import argparse
import json
import math
import os
import time

import torch

from campaign import calibrate as C
from campaign import data as D
from campaign import models as MOD
from campaign import policies as P
from campaign import quant as Q
from campaign import runtime


def peak(reset=False):
    out = {str(i): int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())}
    if reset:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--max-memory-gib', type=float, default=None)
    ap.add_argument('--skip-lmeval', action='store_true')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    out = runtime.out_dir('smoke')
    spec = MOD.REGISTRY[args.model]
    rep = dict(model=args.model, spec=spec, checks={}, timings={}, memory={}, quality_values_recorded=False)
    save = lambda: runtime.atomic_json(out / 'smoke_report.json', rep)
    tok = MOD.load_tokenizer(args.model)
    t = time.time()
    seqs, domains, cmeta = C.calibration_inputs(args.model, 'seed0', tok)
    rep['timings']['calibration_inputs_s'] = time.time() - t
    rep['checks']['calibration_sequences'] = len(seqs)
    rep['calibration_manifest_sha256'] = D.manifest_sha256(cmeta)
    runtime.atomic_json(out / 'calibration_manifest_seed0.json', cmeta)
    if spec['panel'] == 'development':
        prior = json.loads((D.SOURCE_ROOT / spec['archived_calibration'] / 'report.json').read_text())
        rep['checks']['builder_rule_reproduces_archived_manifest'] = None
        try:
            _, bmeta = D.builder_seed0(tok)
            same = all([d['document_sha256'] for d in bmeta[dm]['documents']] == [d['document_sha256'] for d in prior['fit'][dm]['documents']]
                       and [d['offset'] for d in bmeta[dm]['documents']] == [d['offset'] for d in prior['fit'][dm]['documents']]
                       for dm in ('math', 'code'))
            rep['checks']['builder_rule_reproduces_archived_manifest'] = same
        except Exception as exc:
            rep['checks']['builder_rule_error'] = repr(exc)
    t = time.time()
    wiki, wmeta = D.wiki_windows(tok)
    c4, cmeta4 = D.c4_windows(tok)
    rep['timings']['eval_windows_s'] = time.time() - t
    rep['checks']['windows'] = dict(wiki=len(wiki), c4=len(c4), c4_unique_documents=cmeta4['unique_documents'], wiki_articles=wmeta['articles'])
    if spec['panel'] == 'development':
        kse = {'llama8b': 'results/kse_paper/job_336566/llama8b', 'qwen4b': 'results/kse_paper/job_336566/qwen4b',
               'qwen27b': 'results/kse_paper/job_336969/qwen27b'}[args.model]
        kr = json.loads((D.SOURCE_ROOT / kse / 'report.json').read_text())
        rep['checks']['wiki_tokens_equal_archived'] = kr['data']['wiki']['token_sha256'] == wmeta['token_sha256']
        rep['checks']['c4_tokens_equal_archived'] = kr['data']['c4_paper']['token_sha256'] == cmeta4['token_sha256']
        rep['checks']['c4_documents_equal_archived'] = kr['data']['c4_paper']['documents'] == cmeta4['documents']
    save()
    runtime.phase('load_model')
    ngpu = torch.cuda.device_count()
    mm = {i: f'{args.max_memory_gib}GiB' for i in range(ngpu)} if args.max_memory_gib else None
    t = time.time()
    model, info = MOD.load_model(args.model, attn='sdpa', device_map=('cuda' if ngpu == 1 else 'balanced'), max_memory=mm)
    rep['timings']['load_s'] = time.time() - t
    rep['memory']['after_load'] = peak(reset=True)
    rep['model_class'] = type(model).__name__
    rep['attn_implementation'] = model.config._attn_implementation
    rep['device_map'] = getattr(model, 'hf_device_map', None)
    modules = MOD.scope(model, args.model)
    mm_sha, entries = MOD.module_manifest(modules)
    runtime.atomic_json(out / 'module_manifest.json', entries)
    t8, bad8 = MOD.tile_totals(modules, (8, 64))
    t16, bad16 = MOD.tile_totals(modules, (16, 64))
    rep['checks'].update(modules=len(modules), n8_total=t8, n16_total=t16, not_n16_divisible=bad16,
                         activation_widths_divisible_by_16=all(m.in_features % 16 == 0 for m in modules.values()),
                         module_manifest_sha256=mm_sha, audited_totals_match=(None if spec['n8_total'] is None else
                                                                             (t8 == spec['n8_total'] and t16 == spec['n16_total'])))
    rep['module_types'] = sorted({n.split('.')[-1] for n in modules})
    rep['shapes'] = sorted({tuple(m.weight.shape) for m in modules.values()})
    if spec['panel'] == 'development':
        rep['checks']['weights_equal_archived_hashes'] = [e['weight_sha256'] for e in entries] == [prior['matrices'][n]['source_sha256'] for n in modules]
        rep['checks']['scope_order_equal_archived'] = list(prior['matrices']) == list(modules)
    save()
    dev0 = model.get_input_embeddings().weight.device
    with torch.no_grad():
        t = time.time()
        lg = model(input_ids=seqs[0].to(dev0), use_cache=False).logits
        rep['checks']['bf16_calibration_forward_finite'] = bool(torch.isfinite(lg).all())
        rep['timings']['bf16_forward_512_s'] = time.time() - t
        teacher = [model(input_ids=s.to(dev0), use_cache=False).logits[:, :-1].float().reshape(-1, lg.shape[-1]).log_softmax(-1).cpu() for s in seqs[:2]]
        del lg
    runtime.phase('scoring_smoke')
    t = time.time()
    alt, stats = C.build_candidates(modules, alt_on_gpu=False)
    rep['timings']['candidates_s'] = time.time() - t
    rep['memory']['after_candidates'] = peak(reset=True)
    t = time.time()
    sc = C.run_scoring(model, modules, seqs[:2], ['math', 'code'], lambda i: teacher[i], alt, raw='none', sample_fraction=0.0)
    rep['timings']['scoring_2seq_s'] = time.time() - t
    rep['timings']['scoring_per_sequence_s'] = rep['timings']['scoring_2seq_s'] / 2
    rep['memory']['scoring_peak'] = peak(reset=True)
    rep['checks']['scoring_finite'] = all(bool(torch.isfinite(sc['full8'][n].ce_sum).all() and torch.isfinite(sc['full8'][n].kl_sum).all()) for n in sc['names'])
    del sc, alt, stats, teacher
    torch.cuda.empty_cache()
    save()
    runtime.phase('eval_smoke')
    # restore pristine weights from the archived loader, then use the policy installer
    import gc
    del model, modules
    gc.collect()
    torch.cuda.empty_cache()
    rep['memory']['after_release'] = {str(i): torch.cuda.memory_allocated(i) for i in range(ngpu)}
    model, _ = MOD.load_model(args.model, attn='sdpa', device_map=('cuda' if ngpu == 1 else 'balanced'), max_memory=mm)
    modules = MOD.scope(model, args.model)
    dev0 = model.get_input_embeddings().weight.device
    inst = P.Installer(args.model, model, modules, cache='none', campaign_root=os.environ.get('CAMPAIGN_ROOT'))
    t = time.time()
    inst.install(dict(name='four_over_six', kind='four_over_six'))
    rep['timings']['install_four_over_six_s'] = time.time() - t
    with torch.no_grad():
        t = time.time()
        lg = model(input_ids=wiki[0].to(dev0), use_cache=False).logits
        torch.cuda.synchronize()
        rep['timings']['w4a4_forward_2048_s'] = time.time() - t
        rep['checks']['w4a4_forward_finite'] = bool(torch.isfinite(lg).all())
        rep['memory']['w4a4_forward_2048'] = peak(reset=True)
        # KV-cache causality: incremental decode logits vs one full forward on the same 64 tokens
        ids = wiki[0][:, :64].to(dev0)
        full = model(input_ids=ids, use_cache=False).logits[0].float()
        outk = model(input_ids=ids[:, :48], use_cache=True)
        past = outk.past_key_values
        inc = [outk.logits[0].float()]
        for j in range(48, 64):
            o = model(input_ids=ids[:, j:j + 1], past_key_values=past, use_cache=True)
            past = o.past_key_values
            inc.append(o.logits[0].float())
        inc = torch.cat(inc)
        rep['checks']['kv_cache_causal_max_abs_logit_diff'] = float((inc - full).abs().max())
        rep['checks']['kv_cache_causal_argmax_equal_fraction'] = float((inc.argmax(-1) == full.argmax(-1)).float().mean())
        del lg, full, inc, past
    save()
    if not args.skip_lmeval:
        runtime.phase('lmeval_smoke')
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            t = time.time()
            lm = HFLM(pretrained=model, tokenizer=tok, batch_size=16)
            res = lm_eval.simple_evaluate(model=lm, tasks=['arc_easy', 'gsm8k'], limit=4, log_samples=True, num_fewshot=None)
            rep['timings']['lmeval_smoke_s'] = time.time() - t
            rep['checks']['lmeval_ran'] = sorted(res['results'])
            rep['checks']['lmeval_samples_logged'] = {k: len(v) for k, v in res.get('samples', {}).items()}
            del res  # metrics discarded before freeze
        except Exception as exc:
            import traceback
            rep['checks']['lmeval_error'] = repr(exc)
            rep['checks']['lmeval_traceback'] = traceback.format_exc()[-4000:]
        save()
    inst.remove()
    runtime.phase('long_context_smoke')
    maxpos = getattr(model.config, 'max_position_embeddings', None) or getattr(getattr(model.config, 'text_config', None), 'max_position_embeddings', None)
    rep['max_position_embeddings'] = maxpos
    for L in (4096, 8192):
        if maxpos and maxpos >= L:
            try:
                ids = torch.cat([wiki[i] for i in range(L // 2048)], 1).to(dev0)
                inst.install(dict(name='four_over_six', kind='four_over_six'))
                with torch.no_grad():
                    t = time.time()
                    lg = model(input_ids=ids, use_cache=False).logits
                    torch.cuda.synchronize()
                rep['timings'][f'w4a4_forward_{L}_s'] = time.time() - t
                rep['memory'][f'w4a4_forward_{L}'] = peak(reset=True)
                rep['checks'][f'long_{L}_finite'] = bool(torch.isfinite(lg).all())
                del lg
                inst.remove()
            except Exception as exc:
                rep['checks'][f'long_{L}_error'] = repr(exc)
        else:
            rep['checks'][f'long_{L}_supported'] = False
    rep['memory']['free_after'] = {str(i): torch.cuda.mem_get_info(i) for i in range(ngpu)}
    rep['status'] = 'complete'
    save()
    print(json.dumps(dict(checks={k: v for k, v in rep['checks'].items() if 'traceback' not in k}, timings=rep['timings']), indent=1, default=str))


if __name__ == '__main__':
    main()

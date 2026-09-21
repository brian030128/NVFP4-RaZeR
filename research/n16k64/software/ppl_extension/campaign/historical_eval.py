"""Historical-protocol PPL anchors (V22) and cross-GPU archived anchor (V23).

Exact replica of the run_kse_paper evaluation loop: tensor-wide FourOverSix activation factor on every scoped
Linear input (quant_nvfp4_4over6(inputs[0], 4, 16)), SDPA attention, WikiText with use_cache=True and C4 with
use_cache=False, per-window F.cross_entropy on float32 logits, FP32 PPL aggregation. Policies:
  four_over_six                    no map
  archived_fixed256               archived maps.json fixed256_math_code128 (independent of regenerated scores)
  hist_n8_<k2..k6|n256>            regenerated historical maps (reloaded + verified from disk)
Every window is compared with the archived per-window NLL for the same policy.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

from campaign import data as D
from campaign import mapio as MIO
from campaign import models as MOD
from campaign import runtime
from campaign import tiles as T
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6

KSE = {'llama8b': 'results/kse_paper/job_336566/llama8b', 'qwen4b': 'results/kse_paper/job_336566/qwen4b',
       'qwen27b': 'results/kse_paper/job_336969/qwen27b'}
ARCHIVED_POLICY = {'four_over_six': 'four_over_six', 'archived_fixed256': 'n256', 'hist_n8_n256': 'n256',
                   'hist_n8_k2': 'k2', 'hist_n8_k3': 'k3', 'hist_n8_k4': 'k4', 'hist_n8_k5': 'k5', 'hist_n8_k6': 'k6'}


def sha(t):
    return D.sha(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(KSE))
    ap.add_argument('--policies', default='four_over_six,archived_fixed256,hist_n8_k3')
    ap.add_argument('--maps', default=None, help='JSON {policy: {"path":..., "sha256":...}} for hist_n8_* policies')
    ap.add_argument('--maps-from-job', default=None, help='historical calibration job id; uses its recorded map paths/digests')
    ap.add_argument('--freeze-sha256', required=True)
    ap.add_argument('--max-memory-gib', type=float, default=44)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    src = D.SOURCE_ROOT
    spec = MOD.REGISTRY[args.model]
    prior = json.loads((src / spec['archived_calibration'] / 'report.json').read_text())
    bundle = json.loads((src / spec['archived_calibration'] / 'maps.json').read_text())
    kse = json.loads((src / KSE[args.model] / 'report.json').read_text())
    out = runtime.out_dir('historical_eval')
    policies = args.policies.split(',')
    maps_spec = json.loads(Path(args.maps).read_text()) if args.maps else {}
    if args.maps_from_job:
        from campaign.policies import latest_complete_run
        hrun = latest_complete_run(os.environ['CAMPAIGN_ROOT'], args.maps_from_job)
        hrep = json.loads((hrun / 'historical' / 'historical_report.json').read_text())
        maps_spec.update({e['policy']: dict(path=e['path'], sha256=e['sha256'], from_run=hrun.name) for e in hrep['maps']})
    r = dict(status='running', model=args.model, protocol_id='historical', policies=policies, torch_version=torch.__version__,
             transformers_version=transformers.__version__, archived_transformers_version=kse['transformers_version'],
             activation='FourOverSix tensor-wide factor', wiki_use_cache=True, c4_use_cache=False, length=2048,
             gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], evaluation={}, comparison={})
    save = lambda: runtime.atomic_json(out / 'historical_eval_report.json', r)
    save()
    if transformers.__version__ != kse['transformers_version']:
        raise SystemExit(f'transformers {transformers.__version__} != archived {kse["transformers_version"]}')
    tok = MOD.load_tokenizer(args.model)
    wiki, wmeta = D.wiki_windows(tok)
    c4, cmeta = D.c4_windows(tok)
    if wmeta['token_sha256'] != kse['data']['wiki']['token_sha256'] or cmeta['token_sha256'] != kse['data']['c4_paper']['token_sha256']:
        raise SystemExit('evaluation tokens differ from archived')
    excluded = {d['document_sha256'] for m in prior['fit'].values() for d in m['documents']}
    assert excluded.isdisjoint(d['document_sha256'] for d in cmeta['documents'])
    r['inputs_equal_archived'] = True
    runtime.phase('load_model')
    path = str(MOD.snapshot_path(args.model))
    target = args.model == 'qwen27b'
    ngpu = torch.cuda.device_count()
    if target:
        from transformers import Qwen3_5ForConditionalGeneration
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map=('cuda' if ngpu == 1 else 'balanced'), max_memory=(None if ngpu == 1 else {i: f'{args.max_memory_gib}GiB' for i in range(ngpu)}),
            output_loading_info=True)
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'language_model' in n and 'head' not in n}
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    model.eval().requires_grad_(False)
    r['model_class'] = type(model).__name__
    r['attention_backend'] = model.config._attn_implementation
    names = list(modules)
    if names != list(prior['matrices']):
        raise SystemExit('scope differs from archived')
    pristine, base = {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            if sha(m.weight) != prior['matrices'][n]['source_sha256']:
                raise SystemExit(f'weight hash mismatch {n}')
            pristine[n] = m.weight.detach().to('cpu', copy=True)
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16).to('cpu', copy=True)
    shapes = {n: tuple(m.weight.shape) for n, m in modules.items()}
    known = None
    masks = {}
    for pol in policies:
        if pol == 'four_over_six':
            masks[pol] = None
        elif pol == 'archived_fixed256':
            mk = {}
            for n in names:
                f = torch.zeros((shapes[n][0] // 8) * (shapes[n][1] // 64), dtype=torch.bool)
                f[bundle['maps']['fixed256_math_code128'][n]] = True
                mk[n] = f.reshape(shapes[n][0] // 8, shapes[n][1] // 64)
            masks[pol] = mk
            r.setdefault('map_digests', {})[pol] = dict(source='archived maps.json', maps_json_sha256=prior['map_sha256'],
                                                         selected=int(sum(int(v.sum()) for v in mk.values())))
        else:
            spec_m = maps_spec[pol]
            header, mk, digest = MIO.read_map(spec_m['path'], expected_sha256=spec_m['sha256'])
            MIO.verify_for_model(header, model_id=spec['model_id'], model_revision=spec['revision'], tokenizer_revision=spec['revision'],
                                 weight_shapes=shapes, type_block=(8, 64), protocol_id='historical', policy_name=pol)
            masks[pol] = mk
            r.setdefault('map_digests', {})[pol] = dict(path=spec_m['path'], sha256=digest, selected=header['totals']['selected_tiles'],
                                                         map_reloaded_for_evaluation=True)
    save()

    def act(module, inputs):
        return (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])
    handles = [m.register_forward_pre_hook(act) for m in modules.values()]
    dev0 = model.get_input_embeddings().weight.device

    @torch.no_grad()
    def install(pol):
        for n, m in modules.items():
            b = base[n].to(m.weight.device)
            if masks[pol] is not None and bool(masks[pol][n].any()):
                a = quant_mix_4_6(pristine[n].to(m.weight.device), 4, 16, type_block=(8, 64), clip='a1', elect='always')
                b = T.apply_mask(b, a, masks[pol][n], (8, 64))
            m.weight.copy_(b)

    for pol in policies:
        runtime.phase(f'policy_{pol}')
        t0 = time.time()
        install(pol)
        ev = {}
        with torch.no_grad():
            for domain, bs in (('wiki', wiki), ('c4', c4)):
                values = []
                for ids in bs:
                    logits = model(input_ids=ids.to(dev0), use_cache=(domain == 'wiki')).logits
                    v = float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1).to(logits.device)))
                    if not math.isfinite(v):
                        raise SystemExit('non-finite NLL')
                    values.append(v)
                    del logits
                t = torch.tensor(values, dtype=torch.float32) * 2048
                ppl = float(torch.exp(t.sum() / (len(values) * 2048)))
                ev[domain] = dict(ppl=ppl, nll=values, windows=len(values))
        ev['seconds'] = time.time() - t0
        r['evaluation'][pol] = ev
        ap_ = ARCHIVED_POLICY[pol]
        cmp = {}
        for domain in ('wiki', 'c4'):
            old = kse['evaluation'][ap_][domain]
            diffs = [a - b for a, b in zip(ev[domain]['nll'], old['nll'])]
            cmp[domain] = dict(archived_ppl=old['ppl'], regenerated_ppl=ev[domain]['ppl'], ppl_rel_diff=ev[domain]['ppl'] / old['ppl'] - 1,
                               window_nll_max_abs_diff=max(abs(d) for d in diffs), window_nll_mean_diff=sum(diffs) / len(diffs),
                               exact_equal=ev[domain]['nll'] == old['nll'])
        r['comparison'][pol] = cmp
        save()
        print(f'HIST {pol} ' + json.dumps({d: (round(c['regenerated_ppl'], 6), round(c['archived_ppl'], 6)) for d, c in cmp.items()}), flush=True)
    for h in handles:
        h.remove()
    # paired effects versus four_over_six: archived vs regenerated
    if 'four_over_six' in r['evaluation']:
        pe = {}
        for pol in policies:
            if pol == 'four_over_six':
                continue
            pe[pol] = {}
            for domain in ('wiki', 'c4'):
                reg = math.log(r['evaluation'][pol][domain]['ppl']) - math.log(r['evaluation']['four_over_six'][domain]['ppl'])
                arc = math.log(kse['evaluation'][ARCHIVED_POLICY[pol]][domain]['ppl']) - math.log(kse['evaluation']['four_over_six'][domain]['ppl'])
                pe[pol][domain] = dict(regenerated_dlogppl=reg, archived_dlogppl=arc, difference=reg - arc,
                                       sign_equal=(reg < 0) == (arc < 0))
        r['paired_effects'] = pe
    r['status'] = 'complete'
    save()
    source_manifest = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id='historical', protocol_freeze_sha256=args.freeze_sha256,
        source=dict(model_id=spec['model_id'], model_revision=spec['revision'], tokenizer_revision=spec['revision'], model_class=type(model).__name__,
                    module_manifest_sha256=MOD.module_manifest(modules, with_weight_hash=False)[0], source_manifest_sha256=source_manifest),
        environment=runtime.environment(attention_backend='sdpa', activation_quantizer='quant_nvfp4_4over6 tensor-wide'),
        data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=D.manifest_sha256(dict(wiki=wmeta, c4=cmeta)),
                  token_hashes=dict(wiki=wmeta['token_sha256'], c4=cmeta['token_sha256']), overlap_audit=dict(c4_calibration_hash_overlap=0)),
        policies=[dict(name=p, weight_format='FourOverSix/E0M3 8x64', activation_format='four_over_six_tensor', scale_block=16, type_block=[8, 64],
                       map_path=r.get('map_digests', {}).get(p, {}).get('path'), map_sha256=r.get('map_digests', {}).get(p, {}).get('sha256'),
                       selected_tiles=r.get('map_digests', {}).get(p, {}).get('selected', 0), total_tiles=MOD.REGISTRY[args.model]['n8_total'],
                       map_reloaded_for_evaluation=(p.startswith('hist_n8') or None)) for p in policies],
        results=dict(raw_outputs=[str(out / 'historical_eval_report.json')], summary=r['comparison'], uncertainty={},
                     attempted_endpoints=[f'{p}:{d}' for p in policies for d in ('wiki', 'c4')], missing_endpoints=[]),
        logs=[], failures=[]))


if __name__ == '__main__':
    main()

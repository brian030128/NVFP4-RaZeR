"""Why does Qwen3-4B MixFP4 score BELOW the unquantized BF16 reference?

Perplexity is a proper scoring rule on the FULL predictive distribution, so it
rewards two different things at once: predicting the right token, and assigning
a well-calibrated probability to it. A perturbation that only flattens an
overconfident distribution lowers cross-entropy while predicting nothing better.

This job separates the two on the released 2048-token WikiText protocol.

For every policy (pristine BF16, FourOverSix W4A4, MixFP4 at 256 and 65,536
tiles) it records per-token target NLL, predictive entropy, top-1 probability,
top-1 next-token accuracy, agreement with the BF16 argmax, and the NLL under a
grid of global logit temperatures.

The decisive comparison is temperature-matched perplexity. Rescaling logits by a
scalar leaves every argmax -- hence every accuracy -- untouched and changes only
sharpness. If BF16 at its own best temperature falls to roughly the MixFP4
perplexity, the gap is sharpness, not modelling. If it does not, the map is
genuinely modelling WikiText better than the unquantized weights.

The temperature is fitted ON the evaluation set on purpose: it is an oracle
upper bound on how much of the gap pure recalibration can explain, not a
deployable method.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.interacting_format import apply_mask
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_adaptive_paper import elect
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import sha

CALIB = '333779'
MODEL = 'qwen4b'
FROZEN = 'fixed256_math_code128'
LENGTH = 2048
COUNTS = (256, 65536)
# 1.0 is the identity; the grid leans above 1 because an overconfident model
# needs T > 1. A few points below 1 are kept so the fit cannot silently sit on
# the edge of the grid without that being visible.
TEMPS = (0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
         1.40, 1.50, 1.60, 1.80, 2.00)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    old = Path(args.stage_root) / f'results/math_code_adaptive/calibration_{CALIB}_{MODEL}'
    prior = json.loads((old / 'report.json').read_text())
    bundle = json.loads((old / 'maps.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    assert not prior['uses_c4_calibration'] and not prior['uses_wiki_calibration']
    assert digest_file(old / 'maps.json') == prior['map_sha256']
    assert transformers.__version__ == prior['transformers_version']

    r = dict(status='running', model=MODEL, source=prior['source'], revision=prior['revision'],
             job_id=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__,
             transformers_version=transformers.__version__, calibration=str(old),
             map_sha256=prior['map_sha256'], length=LENGTH, temperatures=list(TEMPS),
             activation='FourOverSix tensor-wide factor', policies={})

    def save():
        (out / 'report.json').write_text(json.dumps(r, indent=2) + '\n')

    save()
    upper, order, slices, names = elect(old, prior)
    r['eligible_tiles'] = int((upper < 0).sum())
    r['total_tiles'] = int(upper.numel())

    maps = {}
    for count in COUNTS:
        flat = torch.zeros(upper.numel(), dtype=torch.bool)
        take = order[:count]
        flat[take[upper[take] < 0]] = True
        maps[f'n{count}'] = {n: flat[lo:hi].clone() for n, (lo, hi) in slices.items()}
    # Where the elected tiles land, by module -- a map concentrated in one place
    # would mean something different from one spread across the network.
    r['selected_by_module'] = {
        f'n{c}': {n: int(m.sum()) for n, m in maps[f'n{c}'].items() if int(m.sum())}
        for c in COUNTS}
    del upper, order

    model = AutoModelForCausalLM.from_pretrained(
        prior['source'], torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
    model.eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(prior['source'])
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert list(modules) == names
    r['attention_backend'] = model.config._attn_implementation
    r['lm_head_quantized'] = False

    for policy in maps:
        for n, m in modules.items():
            maps[policy][n] = maps[policy][n].reshape(m.weight.shape[0] // 8,
                                                      m.weight.shape[1] // 64)
    for n, m in modules.items():
        want = torch.zeros(maps['n256'][n].numel(), dtype=torch.bool)
        want[bundle['maps'][FROZEN][n]] = True
        assert torch.equal(maps['n256'][n].reshape(-1), want), n
    r['frozen_map_reproduced'] = True
    print('REELECTION AT 256 MATCHES FROZEN MAP', flush=True)

    pristine, base, alt = {}, {}, {}
    with torch.no_grad():
        for n, m in modules.items():
            assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
            pristine[n] = m.weight.detach().clone()
            base[n] = quant_nvfp4_4over6(m.weight, 4, 16)
            alt[n] = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            assert torch.isfinite(base[n]).all() and torch.isfinite(alt[n]).all()
    r['source_weights_verified'] = True

    batches, r['data'] = data(tok, prior, LENGTH)
    wiki = batches['wiki']
    print(f'DATA wiki={len(wiki)}', flush=True)
    save()

    def act(module, inputs):
        return (quant_nvfp4_4over6(inputs[0], 4, 16), *inputs[1:])

    handles = []

    def install(policy):
        for h in handles:
            h.remove()
        handles.clear()
        with torch.no_grad():
            for n, m in modules.items():
                if policy == 'bf16':
                    m.weight.copy_(pristine[n])
                elif policy == 'four_over_six':
                    m.weight.copy_(base[n])
                else:
                    m.weight.copy_(apply_mask(base[n], alt[n], maps[policy][n].cuda()))
        if policy != 'bf16':
            # BF16 is the unquantized reference: no weight AND no activation quantization.
            handles.extend(m.register_forward_pre_hook(act) for m in modules.values())

    def ppl(values):
        # Same FP32 aggregation as the released run_ppl.py.
        t = torch.tensor(values, dtype=torch.float32) * LENGTH
        return float(torch.exp(t.sum() / (len(values) * LENGTH)))

    bf16_top1 = []

    with torch.no_grad():
        for policy in ('bf16', 'four_over_six', 'n256', 'n65536'):
            install(policy)
            nll, ent, top1p, correct, agree, logit_sd, tok_nll = [], [], [], [], [], [], []
            temp_nll = {t: [] for t in TEMPS}
            for w, ids in enumerate(wiki):
                ids = ids.cuda()
                logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
                target = ids[:, 1:].reshape(-1)
                flat = logits.reshape(-1, logits.shape[-1])
                lp = F.log_softmax(flat, dim=-1)
                per = -lp.gather(1, target[:, None]).squeeze(1)
                nll.append(float(per.mean()))
                tok_nll.append(per.to(torch.float16).cpu())
                p = lp.exp()
                ent.append(float((-(p * lp).sum(-1)).mean()))
                mx = p.max(-1)
                top1p.append(float(mx.values.mean()))
                arg = mx.indices
                correct.append(float((arg == target).float().mean()))
                if policy == 'bf16':
                    bf16_top1.append(arg.cpu())
                    agree.append(1.0)
                else:
                    agree.append(float((arg.cpu() == bf16_top1[w]).float().mean()))
                logit_sd.append(float(flat.std()))
                for t in TEMPS:
                    temp_nll[t].append(float(F.cross_entropy(flat / t, target)))
                del logits, flat, lp, p, mx
            entry = dict(
                ppl=ppl(nll), nll=nll, mean_nll=sum(nll) / len(nll),
                mean_entropy=sum(ent) / len(ent),
                mean_top1_prob=sum(top1p) / len(top1p),
                top1_accuracy=sum(correct) / len(correct),
                bf16_argmax_agreement=sum(agree) / len(agree),
                mean_logit_sd=sum(logit_sd) / len(logit_sd),
                temperature_ppl={str(t): ppl(v) for t, v in temp_nll.items()})
            best = min(entry['temperature_ppl'], key=lambda k: entry['temperature_ppl'][k])
            entry['best_temperature'] = float(best)
            entry['best_temperature_ppl'] = entry['temperature_ppl'][best]
            r['policies'][policy] = entry
            torch.save(torch.cat(tok_nll), out / f'token_nll_{policy}.pt')
            print(f'{policy} ppl={entry["ppl"]:.6f} H={entry["mean_entropy"]:.4f} '
                  f'top1p={entry["mean_top1_prob"]:.4f} acc={entry["top1_accuracy"]:.4f} '
                  f'agree={entry["bf16_argmax_agreement"]:.4f} '
                  f'T*={entry["best_temperature"]} ppl@T*={entry["best_temperature_ppl"]:.6f}',
                  flush=True)
            save()
            assert math.isfinite(entry['ppl'])

    r['status'] = 'complete'
    save()
    print('=== complete ===', flush=True)


if __name__ == '__main__':
    main()

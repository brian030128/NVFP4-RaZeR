"""Investigation for the historical Tier-B BF16 fit-NLL tolerance miss (A6000 vs archived H100).

Same GPU, same verified weights and tokens: per-sequence BF16 NLL of the 128 archived calibration sequences under
kernel-level variants (eager bs1 = archived configuration, sdpa bs1, eager bs2, sdpa bs2, float32 eager bs1), plus the
archived values. If same-hardware kernel variants already differ at the magnitude seen against the archive, the miss is
numerical noise rather than an input/weight/protocol mismatch. The tolerance itself is not changed.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from campaign import data as D
from campaign import models as MOD
from campaign import runtime


def nlls(model, seqs, bs):
    dev = model.get_input_embeddings().weight.device
    out = []
    with torch.no_grad():
        for s in range(0, len(seqs), bs):
            ids = torch.cat(seqs[s:s + bs]).to(dev)
            lg = model(input_ids=ids, use_cache=False).logits
            for j in range(ids.shape[0]):
                lp = lg[j:j + 1, :-1].float().reshape(-1, lg.shape[-1]).log_softmax(-1)
                out.append(float(F.nll_loss(lp, ids[j, 1:].reshape(-1))))
            del lg
    return out


def cmp(a, b):
    d = [x - y for x, y in zip(a, b)]
    return dict(max_abs=max(abs(v) for v in d), mean_abs=sum(abs(v) for v in d) / len(d), mean_signed=sum(d) / len(d), exact_equal=a == b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    out = runtime.out_dir('diag')
    spec = MOD.REGISTRY[args.model]
    arch = json.loads((D.SOURCE_ROOT / spec['archived_calibration'] / 'report.json').read_text())
    tok = MOD.load_tokenizer(args.model)
    fit, _ = D.archived_manifest(args.model)
    b, meta = D.crops_from_manifest(tok, fit)
    seqs = b['math'] + b['code']
    res = dict(model=args.model, variants={})
    runtime.phase('load_model')
    for attn, dtype, bs in (('eager', torch.bfloat16, 1), ('sdpa', torch.bfloat16, 1), ('eager', torch.bfloat16, 2), ('sdpa', torch.bfloat16, 2), ('eager', torch.float32, 1)):
        model, _ = MOD.load_model(args.model, attn=attn, device_map='cuda', dtype=dtype)
        res['variants'][f'{attn}_{str(dtype).split(".")[-1]}_bs{bs}'] = nlls(model, seqs, bs)
        del model
        torch.cuda.empty_cache()
    v = res['variants']
    ref = 'eager_bfloat16_bs1'
    res['vs_archived'] = {k: cmp(x, arch['bf16_fit_nll']) for k, x in v.items()}
    res['vs_archived_configuration_on_this_gpu'] = {k: cmp(x, v[ref]) for k, x in v.items() if k != ref}
    runtime.atomic_json(out / 'numeric_noise_diag.json', res)
    print(json.dumps(dict(vs_archived=res['vs_archived'], same_gpu=res['vs_archived_configuration_on_this_gpu']), indent=1))


if __name__ == '__main__':
    main()

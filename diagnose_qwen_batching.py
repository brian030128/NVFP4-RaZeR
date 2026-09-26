"""Why does Qwen3.8-27B give a different development loss when documents are batched?

For the first 16 development documents, compares each document's logits computed
alone against the same document inside a batch of 8, in plain BF16 and with the
per-document FourOverSix activation hook. Reports max |logit| difference and the
mean CE/KL change per condition.
"""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from quantize.fast_act import quant_per_document
from run_multiround import CALIBRATIONS, load_development


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    from transformers import Qwen3_5ForConditionalGeneration
    prior = json.loads((CALIBRATIONS['qwen27b'] / 'report.json').read_text())
    model = Qwen3_5ForConditionalGeneration.from_pretrained(prior['source'], revision=prior['revision'],
                                                            dtype=torch.bfloat16, attn_implementation='sdpa',
                                                            device_map='cuda')
    model.eval()
    modules = [m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
               and 'language_model' in n and 'head' not in n]
    dev, _ = load_development('qwen27b')
    ids = torch.cat([r['ids'] for r in dev[:16]]).cuda()
    out = {}
    for label, hooked in (('bf16', False), ('fake_quant', True)):
        handles = [m.register_forward_pre_hook(lambda mod, inp: (quant_per_document(inp[0]), *inp[1:]))
                   for m in modules] if hooked else []
        alone = torch.cat([model(input_ids=ids[i:i + 1], use_cache=False).logits.float() for i in range(16)])
        batched = torch.cat([model(input_ids=ids[i:i + 8], use_cache=False).logits.float() for i in (0, 8)])
        for h in handles:
            h.remove()
        diff = (alone - batched).abs()
        ce = lambda lg: F.cross_entropy(lg[:, :-1].transpose(1, 2), ids[:, 1:], reduction='none').mean(-1)
        out[label] = dict(max_abs_logit_diff=float(diff.max()), mean_abs_logit_diff=float(diff.mean()),
                          per_doc_max=[float(diff[i].max()) for i in range(16)],
                          ce_alone=float(ce(alone).mean()), ce_batched=float(ce(batched).mean()),
                          first_bad_position=[int((diff[i].amax(-1) > 1e-2).nonzero()[0]) if bool((diff[i].amax(-1) > 1e-2).any()) else -1
                                              for i in range(16)])
        print(label, json.dumps({k: v for k, v in out[label].items() if k != 'per_doc_max'}), flush=True)
    Path('/work/u4320956/mixfp4_potential/qwen_batching_diagnosis.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

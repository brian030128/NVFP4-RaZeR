"""Option A unit tests (results/speedups/PROTOCOL_CHUNKED.md): chunked_loss vs run_multiround's whole-batch loss, bitwise.

For the four final models' vocabularies (Mistral 32,768, Phi-4 100,352, Llama 128,256, Qwen3.8-27B 248,320), 512-token
documents and batches 1, 2, 3, 4, 8, 16:
  - per-document CE and KL of the development evaluation: chunked_loss.per_sequence_losses vs
    per_sequence_losses(logits[:, :-1].float().log_softmax(-1), ids, cat(teacher).float());
  - the scoring gradient into the logits: chunked_loss.kl_logit_gradient vs the autograd gradient of kl.sum()
    (batches up to 8, the scoring batch).
Bitwise: FP32 per-document values as int32 bit patterns, the bf16 logit gradient as int16.

python repro_local/realquant/test_chunked_loss.py OUT_JSON
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import chunked_loss  # noqa: E402

VOCABS = {'mistral7b': 32768, 'phi4': 100352, 'llama8b': 128256, 'qwen27b': 248320}
T = 512


def legacy_losses(lp, ids, t):
    # run_multiround.per_sequence_losses, verbatim
    ce = F.nll_loss(lp.transpose(1, 2), ids[:, 1:].to(lp.device), reduction='none').mean(-1)
    kl = (t.exp() * (t - lp)).sum(-1).mean(-1)
    return ce, kl


def bits(a):
    return a.view({torch.float32: torch.int32, torch.bfloat16: torch.int16}[a.dtype])


def main():
    out = Path(sys.argv[1])
    g = torch.Generator(device='cuda').manual_seed(0)
    result = dict(documents_tokens=T, cases={}, failures=[])
    for model, v in VOCABS.items():
        for batch in (1, 2, 3, 4, 8, 16):
            logits = (torch.randn(batch, T, v, generator=g, device='cuda') * 3).bfloat16()
            ids = torch.randint(0, v, (batch, T), generator=g, device='cuda')
            teacher = [((torch.randn(1, T - 1, v, generator=g, device='cuda') * 3).float().log_softmax(-1)).bfloat16().cpu()
                       for _ in range(batch)]
            case = dict(chunks=chunked_loss.document_chunks(batch))
            with torch.no_grad():
                lp = logits[:, :-1].float().log_softmax(-1)
                t = torch.cat(teacher).to('cuda').float()
                ce0, kl0 = legacy_losses(lp, ids, t)
                del lp, t
                ce1, kl1 = chunked_loss.per_sequence_losses(logits, ids, teacher, 'cuda')
            case['dev_ce_bitwise'] = torch.equal(bits(ce0), bits(ce1))
            case['dev_kl_bitwise'] = torch.equal(bits(kl0), bits(kl1))
            if batch <= 8:
                with torch.enable_grad():
                    leaf = logits.clone().requires_grad_()
                    lp = leaf[:, :-1].float().log_softmax(-1)
                    t = torch.cat(teacher).to('cuda').float()
                    kl = (t.exp() * (t - lp)).sum(-1).mean(-1)
                    kl.sum().backward()
                    ref = leaf.grad
                    del lp, t, kl
                    got = chunked_loss.kl_logit_gradient(logits.clone().requires_grad_(), teacher, 'cuda')
                case['score_grad_bitwise'] = torch.equal(bits(ref), bits(got))
                del ref, got, leaf
            ok = all(v_ for k_, v_ in case.items() if k_.endswith('bitwise'))
            case['passed'] = ok
            if not ok:
                result['failures'].append(f'{model} batch {batch}')
            result['cases'][f'{model} V={v} batch={batch}'] = case
            print(f'{model:10s} V={v:6d} batch={batch:2d} chunks={case["chunks"]} ' +
                  ' '.join(f'{k}={v_}' for k, v_ in case.items() if k.endswith('bitwise')), flush=True)
            del logits, teacher
            torch.cuda.empty_cache()
    result['passed'] = not result['failures']
    out.write_text(json.dumps(result, indent=1) + '\n')
    print('PASS' if result['passed'] else f'FAIL {result["failures"]}')
    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()

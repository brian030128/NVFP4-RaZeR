"""Diagnostic: does batching change the (quantized) forward pass?

For the first 8 C1-schedule fit sequences: log-probs at batch 1 vs batch 8 for
(1) the BF16 model, (2) W4A4 with FourOverSix weights and run_multiround.py's per-document
activation hook; and per Linear, the largest difference of the (pre-quantization) input
between batch 1 and batch 8, in forward order.

python results/cost_comparison/batch_check.py DATA_ROOT OUT_JSON
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantize.fast_act import quant_per_document  # noqa: E402
from quantize.quantizer import quant_nvfp4_4over6  # noqa: E402
from run_math_code_calibration import load_model, math_code_data  # noqa: E402
from run_multiround import data_paths  # noqa: E402


@torch.no_grad()
def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    torch.backends.cuda.matmul.allow_tf32 = False
    calibration, _ = data_paths('llama8b', root)
    prior = json.loads((calibration / 'report.json').read_text())
    model, modules = load_model(prior, False)
    model.set_attn_implementation('sdpa')
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    fit, _ = math_code_data(tok, prior['fit'])
    train = [b for s in ('math', 'code') for b in fit[s]]
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(0)).tolist()[:8]
    ids = torch.cat([train[i] for i in order]).cuda()

    def logprobs(batch):
        return torch.cat([model(input_ids=ids[i:i + batch], use_cache=False).logits[:, :-1].float().log_softmax(-1)
                          for i in range(0, 8, batch)])

    captured = {}

    def capture(name):
        def hook(module, inputs):
            captured.setdefault(name, []).append(inputs[0].detach().float().cpu())
        return hook

    result = {}
    t1 = logprobs(1)
    watched = {n: modules[n] for n in list(modules)[:14] + list(modules)[-7:]}
    handles = [m.register_forward_pre_hook(capture(n)) for n, m in watched.items()]
    t8 = logprobs(8)
    for h in handles:
        h.remove()
    b8_inputs = {n: v[0] for n, v in captured.items()}
    captured.clear()
    handles = [m.register_forward_pre_hook(capture(n)) for n, m in watched.items()]
    logprobs(1)
    for h in handles:
        h.remove()
    b1_inputs = {n: torch.cat(v) for n, v in captured.items()}
    result['bf16'] = dict(max_abs_logprob_diff=float((t1 - t8).abs().max()),
                          kl_b1_vs_b8=[float(x) for x in (t1.exp() * (t1 - t8)).sum(-1).mean(-1)])
    result['bf16_linear_inputs_max_abs_diff'] = {n: float((b1_inputs[n] - b8_inputs[n]).abs().max()) for n in watched}
    del b1_inputs, b8_inputs, captured
    teacher = t1
    for m in modules.values():
        m.weight.copy_(quant_nvfp4_4over6(m.weight, 4, 16))
    batch_now = [1]

    def act(module, inputs):
        x = inputs[0]
        return (quant_per_document(x).reshape(x.shape), *inputs[1:])
    handles = [m.register_forward_pre_hook(act) for m in modules.values()]
    for batch in (1, 2, 4, 8):
        lp = logprobs(batch)
        result[f'w4a4_kl_vs_teacher_batch{batch}'] = [float(x) for x in (teacher.exp() * (teacher - lp)).sum(-1).mean(-1)]
    for h in handles:
        h.remove()
    result['order'] = order
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

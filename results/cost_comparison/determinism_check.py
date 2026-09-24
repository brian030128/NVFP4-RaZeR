"""Diagnostic: are run_multiround.py's KL scoring gradients deterministic on this GPU?

One scoring batch (the first 8 fit sequences, score batch 8) at the FourOverSix start
point, run_multiround.py's scoring forward (causal per-token activation factors,
straight-through) and KL loss. For every Linear the weight gradient of the KL backward,
sum_i dy_i^T x_i in FP32, is hashed. Repeats: (a) CE backward then KL backward (B), twice;
(b) KL backward only (B'), twice. Also records the attention kernels that run in backward.

python results/cost_comparison/determinism_check.py DATA_ROOT OUT_JSON [--deterministic]
"""
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantize.causal_four_over_six import quantize_rows  # noqa: E402
from quantize.quantizer import quant_nvfp4_4over6  # noqa: E402
from run_math_code_calibration import load_model, math_code_data  # noqa: E402
from run_multiround import data_paths  # noqa: E402


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    deterministic = len(sys.argv) > 3 and sys.argv[3] == '--deterministic'
    if deterministic:
        # Needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment for deterministic cuBLAS.
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    calibration, _ = data_paths('llama8b', root)
    prior = json.loads((calibration / 'report.json').read_text())
    model, modules = load_model(prior, False)
    model.set_attn_implementation('sdpa')
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    fit, _ = math_code_data(tok, prior['fit'])
    ids = torch.cat([b for s in ('math', 'code') for b in fit[s]][:8]).cuda()
    with torch.no_grad():
        teacher = torch.cat([model(input_ids=ids[i:i + 1], use_cache=False).logits[:, :-1].float().log_softmax(-1)
                             .bfloat16() for i in range(8)]).float()
        for m in modules.values():
            m.weight.copy_(quant_nvfp4_4over6(m.weight, 4, 16))
    grads = {}

    def act(module, inputs):
        x = inputs[0]
        return (quantize_rows(x.detach()) + (x - x.detach()), *inputs[1:])

    def make_hook(n, phase):
        def forward(module, inputs, output):
            x = inputs[0].detach()

            def backward(dy):
                if phase[0] == 1:
                    g = torch.einsum('bto,btk->ok', dy.float(), x.float())
                    grads[n] = hashlib.sha256(g.cpu().numpy().tobytes()).hexdigest()
            output.register_hook(backward)
        return forward

    def run(with_ce):
        grads.clear()
        phase = [0]
        handles = [m.register_forward_pre_hook(act) for m in modules.values()]
        handles += [m.register_forward_hook(make_hook(n, phase)) for n, m in modules.items()]
        with torch.enable_grad():
            embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
            lp = model(inputs_embeds=embeds, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            ce = F.nll_loss(lp.transpose(1, 2), ids[:, 1:], reduction='none').mean(-1)
            kl = (teacher.exp() * (teacher - lp)).sum(-1).mean(-1)
            if with_ce:
                phase[0] = 0
                ce.sum().backward(retain_graph=True)
            phase[0] = 1
            kl.sum().backward()
        for h in handles:
            h.remove()
        return dict(grads)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        first = run(True)
    kernels = sorted({e.name for e in prof.events() if any(k in e.name.lower() for k in ('flash', 'fmha', 'attention', 'attn', 'sdpa'))})
    runs = dict(ce_then_kl_1=first, ce_then_kl_2=run(True), kl_only_1=run(False), kl_only_2=run(False))

    def same(a, b):
        return sum(runs[a][n] == runs[b][n] for n in modules)
    result = dict(modules=len(modules), attention_kernels=kernels,
                  identical_modules={f'{a} vs {b}': same(a, b) for a, b in (
                      ('ce_then_kl_1', 'ce_then_kl_2'), ('kl_only_1', 'kl_only_2'),
                      ('ce_then_kl_1', 'kl_only_1'), ('ce_then_kl_2', 'kl_only_2'))},
                  torch=torch.__version__, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  sdp_flash=torch.backends.cuda.flash_sdp_enabled(), sdp_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                  sdp_cudnn=torch.backends.cuda.cudnn_sdp_enabled(), sdp_math=torch.backends.cuda.math_sdp_enabled())
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

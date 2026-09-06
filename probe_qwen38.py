"""Native Qwen3.8 integration checks. Run only in a two-H100 Slurm job."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('All compute must run through Slurm.')
for key, subdir in [('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch'),
                    ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
    os.environ.setdefault(key, os.path.join(os.environ['HF_HOME'], subdir))
import contextlib
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from quantize import QuantConfig
from quantize.quantizer import quant_act, quant_nvfp4, quant_nvfp4_4over6, quant_mix_4_6
import analyze_task_sensitivity as task


def native_loss(model, ids, use_cache=False):
    ids = ids.to(model.get_input_embeddings().weight.device)
    logits = model(ids, use_cache=use_cache).logits
    labels = ids[:, 1:].reshape(-1).to(logits.device)
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels)


def native_wikitext(tokenizer):
    from datasets import load_dataset
    data = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    return tokenizer('\n\n'.join(data['text']), return_tensors='pt').input_ids


class NativeActivationQuantization:
    """Quantize each text GEMM input; STE affects backward only.

    Recurrent state, depthwise convolution, normalization, embedding, head,
    and the unused vision tower stay at their native precision.
    """
    def __init__(self, modules, config):
        self.config = config
        self.ste = False
        self.handles = [m.register_forward_pre_hook(self.pre) for m in modules.values()]

    def pre(self, module, inputs):
        x = inputs[0]
        q = quant_act(x, self.config)
        if self.ste:
            q = task.IdentityBackward.apply(x, q)
        return (q, *inputs[1:])

    @contextlib.contextmanager
    def backward(self):
        previous = self.ste
        self.ste = True
        try:
            yield
        finally:
            self.ste = previous

    def close(self):
        for h in self.handles:
            h.remove()


def text_modules(model):
    return {n: m for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and 'language_model' in n and 'head' not in n}


@torch.no_grad()
def apply_native_type_map(model, spec):
    """Rewrite PRISTINE native text weights; leaves the vision tower untouched."""
    assert spec['scope'] == 'qwen3_5_text_linear'
    assert spec['weight_type_block'] == [8, 64] and spec['scale_block'] == 16
    assert spec['default'] == 'E2M1'
    baseline = spec.get('baseline_weight_dtype', 'nvfp4')
    assert baseline in ('nvfp4', 'nvfp4_4over6')
    if baseline == 'nvfp4':
        assert spec['alpha'] == 1
    else:
        assert spec['e2m1_alphas'] == [1., 1.5] and spec['e0m3_alpha'] == 1.
    assert getattr(model.config, '_commit_hash', None) == spec['model_commit']
    modules = text_modules(model)
    assert modules and set(spec['modules']) <= set(modules)
    for n, m in modules.items():
        w = m.weight.detach()
        q = (quant_nvfp4_4over6 if baseline == 'nvfp4_4over6' else quant_nvfp4)(w, 4, 16)
        if n in spec['modules']:
            entry = spec['modules'][n]
            shape = [w.shape[0]//8, w.shape[1]//64]
            assert entry['tile_grid_shape'] == shape
            indices = entry['e0m3_flat_indices']
            assert len(indices) == len(set(indices))
            mask = torch.zeros(shape[0]*shape[1], dtype=torch.bool, device=w.device)
            assert all(0 <= i < mask.numel() for i in indices)
            mask[indices] = True
            alt = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            q = torch.where(task.expand_mask(mask.reshape(shape)), alt, q)
        m.weight.copy_(q)


def main():
    out = Path('results/task_sensitivity_qwen38_probe')
    out.mkdir(exist_ok=True)
    assert torch.cuda.device_count() == 2
    assert all('H100' in torch.cuda.get_device_name(i) for i in range(2))
    torch.set_num_threads(12)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260909)
    task.loss = native_loss
    fixture = torch.nn.Module()
    fixture.language_model = torch.nn.Linear(128, 64, bias=False, device='cuda:0', dtype=torch.bfloat16)
    fixture.visual = torch.nn.Linear(128, 64, bias=False, device='cuda:0', dtype=torch.bfloat16)
    fixture.lm_head = torch.nn.Linear(128, 64, bias=False, device='cuda:0', dtype=torch.bfloat16)
    fixture.config = SimpleNamespace(_commit_hash='fixture')
    pristine = fixture.language_model.weight.detach().clone()
    visual = fixture.visual.weight.detach().clone()
    head = fixture.lm_head.weight.detach().clone()
    spec = {'scope': 'qwen3_5_text_linear', 'model_commit': 'fixture',
            'weight_type_block': [8,64], 'scale_block': 16, 'alpha': 1., 'default': 'E2M1',
            'modules': {'language_model': {'tile_grid_shape': [8,2], 'e0m3_flat_indices': [5]}}}
    apply_native_type_map(fixture, spec)
    expected = quant_nvfp4(pristine, 4, 16)
    alternative = quant_mix_4_6(pristine, 4, 16, type_block=(8,64), clip='a1', elect='always')
    expected[16:24,64:128] = alternative[16:24,64:128]
    assert torch.equal(fixture.language_model.weight, expected)
    assert torch.equal(fixture.visual.weight, visual) and torch.equal(fixture.lm_head.weight, head)
    del fixture, pristine, visual, head, expected, alternative
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'transformers': transformers.__version__,
              'torch': torch.__version__, 'model': 'Qwen/Qwen3.8-27B', 'complete': False}
    task.atomic_json(out/'report.json', report)
    native_source = inspect.getsource(native)
    (out/'native_model_source.py').write_text(native_source)
    cfg = AutoConfig.from_pretrained(report['model'])
    assert cfg.text_config.output_gate_type == 'swish'
    report['model_commit'] = getattr(cfg, '_commit_hash', None)
    task.atomic_json(out/'model_config.json', cfg.to_dict())
    tok = AutoTokenizer.from_pretrained(report['model'], revision=report['model_commit'])
    fit, val, probe, hashes = task.data_splits(tok, 2048, 64, 16, 20260909, dataset_name='Salesforce/wikitext')
    report['data_sha256'] = hashes
    assert native_wikitext(tok).shape[1] >= 2048
    # Verify the native hybrid block backward on a small model before downloading weights.
    tiny = cfg.text_config.to_dict()
    tiny.update(hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                linear_num_key_heads=4, linear_num_value_heads=8,
                linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=512,
                bos_token_id=0, eos_token_id=1, pad_token_id=0,
                layer_types=['linear_attention']*3+['full_attention'])
    tiny_model = native.Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tiny)).to('cuda:0', dtype=torch.bfloat16).eval()
    # SGLang's Qwen3.5 implementation applies output_gate_type to the
    # GatedDeltaNet RMSNorm gate. Its "swish" is native Transformers' SiLU;
    # the separate full-attention output gate correctly remains sigmoid.
    norm = tiny_model.model.layers[0].linear_attn.norm
    assert norm.activation in ('silu', 'swish')
    x = torch.randn(4, 16, device='cuda:0', dtype=torch.bfloat16)
    g = torch.randn_like(x)
    normalized = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.variance_epsilon)).to(x.dtype)
    expected_gate = (norm.weight * normalized * torch.nn.functional.silu(g.float())).to(x.dtype)
    assert torch.equal(norm(x, g), expected_gate)
    report['swish_gate_verified'] = True
    tiny_modules = {n: m for n, m in tiny_model.named_modules()
                    if isinstance(m, torch.nn.Linear) and 'head' not in n}
    # Tiny gate projections need not obey production tile sizes; only test activation/backward here.
    ids = torch.randint(0, 512, (1, 128), device='cuda:0')
    qconfig = QuantConfig(a_bits=16, a_dtype='nvfp4_4over6', a_groupsize=16)
    with torch.no_grad():
        native_nll = task.loss(tiny_model, ids).item()
    wrapper = NativeActivationQuantization(tiny_modules, qconfig)
    with torch.no_grad():
        assert task.loss(tiny_model, ids).item() == native_nll
    qconfig.a_bits = 4
    with torch.no_grad():
        plain = task.loss(tiny_model, ids).item()
        with wrapper.backward():
            ste = task.loss(tiny_model, ids).item()
    assert plain == ste
    with wrapper.backward():
        l = task.loss(tiny_model, ids)
        l.backward()
    assert all(m.weight.grad is not None and torch.isfinite(m.weight.grad).all() for m in tiny_modules.values())
    wrapper.close()
    del tiny_model, tiny_modules, wrapper, l, ids
    torch.cuda.empty_cache()
    report['tiny_hybrid_forward_backward_passed'] = True
    task.atomic_json(out/'report.json', report)
    print('TINY HYBRID CHECK PASSED', flush=True)
    start = time.time()
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        report['model'], revision=report['model_commit'], dtype=torch.bfloat16,
        device_map='balanced', max_memory={0: '65GiB', 1: '65GiB'},
        output_loading_info=True)
    assert not loading['missing_keys'], loading['missing_keys']
    # Transformers 5 returns sets in loading_info; normalize metadata before
    # atomic JSON writes so successful model checks cannot fail serialization.
    loading = json.loads(json.dumps(loading, default=lambda x: sorted(x) if isinstance(x, set) else str(x)))
    report['loading_info'] = {k: v for k, v in loading.items() if k != 'unexpected_keys'}
    report['unexpected_keys'] = loading.get('unexpected_keys', [])
    model.eval()
    report['load_seconds'] = time.time()-start
    report['device_map'] = model.hf_device_map
    task.atomic_json(out/'report.json', report)
    modules = text_modules(model)
    assert modules
    report['quantized_shapes'] = {n: list(m.weight.shape) for n, m in modules.items()}
    assert all(m.weight.shape[0] % 8 == 0 and m.weight.shape[1] % 64 == 0 for m in modules.values())
    base, alt = {}, {}
    with torch.no_grad():
        for i, (n, m) in enumerate(modules.items()):
            w = m.weight.detach()
            b = quant_nvfp4(w, 4, 16)
            b2 = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='never')
            if not torch.equal(b, b2):
                bad = (b != b2).nonzero()[:8]
                diagnostic = {'module': n, 'shape': list(w.shape), 'dtype': str(w.dtype),
                              'max_abs_weight': w.abs().max().item(),
                              'finite_weight': bool(torch.isfinite(w).all()),
                              'finite_nvfp4': bool(torch.isfinite(b).all()),
                              'finite_mix': bool(torch.isfinite(b2).all()),
                              'different_elements': int((b != b2).sum()),
                              'examples': [{'index': idx.tolist(), 'weight': w[tuple(idx)].item(),
                                            'nvfp4': b[tuple(idx)].item(), 'mix': b2[tuple(idx)].item()}
                                           for idx in bad]}
                torch.save(w.cpu(), out/'mismatch_weight.pt')
                # Permit NaNs in this diagnostic only, to preserve evidence of quantizer failures.
                (out/'quantizer_mismatch.json').write_text(json.dumps(diagnostic, indent=2))
                raise AssertionError(f'Quantizer mismatch recorded for {n}')
            a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
            base[n], alt[n] = b.cpu(), a.cpu()
            m.weight.copy_(b)
            if (i+1) % 32 == 0:
                print(f'CANDIDATES {i+1}/{len(modules)}', flush=True)
    del a, b, b2, w
    wrapper = NativeActivationQuantization(modules, qconfig)
    task.activation_ste = wrapper.backward
    with torch.no_grad():
        plain = task.loss(model, fit[0]).item()
        with wrapper.backward():
            ste = task.loss(model, fit[0]).item()
    assert plain == ste
    report['real_forward_equal'] = True
    report['first_w4a4_nll'] = plain
    print(f'REAL W4A4 FORWARD nll={plain:.6f}; STE forward exact', flush=True)
    task.atomic_json(out/'report.json', report)
    start = time.time()
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        means, ses, losses = task.score_tiles(model, modules, base, alt, fit[:2])
    report['two_sequence_backward_seconds'] = time.time()-start
    report['backward_nll'] = losses
    report['forward_backward_nll_equal'] = plain == losses[0]
    assert report['forward_backward_nll_equal']
    report['peak_gpu_gib'] = [torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)]
    report['complete'] = True
    task.atomic_json(out/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

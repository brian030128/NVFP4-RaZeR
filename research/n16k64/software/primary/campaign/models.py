"""Model registry, offline loading from the pinned HF cache, quantization scope and manifests."""
import hashlib
import json
import os
from pathlib import Path

import torch

REGISTRY = {
    'llama8b': dict(model_id='meta-llama/Llama-3.1-8B', revision='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b',
                    family='Llama (Meta)', panel='development', loader='causal_lm', n8_total=13631488, n16_total=6815744,
                    archived_calibration='results/math_code_adaptive/calibration_333779_llama8b'),
    'qwen4b': dict(model_id='Qwen/Qwen3-4B', revision='1cfa9a7208912126459214e8b04321603b3df60c',
                   family='Qwen (Alibaba)', panel='development', loader='causal_lm', n8_total=7096320, n16_total=3548160,
                   archived_calibration='results/math_code_adaptive/calibration_333779_qwen4b'),
    'qwen27b': dict(model_id='Qwen/Qwen3.8-27B', revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
                    family='Qwen (Alibaba)', panel='development', loader='qwen3_5_conditional', n8_total=47559680,
                    n16_total=23779840, archived_calibration='results/math_code_adaptive/calibration_333787_qwen27b'),
    'mistral7b': dict(model_id='mistralai/Mistral-7B-v0.3', revision='caa1feb0e54d415e2df31207e5f4e273e33509b1',
                      family='Mistral (Mistral AI)', panel='confirmatory', loader='causal_lm', n8_total=None, n16_total=None),
    'phi4': dict(model_id='microsoft/phi-4', revision='2db69c1c3e91a05d2c64a3185acfbaf36f744e25',
                 family='Phi (Microsoft)', panel='confirmatory', loader='causal_lm', n8_total=None, n16_total=None),
    'olmo2_13b': dict(model_id='allenai/OLMo-2-1124-13B', revision='3fefddc1bf18a30e1d9b91000271630718f2aa8b',
                      family='OLMo (AI2)', panel='confirmatory', loader='causal_lm', n8_total=None, n16_total=None),
}


def snapshot_path(key):
    spec = REGISTRY[key]
    hub = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ['HF_HOME']) / 'hub'))
    p = hub / ('models--' + spec['model_id'].replace('/', '--')) / 'snapshots' / spec['revision']
    if not p.is_dir():
        raise FileNotFoundError(f'pinned snapshot missing: {p}')
    return p


def load_tokenizer(key):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(snapshot_path(key)))


def load_model(key, attn='sdpa', device_map='cuda', max_memory=None, dtype=torch.bfloat16):
    import transformers
    spec = REGISTRY[key]
    path = str(snapshot_path(key))
    major = int(transformers.__version__.split('.')[0])
    kw = dict(attn_implementation=attn, device_map=device_map)
    kw['dtype' if major >= 5 else 'torch_dtype'] = dtype
    if max_memory is not None:
        kw['max_memory'] = max_memory
    if spec['loader'] == 'qwen3_5_conditional':
        from transformers import Qwen3_5ForConditionalGeneration as cls
    else:
        from transformers import AutoModelForCausalLM as cls
    model, info = cls.from_pretrained(path, output_loading_info=True, **kw)
    info = json.loads(json.dumps(info, default=lambda x: sorted(x) if isinstance(x, set) else str(x)))
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f'incomplete checkpoint load for {key}: {info}')
    model.eval().requires_grad_(False)
    return model, info


def scope(model, key):
    """Quantized modules: every text nn.Linear except the output head (archived scope rule)."""
    if REGISTRY[key]['loader'] == 'qwen3_5_conditional':
        return {n: m for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and 'language_model' in n and 'head' not in n}
    head = model.get_output_embeddings()
    return {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not head}


def tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def module_manifest(modules, with_weight_hash=True):
    entries = []
    for n, m in modules.items():
        e = dict(name=n, shape=list(m.weight.shape), dtype=str(m.weight.dtype), bias=m.bias is not None)
        if with_weight_hash:
            e['weight_sha256'] = tensor_sha256(m.weight)
        entries.append(e)
    blob = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest(), entries


def tile_totals(modules, type_block):
    bm, bk = type_block
    bad = [n for n, m in modules.items() if m.weight.shape[0] % bm or m.weight.shape[1] % bk]
    total = sum((m.weight.shape[0] // bm) * (m.weight.shape[1] // bk) for m in modules.values())
    return total, bad


def tokenizer_manifest(key):
    p = snapshot_path(key)
    files = {}
    for f in sorted(p.iterdir()):
        if f.name.startswith('tokenizer') or f.name in ('vocab.json', 'merges.txt', 'special_tokens_map.json'):
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            files[f.name] = h
    blob = json.dumps(files, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest(), files

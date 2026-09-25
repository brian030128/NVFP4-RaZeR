"""Shared pieces of the native evaluations: pinned models, evaluation windows, fake-quant policies.

The windows and the fake-quant policy reproduce the N16K64 campaign exactly
(research/n16k64/software/primary/campaign/{data,quant,policies,tiles}.py on the research branch):
  * WikiText-2 test joined by blank lines, tokenized once, cut into 2048-token windows;
  * 256 C4 validation crops of 2048 tokens drawn with random.Random(0) from shard 00000;
  * fake quant = FourOverSix weights with E0M3 on the map's tiles, installed in BF16, plus a forward
    pre-hook quantizing every scoped Linear input with per-token FourOverSix (quantize_rows).
Window token hashes are compared with the hashes the repro branch recorded (eval/reference/).
"""
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

SM120 = Path(__file__).resolve().parents[1]
REPO = SM120.parent
for p in (str(SM120), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from mixfp4_sm120 import artifact as A  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import model as NM  # noqa: E402

MODELS = {
    'llama8b': dict(model_id='meta-llama/Llama-3.1-8B', revision='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b', loader='causal_lm'),
    'qwen4b': dict(model_id='Qwen/Qwen3-4B', revision='1cfa9a7208912126459214e8b04321603b3df60c', loader='causal_lm'),
    'mistral7b': dict(model_id='mistralai/Mistral-7B-v0.3', revision='caa1feb0e54d415e2df31207e5f4e273e33509b1', loader='causal_lm'),
    'phi4': dict(model_id='microsoft/phi-4', revision='2db69c1c3e91a05d2c64a3185acfbaf36f744e25', loader='causal_lm'),
    'qwen27b': dict(model_id='Qwen/Qwen3.8-27B', revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', loader='qwen3_5_conditional'),
}
WIKI = ('Salesforce/wikitext', 'b08601e04326c79dfdd32d625aee71d232d685c3', 'wikitext-2-raw-v1/test-00000-of-00001.parquet')
C4 = ('allenai/c4', '1588ec454efa1a09f29cd18ddd04fe05fc8653a2', 'en/c4-validation.00000-of-00008.json.gz')
REFERENCE = SM120 / 'eval' / 'reference'


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def snapshot(key):
    """The pinned snapshot directory in the local HF hub cache (never downloads)."""
    from huggingface_hub import constants
    spec = MODELS[key]
    p = Path(constants.HF_HUB_CACHE) / ('models--' + spec['model_id'].replace('/', '--')) / 'snapshots' / spec['revision']
    if not (p / 'config.json').exists():
        raise FileNotFoundError(f'pinned snapshot missing: {p} (download {spec["model_id"]} at {spec["revision"]})')
    return str(p)


def load_tokenizer(key):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snapshot(key))


def load_model(key, attn='sdpa', dtype=torch.bfloat16, device_map='cuda'):
    import transformers
    spec = MODELS[key]
    kw = dict(attn_implementation=attn, device_map=device_map)
    kw['dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'] = dtype
    if spec['loader'] == 'qwen3_5_conditional':
        from transformers import Qwen3_5ForConditionalGeneration as cls
    else:
        from transformers import AutoModelForCausalLM as cls
    model = cls.from_pretrained(snapshot(key), **kw)
    model.eval().requires_grad_(False)
    return model


def scope(model, key):
    return NM.scope(model, MODELS[key]['loader'])


def _hub_file(repo, revision, path):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type='dataset', revision=revision, local_files_only=True)


def wiki_windows(tok, length=2048):
    import pyarrow.parquet as pq
    lines = pq.read_table(_hub_file(*WIKI), columns=['text']).column('text').to_pylist()
    ids = tok('\n\n'.join(lines), return_tensors='pt').input_ids
    usable = ids.shape[1] // 2048 * 2048
    return [ids[:, i:i + length].clone() for i in range(0, usable, length)]


def c4_windows(tok, length=2048, count=256, seed=0):
    texts = []
    with gzip.open(_hub_file(*C4), 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)['text'])
    rng = random.Random(seed)
    crops = []
    for _ in range(count):
        while True:
            text = texts[rng.randint(0, len(texts) - 1)]
            tokens = tok(text, return_tensors='pt').input_ids
            if tokens.shape[1] > 2049:
                break
        offset = rng.randint(0, tokens.shape[1] - 2048 - 1)
        crop = tokens[:, offset:offset + 2048]
        crops.extend(crop[:, i:i + length].clone() for i in range(0, 2048, length))
    return crops


def windows(tok, key, domain, length=2048, check=True):
    wins = wiki_windows(tok, length) if domain == 'wiki' else c4_windows(tok, length)
    hashes = [sha(w) for w in wins]
    ref = REFERENCE / f'{key}_windows_{domain}.json'
    status = 'no reference'
    if check and ref.exists():
        want = json.loads(ref.read_text())['token_sha256']
        if hashes != want:
            raise RuntimeError(f'{key} {domain}: window token hashes differ from the campaign reference')
        status = 'identical to campaign reference'
    return wins, hashes, status


# ------------------------------------------------------------------------------ fake-quant policy

class FakeQuant:
    """Install campaign fake-quant weights + per-token activation pre-hooks on the scoped Linears."""

    def __init__(self, modules):
        self.modules = modules
        self.pristine = {n: m.weight.detach().to('cpu', copy=True) for n, m in modules.items()}
        self.handles = []

    @torch.no_grad()
    def install(self, kind, masks=None, type_block=None):
        from quantize.causal_four_over_six import quantize_rows
        self.remove()
        h = hashlib.sha256()
        for n, m in self.modules.items():
            w = self.pristine[n].to(m.weight.device)
            if kind == 'bf16':
                q = w
            else:
                q = A.fake_quant_weight(w, kind, None if masks is None else masks[n], type_block)
            m.weight.copy_(q)
            h.update(n.encode())
            h.update(q.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        if kind != 'bf16':
            act = quantize_rows if kind in ('map', 'four_over_six') else _nvfp4_rows
            for m in self.modules.values():
                self.handles.append(m.register_forward_pre_hook(lambda mod, inp, _a=act: (_chunked(_a, inp[0]), *inp[1:])))
        return h.hexdigest()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def restore(self):
        self.remove()
        for n, m in self.modules.items():
            m.weight.copy_(self.pristine[n].to(m.weight.device))


def _chunked(fn, x, max_rows=4096):
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    if flat.shape[0] <= max_rows:
        q = fn(x)
    else:
        q = torch.cat([fn(flat[i:i + max_rows]) for i in range(0, flat.shape[0], max_rows)]).reshape(shape)
    return q if q.dtype == x.dtype else q.to(x.dtype)


def _nvfp4_rows(x):
    from mixfp4_sm120 import numerics as N
    return N.fake_quant_act_rows(x, 'nvfp4_rows')


def read_map_for(key, map_path, model_modules=None):
    header, masks, digest = mapio.read_map(map_path)
    spec = MODELS[key]
    if header['model']['model_id'] != spec['model_id'] or header['model']['revision'] != spec['revision']:
        raise RuntimeError(f'map {map_path} is for {header["model"]}, not {spec}')
    if model_modules is not None:
        names = [m['name'] for m in header['modules']]
        if names != list(model_modules):
            raise RuntimeError('map module order differs from the model scope')
    return header, masks, digest

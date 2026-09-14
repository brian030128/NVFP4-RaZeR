"""
    Serve one of MIXFP4_REPORT.md's policies behind an OpenAI-compatible endpoint.

    Terminal-Bench drives an agent that talks to such an endpoint, so measuring a quantization
    policy there means serving it. The obvious route -- bake the weights into a checkpoint and
    hand it to vLLM -- cannot express these policies:

      * MixFP4 (k=3) is a MAP over 8x64 tiles, not a dtype, so there is nothing to write into a
        config; it has to be applied per module from the election.
      * every policy in that report is W4A4, and the activation half lives in forward hooks that
        `save_pretrained` does not carry. A served checkpoint would quietly be W4A16 -- a
        different method from the one being reported.

    So this serves the model in-process, with exactly the weights and the activation hooks
    `run_zeroshot_kse.py` installs. It is much slower than vLLM (plain HF generate, one request
    at a time), which is affordable only because the agent is capped; correctness is the point.

        python serve_quantized.py --model qwen4b --calib <dir> --policy k3 --port 18123
"""

import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.interacting_format import apply_mask
from quantize.quantizer import quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6
from run_adaptive_paper import FROZEN
from run_kse_paper import MODELS, STREAMED, elect_k

K = 3
POLICIES = ('bf16', 'nvfp4', 'four_over_six', f'k{K}')


def build(model_name, calib_dir, policy, stage_root, allow_drift, activation_hooks=True,
          map_path=None, save_map=None):
    """
        Load the model and install `policy`, returning (model, tokenizer, provenance).

        `activation_hooks=False` installs the weight half only. That is not the reported policy
        -- every row in MIXFP4_REPORT.md is W4A4 -- and exists solely so the weights can be
        written to a checkpoint for a serving stack that cannot run the hooks. The provenance
        records which half was applied, so a W4A16 measurement cannot be mistaken for a W4A4 one.
    """
    calib = Path(calib_dir)
    prior = json.loads((calib / 'report.json').read_text())   # source path and version pins
    assert prior['status'] == 'complete' and prior['maps_frozen']
    assert transformers.__version__ == prior['transformers_version']

    prov = dict(model=model_name, policy=policy, source=prior['source'], k=K,
                transformers_version=transformers.__version__,
                activation_quantized=bool(activation_hooks) and policy != 'bf16',
                precision='W4A4' if (activation_hooks and policy != 'bf16')
                          else ('BF16' if policy == 'bf16' else 'W4A16'))

    target = model_name in STREAMED
    if target:
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            prior['source'], dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                   and 'language_model' in n and 'head' not in n}
    else:
        model = AutoModelForCausalLM.from_pretrained(
            prior['source'], torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map='cuda')
        modules = {n: m for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    model.eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(prior['source'])

    if policy == f'k{K}' and map_path:
        # The election is a pure function of the frozen calibration, so it is computed once and
        # cached. Rebuilding it costs a 128-sequence scoring pass -- 25 minutes, and two GPUs on
        # the 27B because its calibration shards the model with device_map='balanced' -- to
        # arrive at the same few thousand tile indices every time.
        cached = torch.load(map_path, map_location='cpu', weights_only=False)
        assert cached['model'] == model_name and cached['k'] == K, cached
        tile_map = cached['map']
        assert set(tile_map) == set(modules), 'cached map does not match this model'
        prov.update(elected_tiles=cached['elected_tiles'], total_tiles=cached['total_tiles'],
                    frozen_map_reproduced=cached['frozen_map_reproduced'],
                    frozen_map_drift_tiles=cached['frozen_map_drift_tiles'],
                    map_source=str(map_path))
        print(f'loaded cached k={K} election from {map_path}: '
              f"{cached['elected_tiles']:,} tiles", flush=True)
        for n, m in modules.items():
            tile_map[n] = tile_map[n].reshape(m.weight.shape[0] // 8, m.weight.shape[1] // 64)
    elif policy == f'k{K}':
        uppers, slices, names = elect_k(calib, prior, (2, K))
        assert list(modules) == names
        flat = uppers[K] < 0
        tile_map = {n: flat[lo:hi].clone() for n, (lo, hi) in slices.items()}
        prov['elected_tiles'] = int(flat.sum())
        prov['total_tiles'] = int(flat.numel())

        # Same validation as the accuracy run: the 256-prefix must reproduce the shipped map.
        order = torch.argsort(uppers[2], stable=True)[:256]
        flat256 = torch.zeros(uppers[2].numel(), dtype=torch.bool)
        flat256[order[uppers[2][order] < 0]] = True
        shipped = Path(stage_root) / \
            f'results/math_code_adaptive/calibration_{MODELS[model_name]}_{model_name}/maps.json'
        bundle = json.loads((shipped if shipped.is_file() else calib / 'maps.json').read_text())
        drift = 0
        for n, m in modules.items():
            want = torch.zeros(tile_map[n].numel(), dtype=torch.bool)
            want[bundle['maps'][FROZEN][n]] = True
            got = flat256[slices[n][0]:slices[n][1]]
            drift += int((got != want).sum())
        prov['frozen_map_reproduced'] = drift == 0
        prov['frozen_map_drift_tiles'] = drift
        if drift:
            assert allow_drift, (f're-election differs from the shipped frozen map by {drift} '
                                 f'tile(s); pass --allow-map-drift to serve it anyway, labelled '
                                 f'as a re-derivation')
        if save_map:
            # Flat, before the per-module reshape, so a reload is independent of module shapes.
            Path(save_map).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model_name, k=K, map={n: v.clone() for n, v in tile_map.items()},
                            elected_tiles=prov['elected_tiles'],
                            total_tiles=prov['total_tiles'],
                            frozen_map_reproduced=prov['frozen_map_reproduced'],
                            frozen_map_drift_tiles=prov['frozen_map_drift_tiles'],
                            source=prior['source'], calibration=str(calib)), save_map)
            print(f'saved k={K} election to {save_map}', flush=True)
        for n, m in modules.items():
            tile_map[n] = tile_map[n].reshape(m.weight.shape[0] // 8, m.weight.shape[1] // 64)

    with torch.no_grad():
        for n, m in modules.items():
            if policy == 'bf16':
                continue
            if policy == 'nvfp4':
                w = quant_nvfp4(m.weight, 4, 16)
            else:
                w = quant_nvfp4_4over6(m.weight, 4, 16)
                if policy == f'k{K}' and bool(tile_map[n].any()):
                    alt = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1',
                                        elect='always')
                    w = apply_mask(w, alt, tile_map[n].to(m.weight.device))
                    del alt
            m.weight.copy_(w)
            del w

    if policy != 'bf16' and activation_hooks:
        q = quant_nvfp4 if policy == 'nvfp4' else quant_nvfp4_4over6

        def act(module, inputs):
            return (q(inputs[0], 4, 16), *inputs[1:])

        for m in modules.values():
            m.register_forward_pre_hook(act)
    return model, tok, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--policy', choices=POLICIES, required=True)
    ap.add_argument('--port', type=int, default=18123)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--served-name', default='qmodel')
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    ap.add_argument('--allow-map-drift', action='store_true')
    ap.add_argument('--map', dest='map_path', default=None,
                    help='Load the k=3 election from this file instead of rebuilding it from '
                         'the calibration scores. The election is a pure function of the frozen '
                         'calibration, so recomputing it per job is wasted work.')
    ap.add_argument('--save-map', default=None,
                    help='Write the election here after computing it, for later runs.')
    ap.add_argument('--provenance', default=None)
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm -- see /home/u4320956/CLAUDE.md'

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tok, prov = build(args.model, args.calib, args.policy, args.stage_root,
                             args.allow_map_drift, map_path=args.map_path,
                             save_map=args.save_map)
    print('PROVENANCE ' + json.dumps(prov), flush=True)
    if args.provenance:
        Path(args.provenance).write_text(json.dumps(prov, indent=2) + '\n')

    app = FastAPI()
    lock = threading.Lock()   # one GPU, one request at a time

    @app.get('/health')
    def health():
        return {'status': 'ok', 'policy': args.policy}

    @app.get('/v1/models')
    def models():
        return {'object': 'list',
                'data': [{'id': args.served_name, 'object': 'model', 'owned_by': 'local'}]}

    def generate(prompt_ids, max_tokens, temperature, stop):
        with lock, torch.no_grad():
            out = model.generate(
                input_ids=prompt_ids.to(model.device),
                max_new_tokens=min(max_tokens or args.max_new_tokens, args.max_new_tokens),
                do_sample=temperature is not None and temperature > 0,
                temperature=temperature if temperature and temperature > 0 else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        finish = 'stop'
        for s in (stop or []):
            if s and s in text:
                text = text.split(s)[0]
                break
        else:
            if out.shape[1] - prompt_ids.shape[1] >= (max_tokens or args.max_new_tokens):
                finish = 'length'
        return text, finish, int(prompt_ids.shape[1]), int(out.shape[1] - prompt_ids.shape[1])

    @app.post('/v1/chat/completions')
    async def chat(body: dict):
        messages = body.get('messages', [])
        try:
            enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                          return_tensors='pt')
            # transformers 5 returns a BatchEncoding here where 4.x returned the tensor, and
            # passing the mapping to generate() fails deep inside with an opaque AttributeError
            # on .shape. Qwen3.8-27B pins transformers 5.16.1, so both shapes occur.
            ids = enc['input_ids'] if hasattr(enc, 'keys') else enc
            if isinstance(ids, list):
                ids = torch.tensor([ids] if not isinstance(ids[0], list) else ids)
        except Exception:
            # Models without a chat template still have to answer something coherent.
            flat = '\n'.join(f"{m.get('role')}: {m.get('content')}" for m in messages)
            ids = tok(flat + '\nassistant:', return_tensors='pt').input_ids
        text, finish, n_in, n_out = generate(ids, body.get('max_tokens'),
                                             body.get('temperature'), body.get('stop'))
        return JSONResponse({
            'id': f'chatcmpl-{int(time.time()*1000)}', 'object': 'chat.completion',
            'created': int(time.time()), 'model': args.served_name,
            'choices': [{'index': 0, 'finish_reason': finish,
                         'message': {'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': n_in, 'completion_tokens': n_out,
                      'total_tokens': n_in + n_out}})

    @app.post('/v1/completions')
    async def completions(body: dict):
        ids = tok(body.get('prompt', ''), return_tensors='pt').input_ids
        text, finish, n_in, n_out = generate(ids, body.get('max_tokens'),
                                             body.get('temperature'), body.get('stop'))
        return JSONResponse({
            'id': f'cmpl-{int(time.time()*1000)}', 'object': 'text_completion',
            'created': int(time.time()), 'model': args.served_name,
            'choices': [{'index': 0, 'text': text, 'finish_reason': finish}],
            'usage': {'prompt_tokens': n_in, 'completion_tokens': n_out,
                      'total_tokens': n_in + n_out}})

    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()

"""Inference GPU memory of one (model, policy), in its own process (results/inference_memory/PROTOCOL.md).

    python results/inference_memory/measure_memory.py --model llama8b --policy tc-16x64 --out JSON

Policies: bf16 | nvfp4@stock_wA | fo6@stock_wA | fo6@stock_wB | tc-8x64 | tc-16x64 | tc-256x64 (the Parts 2-3
artifacts; kernels auto_stock / stock_wB / n8k64_wB / auto). Static: every parameter and buffer by group, the
allocation after load and after install, and the freed check. Dynamic: the peaks of prefill 1x2048 and 4x2048
(sm120/bench/model.py prefill) and of batch-1 decode after a 2048-token prompt, 128 tokens (decode_graph over a
StaticCache; HF generate for Qwen3.8-27B, whose hybrid cache the SM120 decode does not support).
"""
import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402  (sm120/bench/common.py)
from mixfp4_sm120 import model as NM  # noqa: E402
from mixfp4_sm120.lib import Kernel, sf_buffer_size  # noqa: E402
from mixfp4_sm120.linear import NativeLinear, clear_quant_cache  # noqa: E402

GIB = 2 ** 30
ART = Path('/home/dev/n16k64_campaign/deploy_eval/artifacts')
POLICIES = {'bf16': (None, None), 'nvfp4@stock_wA': ('nvfp4', 'auto_stock'), 'fo6@stock_wA': ('fo6', 'auto_stock'),
            'fo6@stock_wB': ('fo6', 'stock_wB'), 'tc-8x64': ('tc_8x64', 'n8k64_wB'), 'tc-16x64': ('tc_16x64', 'auto'),
            'tc-256x64': ('tc_256x64', 'auto')}
PROMPT, GEN = 2048, 128


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group_of(name, scoped, head_name):
    module = name.rsplit('.', 1)[0]
    if module in scoped:
        return 'quantized_linears'
    if module == head_name:
        return 'lm_head'
    if 'embed_tokens' in name:
        return 'embeddings'
    if 'visual' in name:
        return 'vision'
    if 'linear_attn' in name:
        return 'linear_attention_other'
    if 'norm' in name:
        return 'norms'
    return 'other'


def tensor_bytes(model, scoped, head_name):
    """Bytes of every distinct parameter and buffer storage on the GPU, by group."""
    seen, out = set(), {}
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t is None or not t.is_cuda or t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        g = group_of(n, scoped, head_name)
        out[g] = out.get(g, 0) + t.numel() * t.element_size()
    return out


def block_accounting(model):
    """From the allocator's snapshot: active blocks not starting at any parameter or buffer (stray allocations, e.g. a
    lingering weight copy) and the slack of the owned blocks (block size minus the bytes of the tensor at that address:
    the caching allocator's rounding and unsplit remainders)."""
    own = {}
    for t in list(model.parameters()) + list(model.buffers()):
        if t is not None and t.is_cuda:
            own[t.data_ptr()] = max(own.get(t.data_ptr(), 0), t.untyped_storage().nbytes())
    blocks = [b for seg in torch.cuda.memory_snapshot() for b in seg['blocks'] if b['state'] == 'active_allocated']
    stray = [b for b in blocks if b['address'] not in own]
    return dict(active_blocks=len(blocks), block_bytes=sum(b['size'] for b in blocks), stray_blocks=len(stray),
                stray_bytes=sum(b['size'] for b in stray), stray_sizes=sorted(b['size'] for b in stray)[-8:],
                slack_bytes=sum(b['size'] - own[b['address']] for b in blocks if b['address'] in own))


def peak(fn):
    clear_quant_cache()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        extra = fn()
    except torch.OutOfMemoryError as e:      # recorded, as the Part R harness does
        extra = dict(error=f'OOM: {str(e).splitlines()[0][:200]}')
    torch.cuda.synchronize()
    res = dict(base_gib=base / GIB, peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
               peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
               increase_gib=(torch.cuda.max_memory_allocated() - base) / GIB, seconds=time.time() - t0)
    if isinstance(extra, dict):
        res.update(extra)
    return res


def cache_tensor_bytes(cache):
    """Bytes of every tensor reachable from a transformers cache object (its layers' keys, values, states)."""
    seen, total = set(), 0

    def walk(obj, depth=0):
        nonlocal total
        if depth > 4:
            return
        if torch.is_tensor(obj):
            if obj.is_cuda and obj.data_ptr() not in seen:
                seen.add(obj.data_ptr())
                total += obj.numel() * obj.element_size()
            return
        if isinstance(obj, (list, tuple)):
            for x in obj:
                walk(x, depth + 1)
        elif isinstance(obj, dict):
            for x in obj.values():
                walk(x, depth + 1)
        elif hasattr(obj, '__dict__') and not isinstance(obj, (torch.nn.Module, type)):
            for x in vars(obj).values():
                walk(x, depth + 1)
    walk(cache)
    return total


def kv_static_bytes(config, length, batch=1):
    cfg = getattr(config, 'text_config', config)
    head_dim = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * batch * cfg.num_key_value_heads * length * head_dim * 2


def scratch_bytes(t, k):
    """NativeLinear's activation-quantization scratch for t tokens of width k (linear.py: codes | scales | token gs)."""
    off_sf = (t * k // 2 + 255) // 256 * 256
    off_gs = (off_sf + sf_buffer_size(t, k) + 255) // 256 * 256
    return off_gs + 4 * t


def workspace_bytes():
    total = 0
    for k in list(Kernel._cache.values()):
        for ws in getattr(k, '_ws', {}).values():
            if ws is not None:
                total += ws.numel() * ws.element_size()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--policy', required=True, choices=tuple(POLICIES))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    torch.backends.cuda.matmul.allow_tf32 = False
    C = load_path('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    BM = load_path('sm120_bench_model', REPO / 'sm120' / 'bench' / 'model.py')
    loader = C.MODELS[args.model]['loader']
    res = dict(model=args.model, policy=args.policy, gpu=B.gpu_info(), allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(unset: default)'),
               torch=torch.__version__, prompt=PROMPT, gen=GEN)
    t0 = time.time()
    model = C.load_model(args.model)
    torch.cuda.synchronize()
    res['load_seconds'] = time.time() - t0
    head = model.get_output_embeddings()
    head_name = next(n for n, m in model.named_modules() if m is head)
    scoped = set(C.scope(model, args.model))
    ks = {n: m.in_features for n, m in model.named_modules() if n in scoped}
    res['static'] = dict(allocated_after_load_gib=torch.cuda.memory_allocated() / GIB,
                         bf16_groups_bytes=tensor_bytes(model, scoped, head_name), blocks_after_load=block_accounting(model))
    bf16_scoped = res['static']['bf16_groups_bytes']['quantized_linears']
    artifact, kernel = POLICIES[args.policy]
    if artifact is not None:
        before = torch.cuda.memory_allocated()
        rep = NM.install(model, str(ART / f'{args.model}_{artifact}'), kernel=kernel, loader=loader)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        installed = rep.device_bytes['packed'] + rep.device_bytes['scales_placed'] + rep.device_bytes['bias']
        groups = tensor_bytes(model, scoped, head_name)
        cov = NM.coverage(model)
        res['install'] = dict(artifact=str(ART / f'{args.model}_{artifact}'), kernel=kernel, report=rep.as_dict(),
                              native_modules=len(NM.native_modules(model)))
        res['static'].update(
            allocated_after_install_gib=after / GIB, groups_bytes=groups, installed_bytes=installed,
            expected_after_install_bytes=before - bf16_scoped + installed,
            freed_check_difference_bytes=after - (before - bf16_scoped + installed),
            allocated_not_in_parameters_or_buffers_bytes=after - sum(groups.values()), blocks_after_install=block_accounting(model),
            scoped_bf16_linears_remaining=[n for n in cov['remaining_bf16_linears'] if n in scoped],
            quantized_ratio_to_bf16=installed / bf16_scoped, model_ratio_to_bf16=after / before)
        assert not res['static']['scoped_bf16_linears_remaining'], res['static']['scoped_bf16_linears_remaining'][:3]
        assert res['static']['blocks_after_install']['stray_blocks'] == 0, res['static']['blocks_after_install']
    else:
        res['static'].update(allocated_after_install_gib=res['static']['allocated_after_load_gib'],
                             groups_bytes=res['static']['bf16_groups_bytes'])
    print('STATIC', json.dumps({k: v for k, v in res['static'].items() if not isinstance(v, dict)}), flush=True)
    res['dynamic'] = {}
    for spec in ('1x2048', '4x2048'):
        b, p = (int(v) for v in spec.split('x'))
        res['dynamic'][f'prefill_{spec}'] = peak(lambda: BM.prefill(model, b, p, reps=1) and None)
        res['dynamic'][f'prefill_{spec}']['logits_bytes'] = b * p * head.out_features * 2
        res['dynamic'][f'prefill_{spec}']['scratch_bytes_max'] = max(scratch_bytes(b * p, k) for k in ks.values()) if artifact else 0
        print('PEAK', spec, json.dumps(res['dynamic'][f'prefill_{spec}']), flush=True)
    if loader == 'qwen3_5_conditional':
        def decode():
            ids = torch.randint(100, 20000, (1, PROMPT), device='cuda', generator=torch.Generator('cuda').manual_seed(1))
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=GEN, min_new_tokens=GEN, do_sample=False,
                                     return_dict_in_generate=True)
            return dict(harness='HF generate (greedy; the SM120 decode functions do not support the hybrid cache)',
                        generated=int(out.sequences.shape[1] - PROMPT), cache_bytes=cache_tensor_bytes(out.past_key_values))
    else:
        def decode():
            timing, toks = BM.decode_graph(model, 1, PROMPT, GEN)
            return dict(harness='sm120/bench/model.py decode_graph (CUDA graph, StaticCache)', ms_per_token=timing['ms_per_token'],
                        cache_bytes=kv_static_bytes(model.config, PROMPT + GEN + 8))
    res['dynamic']['decode_b1'] = peak(decode)
    res['dynamic']['decode_b1']['scratch_bytes_max'] = max(scratch_bytes(1, k) for k in ks.values()) if artifact else 0
    print('PEAK decode', json.dumps(res['dynamic']['decode_b1']), flush=True)
    res['kernel_workspace_bytes'] = workspace_bytes()
    if artifact is not None:
        res['install']['calls'] = sum(m.calls for m in NM.native_modules(model).values())
        kset = next(iter(NM.native_modules(model).values())).kernel_set
        if kset is not None:
            res['install']['kernel_set'] = kset.describe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()

"""Native SM100 reasoning evaluation (RaZeR CoT benchmark) of Llama-3.1-8B-Instruct
(--model llama8b_ins) or Qwen3.8-27B (--model qwen27b).

Every text linear layer runs on the packed-FP4 MixFP4 kernel (libmixfp4_model.so):
FourOverSix E2M1 weights, with E0M3 on the tiles of a frozen 256x64 map for the
MixFP4 policy. --policy bf16 runs the unquantized model through the same harness.
Qwen uses its native Transformers 5.16.1 code (all 496 text linears, including the
linear-attention projections, as in calibration), with thinking disabled in the chat
template and an explicit context length (lm-eval cannot read it from Qwen's nested
config and would otherwise truncate the few-shot prompt to 2048 tokens). The kernel quantizes activations itself (FourOverSix, one scale per
call); evaluation uses batch size 1, so each scale covers one sequence at prefill and
one token at decode, matching the per-document fake-quant protocol.

Gates before any evaluation: the loaded library must match a passed kernel gate, and
every packed matrix must decode bitwise to its fake-quant reference.

Tasks come from lm-eval 0.4.9.2 (gsm8k_llama, mmlu_cot_llama) with the chat template
and few-shot as multi-turn, exactly as run_llama_cot.py calls them. --shard i/N splits
each task's documents so a policy fits gb200-dev's 2-hour cap.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from native_model_runtime import Linear, Runtime, decode
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6

PRIORS = {'llama8b_ins': Path('/work/u4320956/mixfp4_potential/llama8b_ins_calibration/report.json'),
          'qwen27b': Path('/work/u4320956/task_reorder/pilot_20260919/qwen27b/calibration/report.json')}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class NativeLinear(torch.nn.Module):
    """Calls the native kernel. The kernel keeps one plan (and output buffer) per row
    count; only the 1-row decode plan is kept, so varying prompt lengths do not
    accumulate buffers. The kernel needs N and K to be multiples of 256; a matrix with
    fewer output rows (Qwen's 48-row in_proj_a/in_proj_b) is packed with zero rows up to
    256 and only its first `out` output columns are returned. Rows are quantized
    independently (16-element scale blocks within a row, global scale from the real
    weights, activation quantization independent of N), so this is exact."""

    def __init__(self, lin, out):
        super().__init__()
        self.lin, self.out = lin, out

    def forward(self, x):
        x = x.contiguous()
        m = x.numel() // self.lin.k
        y = self.lin(x)
        y = y[..., :self.out].contiguous() if self.out != self.lin.n else y.clone()
        if m != 1:
            torch.cuda.current_stream().synchronize()
            p, _ = self.lin.plans.pop(m)
            self.lin.runtime.lib.mf_destroy(p)
        return y


@torch.inference_mode()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    import transformers
    import lm_eval_compat  # noqa: F401  (needed before lm_eval under transformers 5; no-op on 4)
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager, get_task_dict
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lib', required=True)
    ap.add_argument('--kernel-gate', type=Path, required=True)
    ap.add_argument('--model', choices=tuple(PRIORS), default='llama8b_ins')
    ap.add_argument('--policy', choices=('bf16', 'four_over_six', 'mix256x64'), required=True)
    ap.add_argument('--map', type=Path, help='Frozen 256x64 map for --policy mix256x64')
    ap.add_argument('--tasks', default='gsm8k_llama')
    ap.add_argument('--shard', default='0/1', help='i/N: evaluate documents i, i+N, i+2N, ...')
    ap.add_argument('--limit', type=int, default=None, help='Smoke runs: first N documents per task')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert torch.cuda.get_device_capability(0)[0] == 10, 'Native path needs SM100'
    torch.backends.cuda.matmul.allow_tf32 = False
    assert json.loads((args.kernel_gate / 'report.json').read_text())['status'] == 'passed'
    assert digest(args.lib) == json.loads((args.kernel_gate / 'library.json').read_text())['library_sha256']
    qwen = args.model == 'qwen27b'
    assert (args.map is not None) == (args.policy == 'mix256x64')
    prior = json.loads(PRIORS[args.model].read_text())
    if qwen:
        assert transformers.__version__ == prior['transformers_version'], transformers.__version__
    tasks = [t for t in args.tasks.split(',') if t]
    shard, shards = (int(v) for v in args.shard.split('/'))
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(status='preparing', job=os.environ['SLURM_JOB_ID'], device=torch.cuda.get_device_name(0),
                  model=args.model, transformers=transformers.__version__,
                  policy=args.policy, tasks=tasks, shard=args.shard, limit=args.limit, batch_size=1,
                  library_sha256=digest(args.lib), source=prior['source'], revision=prior['revision'],
                  lm_eval=lm_eval.__version__ if hasattr(lm_eval, '__version__') else None,
                  map=str(args.map) if args.map else None, map_sha256=digest(args.map) if args.map else None)
    save = lambda: (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    masks = torch.load(args.map, map_location='cpu', weights_only=True) if args.policy == 'mix256x64' else {}
    checkpoint = Path(os.environ['HF_HOME']) / 'hub' / ('models--' + prior['source'].replace('/', '--')) / 'snapshots' / prior['revision']
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    if qwen:
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(checkpoint), dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa',
            local_files_only=True).eval()
        linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                   and 'language_model' in n and 'head' not in n]
        assert linears == list(prior['matrices'])
    else:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=torch.bfloat16, device_map='cuda',
                                                     attn_implementation='sdpa', local_files_only=True).eval()
    if masks:
        assert set(masks) == set(prior['matrices'])
    rt = Runtime(args.lib)
    started, tiles, pads = time.perf_counter(), 0, 0
    for name, meta in (prior['matrices'] if args.policy != 'bf16' else {}).items():
        w = model.get_submodule(name).weight.data
        assert tensor_sha(w) == meta['source_sha256'], name
        mask = masks.get(name)
        mask = mask if mask is not None and bool(mask.any()) else None
        rows, k = w.shape
        assert k % 256 == 0, (name, 'K must be a multiple of 256')
        padded = -rows % 256
        lin = Linear(rt, torch.nn.functional.pad(w, (0, 0, 0, padded)) if padded else w, mask)
        pads += bool(padded)
        ref = quant_nvfp4_4over6(w, 4, 16)
        if mask is not None:
            tiles += int(mask.sum())
            full = mask.cuda().repeat_interleave(256, 0).repeat_interleave(64, 1)[:w.shape[0], :w.shape[1]]
            ref = torch.where(full, quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always'), ref)
        got = decode(lin.packed, lin.flat_scales, lin.global_scale, None if lin.mask is None else lin.mask.bool())[:rows]
        assert torch.equal(got.to(torch.bfloat16), ref), (name, 'packing is not bitwise')
        model.set_submodule(name, NativeLinear(lin, rows))
        del w, ref, got
    packed = len(prior['matrices']) if args.policy != 'bf16' else 0
    report.update(status='packed', packed_matrices=packed, row_padded_matrices=pads, packed_bitwise=True, e0m3_tiles=tiles,
                  pack_seconds=time.perf_counter() - started)
    save()
    print(f'PACKED {packed} matrices ({pads} row-padded to 256), {tiles} E0M3 tiles, bitwise', flush=True)

    extra = {}
    if qwen:
        extra = dict(enable_thinking=False, max_length=model.config.get_text_config().max_position_embeddings)
    report['hflm'] = dict(backend='causal', **extra)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1, backend='causal', **extra)
    task_manager = TaskManager()
    samples = None
    if shards > 1:
        # Document indices of this shard, per task (lm-eval 0.4.9 `samples`).
        samples = {}
        for task_name, task in get_task_dict(tasks, task_manager).items():
            docs = task.eval_docs if hasattr(task, 'eval_docs') else None
            count = len(docs) if docs is not None else None
            assert count is not None, f'cannot shard {task_name}'
            samples[task_name] = list(range(shard, count, shards))
    started = time.perf_counter()
    full = lm_eval.simple_evaluate(model=lm, tasks=tasks, batch_size=1, limit=args.limit, samples=samples,
                                   apply_chat_template=True, fewshot_as_multiturn=True, bootstrap_iters=0,
                                   log_samples=True, task_manager=task_manager)
    report['eval_seconds'] = time.perf_counter() - started
    report['metrics'] = {t: {k: v for k, v in vals.items() if isinstance(v, (int, float))}
                         for t, vals in full['results'].items()}
    report['samples'] = {t: [{'doc_id': s.get('doc_id'), **{k: v for k, v in s.items() if k.startswith('exact_match')}}
                             for s in recs] for t, recs in (full.get('samples') or {}).items()}
    report['status'] = 'complete'
    save()
    for t in tasks:
        em = {k: round(v, 4) for k, v in report['metrics'].get(t, {}).items() if k.startswith('exact_match') and 'stderr' not in k}
        print(f'ACC {args.policy} {t} shard {args.shard} {json.dumps(em)} ({report["eval_seconds"]:.0f}s)', flush=True)


if __name__ == '__main__':
    main()

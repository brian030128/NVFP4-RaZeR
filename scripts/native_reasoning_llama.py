"""Native SM100 reasoning evaluation of Llama-3.1-8B-Instruct (RaZeR CoT benchmark).

Every text linear layer runs on the packed-FP4 MixFP4 kernel (libmixfp4_model.so):
FourOverSix E2M1 weights, with E0M3 on the tiles of a frozen 256x64 map for the
MixFP4 policy. The kernel quantizes activations itself (FourOverSix, one scale per
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

PRIOR = Path('/work/u4320956/mixfp4_potential/llama8b_ins_calibration/report.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class NativeLinear(torch.nn.Module):
    """Calls the native kernel. The kernel keeps one plan (and output buffer) per row
    count; only the 1-row decode plan is kept, so varying prompt lengths do not
    accumulate buffers."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        x = x.contiguous()
        m = x.numel() // self.lin.k
        y = self.lin(x).clone()
        if m != 1:
            torch.cuda.current_stream().synchronize()
            p, _ = self.lin.plans.pop(m)
            self.lin.runtime.lib.mf_destroy(p)
        return y


@torch.inference_mode()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager, get_task_dict
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lib', required=True)
    ap.add_argument('--kernel-gate', type=Path, required=True)
    ap.add_argument('--policy', choices=('four_over_six', 'mix256x64'), required=True)
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
    prior = json.loads(PRIOR.read_text())
    tasks = [t for t in args.tasks.split(',') if t]
    shard, shards = (int(v) for v in args.shard.split('/'))
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(status='preparing', job=os.environ['SLURM_JOB_ID'], device=torch.cuda.get_device_name(0),
                  policy=args.policy, tasks=tasks, shard=args.shard, limit=args.limit, batch_size=1,
                  library_sha256=digest(args.lib), source=prior['source'], revision=prior['revision'],
                  lm_eval=lm_eval.__version__ if hasattr(lm_eval, '__version__') else None,
                  map=str(args.map) if args.map else None, map_sha256=digest(args.map) if args.map else None)
    save = lambda: (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    masks = torch.load(args.map, map_location='cpu', weights_only=True) if args.policy == 'mix256x64' else {}
    checkpoint = Path(os.environ['HF_HOME']) / 'hub' / ('models--' + prior['source'].replace('/', '--')) / 'snapshots' / prior['revision']
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=torch.bfloat16, device_map='cuda',
                                                 attn_implementation='sdpa', local_files_only=True).eval()
    rt = Runtime(args.lib)
    started, tiles = time.perf_counter(), 0
    for name, meta in prior['matrices'].items():
        w = model.get_submodule(name).weight.data
        assert tensor_sha(w) == meta['source_sha256'], name
        mask = masks.get(name)
        mask = mask if mask is not None and bool(mask.any()) else None
        lin = Linear(rt, w, mask)
        ref = quant_nvfp4_4over6(w, 4, 16)
        if mask is not None:
            tiles += int(mask.sum())
            full = mask.cuda().repeat_interleave(256, 0).repeat_interleave(64, 1)[:w.shape[0], :w.shape[1]]
            ref = torch.where(full, quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always'), ref)
        got = decode(lin.packed, lin.flat_scales, lin.global_scale, None if lin.mask is None else lin.mask.bool())
        assert torch.equal(got.to(torch.bfloat16), ref), (name, 'packing is not bitwise')
        model.set_submodule(name, NativeLinear(lin))
        del w, ref, got
    report.update(status='packed', packed_bitwise=True, e0m3_tiles=tiles, pack_seconds=time.perf_counter() - started)
    save()
    print(f'PACKED {len(prior["matrices"])} matrices, {tiles} E0M3 tiles, bitwise', flush=True)

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1)
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

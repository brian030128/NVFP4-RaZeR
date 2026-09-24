"""KL-distillation baselines for the calibration-cost study (results/cost_comparison/PROTOCOL.md).

Arms, Llama-3.1-8B, W4A4 FourOverSix:
  qat    full-weight QAT: every text Linear weight (the 224 matrices run_multiround.py
         quantizes) is trainable; the forward pass applies FourOverSix fake quantization to
         the weights and per-document tensor-wide FourOverSix to the Linear inputs (the
         evaluation quantizers), with straight-through gradients.
  lora   LoRA-QAT: weights frozen, rank-r adapters on all 224 matrices merged before the
         same fake quantization, Q(W + B A).
  scale  scale-only distillation: weights frozen; a learned factor f on every 16-element
         E4M3 block scale, scale = e4m3(clamp(f * s0)) with s0 the FourOverSix pre-rounding
         scale (straight-through across the E4M3 rounding, LSQ gradient across the element
         rounding). Deployable as plain NVFP4.
Loss: per-sequence KL(BF16 teacher || student) averaged over tokens, as run_multiround.py,
averaged over the sequences of an optimizer step. Teacher: BF16 log-probs precomputed with
the unquantized model and held on the CPU in BF16 (run_multiround.py convention); with
--pool the teacher is precomputed chunk by chunk. --live-teacher (probes only) instead keeps
a second BF16 model on the GPU. Development and WikiText/C4 evaluation replicate
run_multiround.py exactly (eval_batch documents per development forward, per-document
activation scales, reference quantizers for the deployed weights).
"""
import argparse
import json
import math
import os
import socket
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from cost_monitor import GIB, PhaseMonitor
from quantize.fast_act import check as check_act, quant_per_document
from quantize.fused_fourover6 import (LearnedScaleE2M1, STEFourOverSix, fourover6, fourover6_block_scales,
                                      scaled_e2m1, verify_weights, E4M3_MAX, E4M3_MIN)
from quantize.quantizer import quant_nvfp4_4over6
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_math_code_calibration import load_model, math_code_data
from run_multiround import PUBLISHED, data_paths, load_development
from run_task_reorder_eval import validate_evaluation_data

SOURCES = ('run_cost_distill.py', 'cost_monitor.py', 'quantize/fused_fourover6.py', 'quantize/quantizer.py',
           'quantize/fast_act.py', 'run_multiround.py')
LEVELS = [0., .5, 1., 1.5, 2., 3., 4., 6.]


def scaled_dequant_reference(w, scales, gs):
    """Torch reference of scaled_e2m1: E2M1 round-to-nearest with given block scales."""
    ws = w.reshape(-1, 16).float() / gs
    levels = ws.new_tensor(LEVELS)
    q = levels[torch.bucketize((ws / scales[:, None]).abs().contiguous(), (levels[:-1] + levels[1:]) / 2,
                               right=False)] * ws.sign()
    return (q * scales[:, None] * gs).reshape(w.shape).to(torch.bfloat16)


def state_bytes(optimizer):
    """Exact bytes of optimizer state tensors by device (8-bit states count codes and scales)."""
    out = {}
    if optimizer is None:
        return out
    optimizers = list(optimizer.optim_dict.values()) + ([optimizer.d_opt] if optimizer.d_opt else []) \
        if hasattr(optimizer, 'optim_dict') else [optimizer]
    for opt in optimizers:
        for state in opt.state.values():
            for v in state.values():
                if not torch.is_tensor(v):
                    continue
                parts = [v.codes, v.scale, v.qmap] if hasattr(v, 'codes') else [v]
                for t in parts:
                    key = str(t.device)
                    out[key] = out.get(key, 0) + t.numel() * t.element_size()
    return out


def make_optimizer(kind, params, lr):
    kwargs = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    if kind == 'torch_adamw':
        return torch.optim.AdamW(params, fused=True, **kwargs)
    from torchao.optim import AdamW8bit, CPUOffloadOptimizer
    from torchao.optim.adam import _AdamBase

    class AdamWFP32State(_AdamBase):
        """torchao AdamW update (FP32 math, BF16 stochastic rounding) with FP32 m and v."""

        def __init__(self, params, lr, betas, eps, weight_decay, bf16_stochastic_round=True):
            super().__init__(params, lr, betas, eps, weight_decay, False, block_size=256,
                             bf16_stochastic_round=bf16_stochastic_round, is_adamw=True)

        @staticmethod
        def _subclass_zeros(p, signed, block_size):
            return torch.zeros(p.shape, dtype=torch.float32, device=p.device)

    if kind == 'adamw_fp32':
        return AdamWFP32State(params, **kwargs)
    if kind == 'adamw_8bit':
        return AdamW8bit(params, block_size=256, bf16_stochastic_round=True, **kwargs)
    if kind == 'cpu_offload':
        return CPUOffloadOptimizer(params, optimizer_class=AdamWFP32State, offload_gradients=False, **kwargs)
    raise ValueError(kind)


def make_evaluator(model, modules, device, eval_batch):
    """run_multiround.py's per_document_act, dev_eval and final evaluation, unchanged."""
    eval_handles = []
    act_checks = [0]
    current_batch = [1]

    def per_document_act(module, inputs):
        x = inputs[0]
        batch = current_batch[0]
        if batch == 1 or (x.dim() >= 3 and x.shape[0] == batch):
            view = x
        elif x.shape[0] % batch == 0 and x.shape[0] // batch > 1:
            view = x.reshape(batch, -1, x.shape[-1])
        else:
            raise RuntimeError(f'Cannot recover the document axis of a {tuple(x.shape)} input at batch {batch}')
        if act_checks[0] < 64:
            assert check_act(view), 'vectorized activation quantizer differs from quant_nvfp4_4over6'
            act_checks[0] += 1
        return (quant_per_document(view).reshape(x.shape), *inputs[1:])

    def per_sequence_losses(lp, ids, t):
        ce = F.nll_loss(lp.transpose(1, 2), ids[:, 1:].to(lp.device), reduction='none').mean(-1)
        kl = (t.exp() * (t - lp)).sum(-1).mean(-1)
        return ce, kl

    def eval_hooks(on):
        nonlocal eval_handles
        for h in eval_handles:
            h.remove()
        eval_handles = [m.register_forward_pre_hook(per_document_act) for m in modules.values()] if on else []

    def dev_eval(dev, dev_teacher):
        eval_hooks(True)
        ce, kl = [], []
        for start in range(0, len(dev), eval_batch):
            chunk = dev[start:start + eval_batch]
            ids = torch.cat([r['ids'] for r in chunk]).to(device)
            current_batch[0] = ids.shape[0]
            lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            t = torch.cat(dev_teacher[start:start + eval_batch]).to(lp.device).float()
            c, k = per_sequence_losses(lp, ids, t)
            ce.extend(c.tolist()); kl.extend(k.tolist())
            del lp, t
        current_batch[0] = 1
        eval_hooks(False)
        return dict(ce=sum(ce) / len(ce), kl=sum(kl) / len(kl), ce_nll=ce, kl_values=kl)

    def final_eval(batches):
        eval_hooks(True)
        evaluation = {}
        for domain, sequences in batches.items():
            values = []
            for ids in sequences:
                ids = ids.to(device)
                logits = model(input_ids=ids, use_cache=(domain == 'wiki')).logits
                values.append(float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                                    ids[:, 1:].reshape(-1).to(logits.device))))
                del logits
            losses = torch.tensor(values, dtype=torch.float32) * 2048
            key = 'c4' if domain == 'c4_paper' else domain
            evaluation[key] = dict(nll=values, ppl=float(torch.exp(losses.sum() / (len(values) * 2048))))
            print(f'PPL {key} {evaluation[key]["ppl"]:.6f}', flush=True)
        eval_hooks(False)
        return evaluation

    return dev_eval, final_eval


class Student:
    """Patched Linear forwards: fake-quantized training forward when `quantize` is on,
    the original BF16 forward otherwise (teacher passes and hook-based evaluation)."""

    def __init__(self, arm, modules, lora_rank=16, lora_alpha=16):
        self.arm, self.modules, self.quantize = arm, modules, False
        self.act_checks, self.micro_batch = 0, None
        self.factors, self.lora, self.meta = {}, {}, {}
        self.lora_scale = lora_alpha / lora_rank
        for name, m in modules.items():
            if arm == 'qat':
                m.weight.requires_grad_(True)
            elif arm == 'scale':
                gs, pre, _ = fourover6_block_scales(m.weight)
                self.meta[name] = (gs, pre)
                self.factors[name] = torch.nn.Parameter(torch.ones_like(pre))
            elif arm == 'lora':
                o, k = m.weight.shape
                a = torch.nn.Parameter(torch.empty(lora_rank, k, device=m.weight.device, dtype=torch.float32))
                torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                b = torch.nn.Parameter(torch.zeros(o, lora_rank, device=m.weight.device, dtype=torch.float32))
                self.lora[name] = (a, b)
            m.forward = self._forward(name, m, m.forward)

    def parameters(self):
        if self.arm == 'qat':
            return [m.weight for m in self.modules.values()]
        if self.arm == 'scale':
            return list(self.factors.values())
        return [p for ab in self.lora.values() for p in ab]

    def merged(self, name, m):
        a, b = self.lora[name]
        return (m.weight.float() + self.lora_scale * (b @ a)).to(torch.bfloat16)

    def weight(self, name, m):
        if self.arm == 'qat':
            return STEFourOverSix.apply(m.weight, 1)
        if self.arm == 'scale':
            gs, pre = self.meta[name]
            return LearnedScaleE2M1.apply(self.factors[name], m.weight, gs, pre).reshape(m.weight.shape)
        return STEFourOverSix.apply(self.merged(name, m), 1)

    def _forward(self, name, m, original):
        def forward(x):
            if not self.quantize:
                return original(x)
            assert x.dim() == 3 and x.shape[0] == self.micro_batch, tuple(x.shape)
            if self.act_checks < 64:
                assert torch.equal(fourover6(x.detach(), x.shape[0]).view(torch.int16),
                                   quant_per_document(x.detach()).view(torch.int16)), 'fused activation quantizer'
                self.act_checks += 1
            return F.linear(STEFourOverSix.apply(x, x.shape[0]), self.weight(name, m), m.bias)
        return forward

    @torch.no_grad()
    def deployed(self, name, m):
        """Deployed BF16 weight from the reference quantizers."""
        if self.arm == 'qat':
            return quant_nvfp4_4over6(m.weight, 4, 16)
        if self.arm == 'scale':
            gs, pre = self.meta[name]
            scales = (self.factors[name] * pre).clamp(min=E4M3_MIN, max=E4M3_MAX).to(torch.float8_e4m3fn).float()
            return scaled_dequant_reference(m.weight, scales, gs)
        return quant_nvfp4_4over6(self.merged(name, m), 4, 16)

    @torch.no_grad()
    def verify_fused(self):
        """The fused training quantizer equals the reference on the current parameters."""
        bad = []
        for name, m in self.modules.items():
            if self.arm == 'scale':
                gs, pre = self.meta[name]
                scales = (self.factors[name] * pre).clamp(min=E4M3_MIN, max=E4M3_MAX).to(torch.float8_e4m3fn).float()
                fused = scaled_e2m1(m.weight, scales, gs)
            else:
                fused = fourover6(m.weight if self.arm == 'qat' else self.merged(name, m), 1)
            if not torch.equal(fused.view(torch.int16), self.deployed(name, m).view(torch.int16)):
                bad.append(name)
        return bad


def per_step_kl(lp, t):
    return (t.exp() * (t - lp)).sum(-1).mean(-1)


def main():
    monitor = PhaseMonitor()
    monitor.enter('model_load')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', choices=('qat', 'lora', 'scale'), required=True)
    ap.add_argument('--optimizer', choices=('adamw_fp32', 'adamw_8bit', 'cpu_offload', 'torch_adamw'), default=None,
                    help='qat: adamw_fp32 / adamw_8bit / cpu_offload; lora and scale: torch_adamw (FP32 parameters)')
    ap.add_argument('--lr', type=float, default=0.0)
    ap.add_argument('--budget', choices=('zero', 'probe', 'c1', 'c2', 'pool'), required=True)
    ap.add_argument('--time-budget', type=float, default=None, help='c2: seconds from process start')
    ap.add_argument('--pool', type=Path, default=None, help='pool.pt from prepare_distill_pool.py')
    ap.add_argument('--pool-per-source', type=int, default=None, help='first N pool sequences of each source')
    ap.add_argument('--teacher-chunk', type=int, default=256)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--micro-batch', type=int, default=8)
    ap.add_argument('--checkpointing', action='store_true')
    ap.add_argument('--probe-steps', type=int, default=3)
    ap.add_argument('--live-teacher', action='store_true', help='probe: second BF16 model on the GPU as the teacher')
    ap.add_argument('--lora-rank', type=int, default=16)
    ap.add_argument('--eval-batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-state', action='store_true')
    ap.add_argument('--evaluate', type=Path, default=None, help='a finished run directory: deploy its state and run the '
                                                                'development and WikiText/C4 evaluation')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--transformers-deviation', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.evaluate is not None:
        trained = json.loads((args.evaluate / 'report.json').read_text())
        assert trained['status'] == 'complete' and trained['state_sha256'], args.evaluate
        for key in ('arm', 'optimizer', 'lr', 'lora_rank'):
            setattr(args, key, trained['config'][key])
    if args.optimizer is None:
        args.optimizer = 'adamw_fp32' if args.arm == 'qat' else 'torch_adamw'
    assert (args.arm == 'qat') == (args.optimizer != 'torch_adamw'), 'qat uses a BF16-weight optimizer'
    assert args.batch % args.micro_batch == 0
    calibration, development = data_paths('llama8b', args.data_root)
    prior = json.loads((calibration / 'report.json').read_text())
    deviations = []
    if transformers.__version__ != prior['transformers_version']:
        assert args.transformers_deviation, (transformers.__version__, prior['transformers_version'])
        deviations.append(dict(transformers_installed=transformers.__version__,
                               transformers_calibration=prior['transformers_version']))
    args.out.mkdir(parents=True, exist_ok=False)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    report = dict(status='running', job_id=os.environ.get('SLURM_JOB_ID'), config=config, deviations=deviations,
                  host=dict(hostname=socket.gethostname(), gpus=[torch.cuda.get_device_name(0)],
                            torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__),
                  source_sha256={p: digest_file(p) for p in SOURCES})
    save(args.out, report)

    model, modules = load_model(prior, False)
    model.set_attn_implementation('sdpa')
    for name, m in modules.items():
        assert sha(m.weight) == prior['matrices'][name]['source_sha256'], name
    device = model.get_input_embeddings().weight.device
    counts = dict(teacher_forward_sequences=0, student_forward_sequences=0, student_backward_sequences=0,
                  recompute_forward_sequences=0, development_forward_sequences=0, optimizer_steps=0)

    monitor.enter('data_load')
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    dev, report['development'] = load_development(development)
    if args.budget == 'pool':
        pool = torch.load(args.pool, map_location='cpu', weights_only=True)
        per = args.pool_per_source
        train = [r['ids'] for s in ('math', 'code') for r in [r for r in pool if r['source'] == s][:per]]
        assert len(train) == 2 * per, len(train)
        report['pool'] = dict(path=str(args.pool), sha256=digest_file(args.pool), per_source=per, sequences=len(train))
    else:
        fit, _ = math_code_data(tok, prior['fit'])
        train = [b for source in ('math', 'code') for b in fit[source]]
    generator = torch.Generator().manual_seed(args.seed)
    orders = []

    def indices(step):
        """Sequences of optimizer step `step`: consecutive slices of seeded per-epoch permutations."""
        epoch, offset = divmod(step * args.batch, len(train))
        while len(orders) <= epoch:
            orders.append(torch.randperm(len(train), generator=generator).tolist())
        return orders[epoch][offset:offset + args.batch]

    def teacher_logprobs(ids):
        logits = model(input_ids=ids.to(device), use_cache=False).logits
        out = logits[:, :-1].float().log_softmax(-1).bfloat16().cpu()
        counts['teacher_forward_sequences'] += ids.shape[0]
        return out

    monitor.enter('teacher_precompute')
    model.eval()
    probe = args.budget == 'probe'
    with torch.no_grad():
        dev_teacher = [] if probe else [teacher_logprobs(r['ids']) for r in dev]
        teacher = {}
        if args.budget in ('c1', 'c2'):
            for i in range(len(train)):
                teacher[i] = teacher_logprobs(train[i])
        elif probe and not args.live_teacher:
            for k in range(args.probe_steps):
                for i in indices(k):
                    teacher[i] = teacher_logprobs(train[i])
    teacher_bytes = sum(t.numel() * t.element_size() for t in list(teacher.values()) + dev_teacher)
    report['teacher_cpu_bytes'] = teacher_bytes

    monitor.enter('quantizer_verification')
    bad = verify_weights({n: m.weight for n, m in modules.items()})
    report['fused_quantizer_bitwise_on_source_weights'] = not bad
    assert not bad, bad
    pristine = {n: m.weight.detach().to('cpu', copy=True).pin_memory() for n, m in modules.items()}
    student = Student(args.arm, modules, lora_rank=args.lora_rank)
    if args.evaluate is not None:
        state = torch.load(args.evaluate / 'state.pt', map_location='cpu', weights_only=True)
        assert digest_file(args.evaluate / 'state.pt') == trained['state_sha256']
        with torch.no_grad():
            for name, m in modules.items():
                if args.arm == 'qat':
                    m.weight.copy_(state[name])
                elif args.arm == 'scale':
                    student.factors[name].copy_(state[name])
                else:
                    student.lora[name][0].copy_(state[name][0]); student.lora[name][1].copy_(state[name][1])
        del state
    dev_eval, final_eval = make_evaluator(model, modules, device, args.eval_batch)

    @torch.no_grad()
    def evaluate_deployed(fn):
        """Load the deployed weights, run fn(), restore the trainable/frozen weights."""
        backup = {n: m.weight.detach().to('cpu', copy=True) for n, m in modules.items()} if args.arm == 'qat' else pristine
        for name, m in modules.items():
            m.weight.copy_(student.deployed(name, m))
        model.eval()
        student.quantize = False
        result = fn()
        for name, m in modules.items():
            m.weight.copy_(backup[name])
        return result

    if not probe and args.evaluate is None:
        monitor.enter('initial_dev_eval')
        report['initial_dev'] = evaluate_deployed(lambda: dev_eval(dev, dev_teacher))
        counts['development_forward_sequences'] += len(dev)
        print(f'START dev CE {report["initial_dev"]["ce"]:.6f} KL {report["initial_dev"]["kl"]:.6f}', flush=True)
    report['setup_seconds'] = time.time() - started
    save(args.out, report)

    live = None
    if args.live_teacher:
        monitor.enter('live_teacher_load')
        live = AutoModelForCausalLM.from_pretrained(prior['source'], revision=prior['revision'], dtype=torch.bfloat16,
                                                    attn_implementation='sdpa', device_map='cuda')
        live.eval().requires_grad_(False)

    params = student.parameters()
    trainable = sum(p.numel() for p in params)
    optimizer = None
    report['memory_computed'] = dict(
        model_weights_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        trainable_parameters=trainable,
        gradient_bytes=sum(p.numel() * p.element_size() for p in params),
        teacher_cpu_bytes=teacher_bytes,
        pristine_weight_copy_cpu_bytes=sum(t.numel() * t.element_size() for t in pristine.values()))

    if args.budget == 'pool':
        steps_total = len(train) // args.batch
    elif args.budget == 'c1':
        steps_total = len(train) // args.batch
    elif args.budget == 'probe':
        steps_total = args.probe_steps
    elif args.budget == 'c2':
        assert args.time_budget, '--time-budget is required for c2'
        steps_total = None
    else:
        steps_total = 0
    if args.checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.micro_batch = args.micro_batch
    log, step = [], 0
    chunk_teacher = {}
    chunk_steps = args.teacher_chunk // args.batch
    assert args.budget != 'pool' or (args.teacher_chunk % args.batch == 0 and len(train) % args.teacher_chunk == 0)

    def swap_to_pristine(on):
        """qat: BF16 teacher passes need the source weights; the latent weights wait on the CPU."""
        if args.arm != 'qat':
            return
        for name, m in modules.items():
            if on:
                latent[name].copy_(m.weight.detach(), non_blocking=True)
                m.weight.data.copy_(pristine[name], non_blocking=True)
            else:
                m.weight.data.copy_(latent[name], non_blocking=True)
        torch.cuda.synchronize()
    latent = {n: torch.empty_like(t).pin_memory() for n, t in pristine.items()} \
        if (args.arm == 'qat' and args.budget == 'pool') else None

    oom = None
    monitor.enter('training')
    try:
        if steps_total != 0:
            optimizer = make_optimizer(args.optimizer, params, args.lr)
        while steps_total is None or step < steps_total:
            if args.budget == 'c2' and time.time() - started >= args.time_budget:
                report['stopped'] = 'time budget'
                break
            if args.budget == 'pool' and step % chunk_steps == 0:
                # Teacher log-probs for the next chunk of the schedule, precomputed and held on the CPU.
                monitor.enter('teacher_precompute_chunk')
                chunk_teacher.clear()
                model.eval(); student.quantize = False
                swap_to_pristine(True)
                with torch.no_grad():
                    for k in range(step, step + chunk_steps):
                        for i in indices(k):
                            chunk_teacher[i] = teacher_logprobs(train[i])
                swap_to_pristine(False)
                report['teacher_chunk_cpu_bytes'] = sum(t.numel() * t.element_size() for t in chunk_teacher.values())
                monitor.enter('training')
            t0 = time.time()
            batch = indices(step)
            model.train(); student.quantize = True
            losses = []
            for start in range(0, args.batch, args.micro_batch):
                idx = batch[start:start + args.micro_batch]
                ids = torch.cat([train[i] for i in idx]).to(device)
                if live is not None:
                    with torch.no_grad():
                        t = live(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
                        counts['teacher_forward_sequences'] += ids.shape[0]
                else:
                    source = chunk_teacher if args.budget == 'pool' else teacher
                    t = torch.cat([source[i] for i in idx]).to(device).float()
                with torch.enable_grad():
                    lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
                    kl = per_step_kl(lp, t)
                    (kl.sum() / args.batch).backward()
                losses.extend(kl.detach().tolist())
                counts['student_forward_sequences'] += len(idx)
                counts['student_backward_sequences'] += len(idx)
                if args.checkpointing:
                    counts['recompute_forward_sequences'] += len(idx)
                del ids, t, lp, kl
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            counts['optimizer_steps'] = step
            torch.cuda.synchronize()
            log.append(dict(step=step, kl=sum(losses) / len(losses), seconds=time.time() - t0,
                            elapsed=time.time() - started))
            if step == 1 or step % 16 == 0:
                print(f'STEP {step} kl {log[-1]["kl"]:.6f} {log[-1]["seconds"]:.2f}s elapsed {log[-1]["elapsed"]:.0f}s',
                      flush=True)
                report['log'] = log; report['counts'] = counts; save(args.out, report)
    except torch.OutOfMemoryError as e:
        oom = dict(step=step, message=str(e).split('\n')[0],
                   gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
                   gpu_peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
                   gpu_allocated_gib=torch.cuda.memory_allocated() / GIB)
        print('OOM ' + json.dumps(oom), flush=True)
    report['log'] = log
    report['counts'] = counts
    report['optimizer_state_bytes'] = state_bytes(optimizer)
    report['memory_computed']['optimizer_state_bytes'] = report['optimizer_state_bytes']
    report['training_seconds'] = time.time() - started - report['setup_seconds']
    if oom is not None:
        monitor.close(sync=False)
        report.update(status='oom', oom=oom, resources=monitor.summary())
        save(args.out, report)
        return
    if probe:
        monitor.close()
        report.update(status='complete', resources=monitor.summary(),
                      steady_step_seconds=sum(r['seconds'] for r in log[1:]) / max(1, len(log) - 1))
        save(args.out, report)
        return

    # The optimizer state is not needed for evaluation.
    del optimizer
    for p in params:
        p.grad = None
    if args.checkpointing:
        model.gradient_checkpointing_disable()
    torch.cuda.empty_cache()
    monitor.enter('final_dev_eval')
    with torch.no_grad():
        report['fused_quantizer_bitwise_on_final_weights'] = not student.verify_fused()
        assert report['fused_quantizer_bitwise_on_final_weights']
    final_dev = evaluate_deployed(lambda: dev_eval(dev, dev_teacher))
    counts['development_forward_sequences'] += len(dev)
    if args.evaluate is not None:
        report['final_dev'] = final_dev
        report['final_dev_matches_training_run'] = (final_dev['ce_nll'] == trained['final_dev']['ce_nll']
                                                    and final_dev['kl_values'] == trained['final_dev']['kl_values'])
    else:
        report['final_dev'] = final_dev
    print(f'END dev CE {final_dev["ce"]:.6f} KL {final_dev["kl"]:.6f}', flush=True)
    report['final_dev_seconds'] = time.time() - started - report['setup_seconds'] - report['training_seconds']
    if args.save_state:
        monitor.enter('save_state')
        with torch.no_grad():
            if args.arm == 'qat':
                state = {n: m.weight.detach().cpu() for n, m in modules.items()}
            elif args.arm == 'scale':
                state = {n: f.detach().cpu() for n, f in student.factors.items()}
            else:
                state = {n: (a.detach().cpu(), b.detach().cpu()) for n, (a, b) in student.lora.items()}
        torch.save(state, args.out / 'state.pt')
        report['state_sha256'] = digest_file(args.out / 'state.pt')
        del state
    if args.budget == 'zero' or args.evaluate is not None:
        monitor.enter('final_evaluation')
        batches, report['data'] = data(tok, prior, 2048)
        published = json.loads(Path(PUBLISHED['llama8b']).read_text())
        validate_evaluation_data(report['data'], published['data'])
        report['evaluation'] = evaluate_deployed(lambda: final_eval(batches))
    report['counts'] = counts
    monitor.close()
    report['resources'] = monitor.summary()
    report['total_seconds'] = time.time() - started
    report['status'] = 'complete'
    save(args.out, report)
    print('RESOURCES ' + json.dumps({k: v for k, v in report['resources'].items() if k not in ('phases',)}), flush=True)


if __name__ == '__main__':
    main()

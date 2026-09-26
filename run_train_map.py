"""MixFP4 tile selection by gradient training of per-tile logits on the KL loss (no backtracking).

Every type tile u of every text linear layer gets a real latent logit theta_u.
The weights are W(theta) = B + expand(m(theta)) * (A - B), with B the FourOverSix
E2M1 candidate and A the E0M3 alpha=1 candidate (identical to run_multiround.py).

  --param ste      m = 1[theta > 0] in the forward pass (the deployable hard map),
                   straight-through backward dm/dtheta = 1 (BinaryConnect-style
                   latent weights). The gradient dKL/dm_u = <G, P_u (A - B)> is the
                   same first-order tile score run_multiround.py ranks by.
  --param sigmoid  m = sigmoid(theta / tau), a soft interpolation with tau annealed
                   geometrically from --tau-start to --tau-end; the map is rounded
                   (theta > 0) for every development evaluation and at the end.

The loss is the token-mean KL to the BF16 teacher on the 128 math/code calibration
sequences, with the scoring forward convention of run_multiround.py (causal
per-token FourOverSix activations, straight-through). Weight gradients are never
materialized as parameters: a backward hook on each linear output forms
G = dy^T x for the batch and reduces G * (A - B) to tiles. The optimizer updates
the logits; there is no candidate filter and no acceptance test. The 192 held-out
development documents are evaluated (hard map, tensor-wide per-document
activations, as in evaluation) only to MONITOR training; they never change the
map. WikiText-2 / C4 are evaluated once, on the final map.

TM-OPT: the MR-OPT speed and memory optimizations of run_multiround.py, each behind a flag whose default is the
legacy behaviour above. --tm-opt turns all of them on except --tile-grad-kernel (the TM-OPT configuration); --no-<flag>
turns one off again. B1 stays opt-in: for the training step's single batch GEMM per module it is slower than the
legacy hook, and its FP32 error is at the noise level of 4,096-token sums (results/tm_opt/PROTOCOL.md, deviations 1-2).
  --memory-mode lean     one native-format candidate store (candidate_store.CandidateStore). Every quantized Linear
                         decodes its weight on each call, in the forward and again for the backward (saved-tensor
                         hooks), so no BF16 copy of those matrices stays resident. STE: the store's decode of the hard
                         map; sigmoid: b + m * (a - b) from the store's decoded candidates, with the legacy arithmetic
                         and dtypes.
  --fused-act-quant      training: fourover6_rows (bit-exact to quantize_rows) in the same straight-through expression;
                         monitor and final evaluation: fourover6 per document (bit-exact to quant_per_document). The
                         first 64 calls of each are checked bitwise.
  --tile-grad-kernel     B1 for the tile gradient (tile_score.tile_sums, from the packed store): the batch sum of
                         G * (A - B) per tile in FP32; only the summation order differs from the legacy hook. Lean only;
                         opt-in, not part of TM-OPT.
  --chunked-loss         the step's KL gradient into the logits two documents at a time (chunked_loss.train_kl_gradient),
                         and the monitor's per-document CE/KL the same way: bitwise the whole-batch expressions.
  --deterministic        torch.use_deterministic_algorithms(True) (CUBLAS_WORKSPACE_CONFIG must be set).
  --dev-backend native / --eval-backend native
                         the monitor development evaluations / the final WikiText-2 and C4 evaluation on the native
                         b8x64 kernel (native_dev.NativeDev, from the lean store), with --single-pass-epilogue. The
                         fake evaluation stays available.
  --tile-grad-tc         method B (TM-OPT+TC, results/tm_opt/PROTOCOL_TC.md): the tile-gradient GEMM G = dy^T x of the
                         legacy hook on the BF16 tensor cores with FP32 accumulation and FP32 output (tc_matmul). dy
                         and x are bf16, so every product is exact; only the accumulation differs, and nothing is
                         rounded to bf16. The model's GEMMs are unchanged. Opt-in, not part of TM-OPT.
  --profile JSON         profile one training epoch after the setup (torch.profiler; GPU time by phase and region), write
                         JSON and stop.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set by the caller; the report records it. Locally (no Slurm),
--data-root points at the data layout of run_multiround.data_paths.
"""
import argparse
import hashlib
import json
import math
import os
import random
import resource
import socket
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

import chunked_loss
import profile_regions
from cost_monitor import PhaseMonitor
from profile_regions import region
from quantize.causal_four_over_six import quantize_rows
from quantize.fast_act import check as check_act, quant_per_document
from quantize.fused_fourover6 import fourover6, fourover6_rows
from quantize.packed_candidates import decode_alt, decode_base, nbytes, pack
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_math_code_calibration import load_model, math_code_data
from run_multiround import MODELS, PUBLISHED, UNITS, data_paths, expand, load_development, reduce
from run_task_reorder_eval import validate_evaluation_data

# The TM-OPT configuration and the legacy one (every flag's default without --tm-opt). B1 (--tile-grad-kernel) is not
# part of TM-OPT: slower than the legacy hook here and at FP32 noise level (results/tm_opt/PROTOCOL.md, deviation 2).
TM_OPT = dict(memory_mode='lean', fused_act_quant=True, tile_grad_kernel=False, chunked_loss=True, deterministic=True,
              dev_backend='native', eval_backend='native', single_pass_epilogue=True)
LEGACY = dict(memory_mode='legacy', fused_act_quant=False, tile_grad_kernel=False, chunked_loss=False,
              deterministic=False, dev_backend='fake', eval_backend='fake', single_pass_epilogue=False)
SOURCES = ('run_train_map.py', 'run_multiround.py', 'quantize/quantizer.py', 'quantize/causal_four_over_six.py',
           'quantize/fused_fourover6.py', 'chunked_loss.py', 'repro_local/realquant/candidate_store.py',
           'repro_local/realquant/native_dev.py', 'repro_local/realquant/tile_score.py')


def tc_matmul(a, b):
    """TM-OPT+TC: a @ b of two bf16 tensors on the BF16 tensor cores with FP32 accumulation and FP32 output
    (cuBLAS through torch.mm(..., out_dtype=torch.float32)); nothing is rounded to bf16. Every product of two bf16
    values is exact in FP32; only the accumulation differs from the legacy FP32 SIMT GEMM over the float casts. No
    global flag is touched (allow_tf32 stays False). Chosen over the TF32 variant, which was slower and less accurate
    in the pre-registration check (results/tm_opt/PROTOCOL_TC.md)."""
    assert a.dtype == b.dtype == torch.bfloat16
    return torch.mm(a, b, out_dtype=torch.float32)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=MODELS, default='llama8b')
    ap.add_argument('--data-root', type=Path, default=None,
                    help='Local layout of the calibration and development data (run_multiround.data_paths); '
                         'without it the cluster paths are used and a Slurm job is required')
    ap.add_argument('--transformers-deviation', action='store_true',
                    help='Allow a transformers version other than the calibration record\'s (recorded as a deviation)')
    ap.add_argument('--unit', choices=tuple(UNITS), required=True)
    ap.add_argument('--param', choices=('ste', 'sigmoid'), default='ste')
    ap.add_argument('--optimizer', choices=('adam', 'sgd'), default='adam')
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--eps', type=float, default=1e-12,
                    help='Adam epsilon; tile gradients are ~1e-6, so the torch default 1e-8 would dominate')
    ap.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999))
    ap.add_argument('--schedule', choices=('constant', 'cosine'), default='constant')
    ap.add_argument('--init-logit', type=float, default=-1.0,
                    help='Initial theta of every tile (negative = start at FourOverSix E2M1)')
    ap.add_argument('--tau-start', type=float, default=1.0)
    ap.add_argument('--tau-end', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=8, help='Calibration sequences per forward/backward')
    ap.add_argument('--accum', type=int, default=1, help='Forward/backward passes per optimizer step')
    ap.add_argument('--eval-every', type=int, default=2, help='Epochs between development evaluations (monitor only)')
    ap.add_argument('--eval-batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpus', type=int, default=None, help='Qwen: 1 loads on one device, else balanced')
    ap.add_argument('--tm-opt', action='store_true',
                    help='TM-OPT: every optimization flag below on (except --tile-grad-kernel) unless turned off')
    ap.add_argument('--memory-mode', choices=('legacy', 'lean'), default=None)
    ap.add_argument('--fused-act-quant', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--tile-grad-kernel', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--chunked-loss', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--dev-backend', choices=('fake', 'native'), default=None)
    ap.add_argument('--eval-backend', choices=('fake', 'native'), default=None)
    ap.add_argument('--single-pass-epilogue', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--tile-grad-tc', action='store_true',
                    help='TM-OPT+TC: the tile-gradient GEMM on BF16 tensor cores with FP32 accumulation and output')
    ap.add_argument('--profile', type=Path, default=None, metavar='JSON',
                    help='Profile one training epoch after the setup, write the GPU-time breakdown to JSON, and stop')
    ap.add_argument('--record-theta-hashes', action='store_true',
                    help='Record the sha256 of every logit after every optimizer step (bitwise comparisons)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    for key, value in (TM_OPT if args.tm_opt else LEGACY).items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    settings = {key: getattr(args, key) for key in TM_OPT}
    configuration = 'TM-OPT' if settings == TM_OPT else 'legacy' if settings == LEGACY else 'custom'
    if args.tile_grad_tc:
        configuration = 'TM-OPT+TC' if configuration == 'TM-OPT' else configuration + '+TC'
    assert not (args.tile_grad_tc and args.tile_grad_kernel), 'B1 and the TC tile-gradient GEMM are alternatives'
    lean = args.memory_mode == 'lean'
    assert not args.tile_grad_kernel or lean, '--tile-grad-kernel reads the lean candidate store'
    assert (args.dev_backend == args.eval_backend == 'fake') or lean, 'the native evaluator reads the lean store'
    assert os.environ.get('SLURM_JOB_ID') or args.data_root is not None, 'Run through Slurm (or locally with --data-root)'
    if args.deterministic:
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8'), 'set CUBLAS_WORKSPACE_CONFIG=:4096:8'
        torch.use_deterministic_algorithms(True)
    monitor = PhaseMonitor()
    monitor.enter('model_load')
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, cols = UNITS[args.unit]
    qwen = args.model == 'qwen27b'
    if qwen:
        assert args.batch == 1 and args.eval_batch == 1, 'Qwen batched forward is not identical to batch 1'
    calibration, development = data_paths(args.model, args.data_root)
    prior = json.loads((calibration / 'report.json').read_text())
    deviations = []
    if transformers.__version__ != prior['transformers_version']:
        assert args.transformers_deviation, (transformers.__version__, prior['transformers_version'])
        deviations.append(dict(transformers_installed=transformers.__version__,
                               transformers_calibration=prior['transformers_version']))
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', job_id=os.environ.get('SLURM_JOB_ID'), configuration=configuration,
                  settings=dict(settings, tile_grad_tc=args.tile_grad_tc,
                                pytorch_cuda_alloc_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
                  host=dict(hostname=socket.gethostname(),
                            gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                            torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__),
                  data_root=None if args.data_root is None else str(args.data_root), deviations=deviations,
                  args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  source_sha256={p: digest_file(p) for p in SOURCES}, epochs=[])
    save(args.out, report)
    if qwen:
        from transformers import Qwen3_5ForConditionalGeneration
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            prior['source'], revision=prior['revision'], dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map='cuda' if args.gpus == 1 else 'balanced', output_loading_info=True)
        assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
        model.eval().requires_grad_(False)
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                   and 'language_model' in n and 'head' not in n}
        assert list(modules) == list(prior['matrices'])
    else:
        model, modules = load_model(prior, False)
        model.set_attn_implementation('sdpa')
    monitor.enter('data_load')
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    fit, _ = math_code_data(tok, prior['fit'])
    fit = [b for source in ('math', 'code') for b in fit[source]]
    dev, report['development'] = load_development(development)
    device = model.get_input_embeddings().weight.device
    monitor.enter('teacher_precompute')
    dev_teacher = []
    for r in dev:
        logits = model(input_ids=r['ids'].to(device), use_cache=False).logits
        dev_teacher.append(logits[:, :-1].float().log_softmax(-1).bfloat16().cpu())
    teacher = []
    for ids in fit:
        logits = model(input_ids=ids.to(device), use_cache=False).logits
        teacher.append(logits[:, :-1].float().log_softmax(-1).bfloat16().cpu())
    del logits
    # Candidates exactly as run_multiround.py: packed 4-bit codes + FP8 scales,
    # verified bitwise by pack(), dense BF16 fallback otherwise.
    monitor.enter('candidate_packing')
    store = native = None
    if lean:
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'repro_local' / 'realquant'))
        # ONE candidate store, in the native packed format; the training and fake paths decode from it
        from candidate_store import CandidateStore, lean_forward
        store = CandidateStore(rows, cols)
    if 'native' in (args.dev_backend, args.eval_backend):
        assert len(dev) % args.eval_batch == 0
        from native_dev import NativeDev
        native = NativeDev(modules, rows, cols, tokens=args.eval_batch * dev[0]['ids'].shape[1], store=store)
        native.single_pass_epilogue = args.single_pass_epilogue
    if args.tile_grad_kernel:
        from tile_score import tile_sums
    packed, dense, theta = {}, {}, {}
    for n, m in modules.items():
        assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
        b = quant_nvfp4_4over6(m.weight, 4, 16)
        a = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        p = pack(m.weight, b, a)
        if p is None:
            dense[n] = (b, a)
        elif not lean:
            packed[n] = p
        o, k = m.weight.shape
        assert k % cols == 0
        theta[n] = torch.full((-(-o // rows), k // cols), args.init_logit, dtype=torch.float32, device=m.weight.device)
        if lean:
            # both stored candidates must decode to exactly what decode_base / decode_alt return (checked by add);
            # then this matrix's BF16 weight is freed (the placeholder keeps its dtype)
            store.add(n, m.weight, *((b, a) if p is None else (decode_base(p), decode_alt(p))))
            m.weight = torch.nn.Parameter(torch.empty(0, dtype=m.weight.dtype, device=m.weight.device),
                                          requires_grad=False)
        else:
            m.weight.copy_(b)
        del a, b
    if lean:
        dense_fallback, dense = sorted(dense), {}      # the store verified these against (b, a) directly
    torch.cuda.empty_cache()
    if lean:
        report['candidate_storage'] = dict(store='native packed (lean)', modules=len(store.cand),
                                           dense_fallback_modules=dense_fallback, store_gib=store.nbytes() / 2 ** 30)
    else:
        report['candidate_storage'] = dict(packed_modules=len(packed), dense_fallback_modules=sorted(dense),
                                           packed_gib=sum(nbytes(p) for p in packed.values()) / 2 ** 30)
    report['tiles'] = sum(t.numel() for t in theta.values())
    print('CANDIDATES ' + json.dumps(report['candidate_storage']) + f' tiles={report["tiles"]}', flush=True)

    def base(n):
        if lean:
            return store.decode(n, which='base')
        return dense[n][0] if n in dense else decode_base(packed[n])

    def alt(n):
        if lean:
            return store.decode(n, which='alt')
        return dense[n][1] if n in dense else decode_alt(packed[n])

    tau = [args.tau_start]
    hard_mode = [False]         # lean: every forward decodes the hard map (evaluation) instead of the training weight

    def apply(n, hard):
        if lean:
            return              # lean forwards decode their weight on every call (lean_key below)
        b = base(n)
        if hard or args.param == 'ste':
            # Bitwise the multiround decode: each element is exactly B or A.
            modules[n].weight.copy_(torch.where(expand(theta[n] > 0, rows, cols, b.shape[0]), alt(n), b))
            return
        m = expand(torch.sigmoid(theta[n] / tau[0]), rows, cols, b.shape[0])
        modules[n].weight.copy_((b.float() + m * (alt(n).float() - b.float())).to(b.dtype))

    def hard_map():
        return {n: (t > 0) for n, t in theta.items()}

    if lean:
        def lean_weight(n):
            # the weight of module n for a key: ('map', hard map) -> the store's decode, bitwise apply(n, True);
            # ('soft', theta, tau) -> apply(n, False)'s arithmetic on the store's decoded candidates
            def weight_for(key):
                with region('lean weight decode'):
                    if key[0] == 'map':
                        return store.decode(n, key[1])
                    b = store.decode(n, which='base')
                    m = expand(torch.sigmoid(key[1] / key[2]), rows, cols, b.shape[0])
                    return (b.float() + m * (store.decode(n, which='alt').float() - b.float())).to(b.dtype)
            return weight_for

        def lean_key(n):
            if args.param == 'sigmoid' and not hard_mode[0]:
                return ('soft', theta[n], tau[0])
            return ('map', theta[n] > 0)

        for n, m in modules.items():
            m.forward = lean_forward(lean_weight(n), lambda n=n: lean_key(n), m.bias)
        # the store's Triton decode of a map equals its PyTorch select + decode bitwise: start map and a random one
        check = dict(start_map_mismatches=store.verify_map(hard_map()))
        mixer = torch.Generator(device=device).manual_seed(0)
        check['mixed_map_mismatches'] = store.verify_map(
            {n: torch.rand(t.shape, generator=mixer, device=t.device) < 0.5 for n, t in theta.items()})
        if native is not None:
            check['native_start_map_mismatches'] = native.verify_map(hard_map(), lambda n: store.decode(n, theta[n] > 0))
        assert not any(v for v in check.values()), check
        report['lean'] = dict(verification=check)
        print('LEAN ' + json.dumps(check), flush=True)

    eval_handles = []
    act_checks = [0]
    row_checks = [0]
    current_batch = [1]

    def per_document_act(module, inputs):
        # Tensor-wide activation scale per document, as in run_multiround.py's dev_eval.
        x = inputs[0]
        batch = current_batch[0]
        if batch == 1 or (x.dim() >= 3 and x.shape[0] == batch):
            view = x
        elif x.shape[0] % batch == 0 and x.shape[0] // batch > 1:
            view = x.reshape(batch, -1, x.shape[-1])
        else:
            raise RuntimeError(f'Cannot recover the document axis of a {tuple(x.shape)} input at batch {batch}')
        if args.fused_act_quant:
            q = fourover6(view, documents=view.shape[0] if view.dim() >= 3 else 1)
            if act_checks[0] < 64:
                assert torch.equal(q.view(torch.int16), quant_per_document(view).view(torch.int16)) and check_act(view), \
                    'fused per-document activation quantizer differs from quant_per_document / quant_nvfp4_4over6'
                act_checks[0] += 1
            return (q.reshape(x.shape), *inputs[1:])
        if act_checks[0] < 64:
            assert check_act(view), 'vectorized activation quantizer differs from quant_nvfp4_4over6'
            act_checks[0] += 1
        return (quant_per_document(view).reshape(x.shape), *inputs[1:])

    def per_sequence_kl(lp, t):
        return (t.exp() * (t - lp)).sum(-1).mean(-1)

    def eval_hooks(on):
        nonlocal eval_handles
        for h in eval_handles:
            h.remove()
        eval_handles = [m.register_forward_pre_hook(per_document_act) for m in modules.values()] if on else []

    def dev_batches(ce, kl):
        """CE and KL per development document of the model as it stands (hooks or native forwards in place)."""
        for start in range(0, len(dev), args.eval_batch):
            chunk = dev[start:start + args.eval_batch]
            ids = torch.cat([r['ids'] for r in chunk]).to(device)
            current_batch[0] = ids.shape[0]
            if native is not None and args.dev_backend == 'native':
                native.documents = ids.shape[0]
            if args.chunked_loss:
                logits = model(input_ids=ids, use_cache=False).logits
                c, k = chunked_loss.per_sequence_losses(logits, ids, dev_teacher[start:start + args.eval_batch], logits.device)
                ce.extend(c.tolist()); kl.extend(k.tolist())
                del logits
                continue
            lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            t = torch.cat(dev_teacher[start:start + args.eval_batch]).to(lp.device).float()
            ce.extend(F.nll_loss(lp.transpose(1, 2), ids[:, 1:].to(lp.device), reduction='none').mean(-1).tolist())
            kl.extend(per_sequence_kl(lp, t).tolist())
            del lp, t
        current_batch[0] = 1

    def dev_eval():
        """Development KL/CE of the HARD map (monitor only); leaves the training weights in place."""
        ce, kl = [], []
        if args.dev_backend == 'native':
            native.install(hard_map())
            dev_batches(ce, kl)
            native.remove()
            return dict(ce=sum(ce) / len(ce), kl=sum(kl) / len(kl), ce_nll=ce, kl_values=kl)
        if args.param == 'sigmoid':
            hard_mode[0] = True
            for n in modules:
                apply(n, True)
        eval_hooks(True)
        dev_batches(ce, kl)
        eval_hooks(False)
        if args.param == 'sigmoid':
            hard_mode[0] = False
            for n in modules:
                apply(n, False)
        return dict(ce=sum(ce) / len(ce), kl=sum(kl) / len(kl), ce_nll=ce, kl_values=kl)

    # Training hooks: straight-through per-token activations (the scoring convention
    # of run_multiround.py) and a backward hook that turns each linear layer's output
    # gradient into dLoss/dm for every tile of that layer.
    grads = {n: torch.zeros_like(t) for n, t in theta.items()}

    def act(module, inputs):
        x = inputs[0]
        with region('act_quant_rows (training)'):
            if args.fused_act_quant:
                q = fourover6_rows(x.detach())
                if row_checks[0] < 64:
                    assert torch.equal(q.view(torch.int16), quantize_rows(x.detach()).view(torch.int16)), \
                        'fused per-token activation quantizer differs from quantize_rows'
                    row_checks[0] += 1
            else:
                q = quantize_rows(x.detach())
        return (q + (x - x.detach()), *inputs[1:])


    def make_hook(n):
        def forward(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])

            def backward(dy):
                dy = dy.detach().reshape(-1, dy.shape[-1])
                if args.tile_grad_kernel:
                    # B1: the batch's tile sums of G * (A - B), straight from the packed store
                    with region('hook: B1 tile sums'):
                        tile_sums(dy, x, store.cand[n], rows, cols, store._luts(dy.device), grads[n])
                    return
                with region('hook: candidate decode + D'):
                    b = base(n)
                    d = alt(n).float() - b.float()
                    del b
                # dLoss/dm_u = sum over tile u of G * (A - B), with G = dy^T x (the same operations as the one-line
                # grads[n] += reduce((dy.float().T @ x.float()) * d, rows, cols), split for the profiler)
                with region('hook: G = dy^T x (tile-gradient GEMM)'):
                    g = tc_matmul(dy.T, x) if args.tile_grad_tc else dy.float().T @ x.float()
                with region('hook: G*(A-B), tile reduction, accumulate'):
                    gd = g * d
                    del g
                    grads[n] += reduce(gd, rows, cols)
                    del gd
            output.register_hook(backward)
        return forward

    def theta_digest():
        h = hashlib.sha256()
        for t in theta.values():
            h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    params = list(theta.values())
    if args.optimizer == 'adam':
        opt = torch.optim.Adam(params, lr=args.lr, betas=tuple(args.betas), eps=args.eps)
    else:
        opt = torch.optim.SGD(params, lr=args.lr)
    steps_per_epoch = math.ceil(math.ceil(len(fit) / args.batch) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    rng = random.Random(args.seed)

    monitor.enter('initial_dev_eval')
    initial = dev_eval()
    report['initial_dev'] = initial
    report['setup_seconds'] = time.time() - started
    if args.record_theta_hashes:
        report['theta_sha256'] = [theta_digest()]
    save(args.out, report)
    print(f'START dev CE {initial["ce"]:.6f} KL {initial["kl"]:.6f} tiles={report["tiles"]} '
          f'steps/epoch={steps_per_epoch} total={total_steps}', flush=True)
    if args.param == 'sigmoid':
        for n in modules:
            apply(n, False)
    step = 0
    previous = hard_map()
    profiler = None
    if args.profile is not None:
        # one training epoch under torch.profiler; every GPU kernel is attributed to its phase and innermost region
        profile_regions.ON[0] = True
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        profiler.__enter__()
    for epoch in range(args.epochs):
        monitor.enter('training')
        t0 = time.time()
        order = list(range(len(fit)))
        rng.shuffle(order)
        batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        losses, flips_epoch, hash_seconds = [], 0, 0.0
        handles = [m.register_forward_pre_hook(act) for m in modules.values()]
        handles += [m.register_forward_hook(make_hook(n)) for n, m in modules.items()]
        for g0 in range(0, len(batches), args.accum):
            group = batches[g0:g0 + args.accum]
            n_seq = sum(len(b) for b in group)
            for g in grads.values():
                g.zero_()
            with torch.enable_grad():
                for idx in group:
                    ids = torch.cat([fit[i] for i in idx]).to(device)
                    embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
                    if args.chunked_loss:
                        # the loss's gradient into the logits two documents at a time, then the model's backward
                        with region('phase: forward'):
                            logits = model(inputs_embeds=embeds, use_cache=False).logits
                        with region('phase: loss'):
                            grad, kl = chunked_loss.train_kl_gradient(logits, [teacher[i] for i in idx], logits.device, n_seq)
                        with region('phase: backward'):
                            logits.backward(grad)
                        losses.extend(kl.tolist())
                        del embeds, logits, grad, kl
                        continue
                    with region('phase: forward'):
                        lp = model(inputs_embeds=embeds, use_cache=False).logits[:, :-1].float().log_softmax(-1)
                    with region('phase: loss'):
                        t = torch.cat([teacher[i] for i in idx]).to(lp.device).float()
                        kl = per_sequence_kl(lp, t)
                    # Mean KL over the sequences of one optimizer step.
                    with region('phase: backward'):
                        (kl.sum() / n_seq).backward()
                    losses.extend(kl.tolist())
                    del embeds, lp, t, kl
            optimizer_phase = region('phase: optimizer')
            optimizer_phase.__enter__()
            if args.schedule == 'cosine':
                for pg in opt.param_groups:
                    pg['lr'] = args.lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            for n, p in theta.items():
                g = grads[n]
                if args.param == 'sigmoid':
                    s = torch.sigmoid(p / tau[0])
                    g = g * s * (1 - s) / tau[0]
                p.grad = g.clone()
            opt.step()
            step += 1
            if args.record_theta_hashes:
                t_hash = time.time()
                report['theta_sha256'].append(theta_digest())
                hash_seconds += time.time() - t_hash
            if args.param == 'sigmoid':
                tau[0] = args.tau_start * (args.tau_end / args.tau_start) ** (step / total_steps)
                for n in modules:
                    apply(n, False)
            for n in modules:
                now = theta[n] > 0
                changed = int((now != previous[n]).sum())
                if changed:
                    flips_epoch += changed
                    previous[n] = now
                    if args.param == 'ste':
                        apply(n, True)
            optimizer_phase.__exit__(None, None, None)
        for h in handles:
            h.remove()
        e0m3 = sum(int((t > 0).sum()) for t in theta.values())
        entry = dict(epoch=epoch, step=step, train_kl=sum(losses) / len(losses), e0m3_units=e0m3,
                     hard_flips=flips_epoch, tau=tau[0], lr=opt.param_groups[0]['lr'], epoch_seconds=time.time() - t0)
        if args.record_theta_hashes:
            entry['theta_hash_seconds'] = hash_seconds      # included in epoch_seconds
        if args.param == 'sigmoid':
            soft = torch.cat([torch.sigmoid(t / tau[0]).reshape(-1) for t in theta.values()])
            entry['undecided_fraction'] = float(((soft > 0.05) & (soft < 0.95)).float().mean())
        if profiler is not None:
            torch.cuda.synchronize()
            profiler.__exit__(None, None, None)
            profile_regions.ON[0] = False
            phases = ['phase: forward', 'phase: loss', 'phase: backward', 'phase: optimizer']
            regions = ['lean weight decode', 'act_quant_rows (training)', 'hook: candidate decode + D',
                       'hook: G = dy^T x (tile-gradient GEMM)', 'hook: G*(A-B), tile reduction, accumulate',
                       'hook: B1 tile sums']
            trace = args.profile.with_suffix('.trace.json')
            profiler.export_chrome_trace(str(trace))
            result, counted = profile_regions.breakdown_trace(trace, phases, regions)
            result['kernels_counted'] = counted
            result['epoch_wall_seconds_with_profiler'] = entry['epoch_seconds']
            result['trace'] = str(trace)
            args.profile.write_text(json.dumps(dict(model=args.model, unit=args.unit, configuration=configuration,
                                                    settings=report['settings'], steps=step, breakdown=result), indent=1) + '\n')
            print('PROFILE ' + json.dumps({ph: dict(gpu_s=result[ph]['gpu_ms_total'] / 1e3, wall_s=result[ph]['wall_ms'] / 1e3)
                                           for ph in phases}), flush=True)
            report['epochs'].append(entry)
            report['status'] = 'profile_complete'
            save(args.out, report)
            return
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            monitor.enter('dev_evaluation')
            t1 = time.time()
            d = dev_eval()
            entry['dev_seconds'] = time.time() - t1
            entry['dev_ce'], entry['dev_kl'] = d['ce'], d['kl']
            entry['dev_kl_values'] = d['kl_values']
            entry['dev_ce_values'] = d['ce_nll']
            torch.save({n: m.cpu() for n, m in hard_map().items()}, args.out / f'map_epoch{epoch + 1:03d}.pt')
        report['epochs'].append(entry)
        save(args.out, report)
        print('EPOCH ' + json.dumps({k: v for k, v in entry.items() if k not in ('dev_kl_values', 'dev_ce_values')}),
              flush=True)
    monitor.enter('final_dev_eval')
    final_map = hard_map()
    torch.save({n: m.cpu() for n, m in final_map.items()}, args.out / 'map.pt')
    torch.save({n: t.cpu() for n, t in theta.items()}, args.out / 'theta.pt')
    report['final_dev'] = dev_eval()
    report['final_e0m3_units'] = sum(int(m.sum()) for m in final_map.values())
    report['map_sha256'] = digest_file(args.out / 'map.pt')
    report['theta_file_sha256'] = digest_file(args.out / 'theta.pt')
    report['optimization_seconds'] = time.time() - started - report['setup_seconds']
    report['training_seconds'] = sum(e['epoch_seconds'] for e in report['epochs'])
    report['monitor_seconds'] = sum(e.get('dev_seconds', 0.0) for e in report['epochs'])
    save(args.out, report)
    print(f'FINAL dev CE {report["final_dev"]["ce"]:.6f} KL {report["final_dev"]["kl"]:.6f} '
          f'e0m3={report["final_e0m3_units"]}', flush=True)
    # Released PPL windows on the final HARD map.
    monitor.enter('final_evaluation')
    hard_mode[0] = True
    for n in modules:
        apply(n, True)
    batches, report['data'] = data(tok, prior, 2048)
    if args.model in PUBLISHED:
        published = json.loads(Path(PUBLISHED[args.model]).read_text())
        validate_evaluation_data(report['data'], published['data'])
        report['data_validation'] = PUBLISHED[args.model]
    else:
        report['data_validation'] = 'no published record for this model; windows recorded by token hash'
    report['evaluation_backend'] = args.eval_backend
    if args.eval_backend == 'native':
        native.install(final_map)
        native.documents = 1
        native.checked = 0          # the first 64 activation calls are checked bitwise
    else:
        eval_hooks(True)
    report['evaluation'] = {}
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
        report['evaluation'][key] = dict(nll=values, ppl=float(torch.exp(losses.sum() / (len(values) * 2048))))
        print(f'PPL train_{args.param}_{args.unit} {key} {report["evaluation"][key]["ppl"]:.6f}', flush=True)
        save(args.out, report)
    if args.eval_backend == 'native':
        report['native_activation_checks'] = native.checked
        native.remove()
    else:
        eval_hooks(False)
    monitor.close()
    phases = monitor.summary()
    report['resources'] = dict(
        total_seconds=time.time() - started,
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        gpu_peak_allocated_gib=phases['gpu_peak_allocated_gib'], gpu_peak_reserved_gib=phases['gpu_peak_reserved_gib'],
        cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, phases=phases)
    report['status'] = 'complete'
    save(args.out, report)


if __name__ == '__main__':
    main()

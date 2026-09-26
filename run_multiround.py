"""Multi-round relinearized MixFP4 election with backtracking on measured loss (Llama-3.1-8B).

Each round re-scores every unit's legal flip (E2M1 FourOverSix <-> E0M3 alpha=1,
undo included) with per-sequence CE and KL(BF16 teacher) weight gradients at the
CURRENT quantized model on the 128 calibration sequences, exactly the scoring
convention of run_math_code_calibration.py (causal per-token activation factors,
straight-through). Candidates are units whose CE and KL upper bounds
mean + FILTER_K * SE are both negative, ranked by that bound. The step is chosen
by backtracking: the top n, n/2, n/4, ... candidates are applied and the first
set that lowers mean CE on the 192 held-out development documents (tensor-wide
W4A4, as in evaluation) is accepted. The loop stops when no step lowers it.
WikiText-2 / C4 are evaluated once, on the final map only.
"""
import argparse
import json
import math
import os
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
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
from quantize.quantizer import quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import save, sha
from run_math_code_calibration import load_model, math_code_data
from run_task_reorder_eval import validate_evaluation_data

CALIBRATIONS = {'llama8b': Path('/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration'),
                'qwen27b': Path('/work/u4320956/task_reorder/pilot_20260919/qwen27b/calibration')}
# Held-out math/code documents from recorded earlier confirmation gates, tokenized per model.
DEVELOPMENT = {
    'llama8b': [Path('/work/u4320956/task_reorder/transfer_20260920/llama8b') / s
                for s in ('confirmation', 'gate_up_confirm', 'tile_refine_confirm')],
    'qwen27b': [Path('/work/u4320956/task_reorder/pilot_20260919/qwen27b/fine_rows_v2') / s
                for s in ('fisher_subset_validate_v2_confirm', 'preserved_row_confirm', 'ce_target_combinations_confirm')]}
PUBLISHED = {'llama8b': 'results/kse_paper/job_336566/llama8b/report.json',
             'qwen27b': 'results/kse_paper/job_336969/qwen27b/report.json'}
# Models prepared only locally (prepare_model_data.py): three development draws with Llama's rule.
LOCAL_DEVELOPMENT = {m: ('fresh_dev1', 'fresh_dev2', 'fresh_dev3') for m in ('qwen4b', 'mistral7b', 'phi4')}
MODELS = tuple(CALIBRATIONS) + tuple(LOCAL_DEVELOPMENT)


def data_paths(model, data_root=None):
    """Calibration report directory and development directories: the cluster paths, or the
    same layout under a local data root (see prepare_multiround_data.py, prepare_model_data.py)."""
    if data_root is None:
        return CALIBRATIONS[model], DEVELOPMENT[model]
    root = Path(data_root) / model
    names = [d.name for d in DEVELOPMENT[model]] if model in DEVELOPMENT else LOCAL_DEVELOPMENT[model]
    return root / 'calibration', [root / name for name in names]


def load_development(directories):
    records, provenance = [], []
    for directory in directories:
        report = json.loads((directory / 'report.json').read_text())
        assert report.get('status') == 'complete', directory
        assert digest_file(directory / 'fresh.pt') == report['fresh_sha256'], directory
        rows = torch.load(directory / 'fresh.pt', map_location='cpu', weights_only=True)
        records.extend(rows)
        provenance.append(dict(name=str(directory), documents=len(rows), sha256=report['fresh_sha256']))
    return records, provenance


UNITS = {'256x64': (256, 64), '16x64': (16, 64), '8x64': (8, 64), '1x16': (1, 16)}


def expand(mask, rows, cols, height=None):
    # The last row tile may be partial; its padded rows are trimmed (raw256 convention).
    return mask.repeat_interleave(rows, 0).repeat_interleave(cols, 1)[:height]


def reduce(x, rows, cols):
    o, k = x.shape
    x = F.pad(x, (0, 0, 0, (-o) % rows))
    return x.reshape(-1, rows, k // cols, cols).sum((1, 3))


@torch.no_grad()
def main():
    monitor = PhaseMonitor()
    monitor.enter('model_load')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=MODELS, default='llama8b')
    ap.add_argument('--unit', choices=tuple(UNITS), required=True)
    ap.add_argument('--objective', choices=('both', 'ce', 'kl'), required=True,
                    help='Scores that rank candidates AND the development loss(es) a step must lower')
    ap.add_argument('--filter-k', type=float, default=2.0)
    ap.add_argument('--max-rounds', type=int, default=40)
    ap.add_argument('--max-tries', type=int, default=22)
    ap.add_argument('--budget-hours', type=float, default=3.0)
    ap.add_argument('--significant-steps', action='store_true',
                    help='Accept a step only if the paired development change is significant: mean + 2 SE < 0 '
                         'over documents for every accepted objective (same 2 SE convention as the candidate filter)')
    ap.add_argument('--warm-start', action='store_true',
                    help='Start each round\'s backtracking at twice the previous accepted step instead of all candidates')
    ap.add_argument('--eval-batch', type=int, default=1,
                    help='Development documents per forward pass (activation scales stay per document)')
    ap.add_argument('--score-batch', type=int, default=1,
                    help='Calibration sequences per scoring forward/backward (gradients stay per sequence)')
    ap.add_argument('--check-start', action='store_true',
                    help='Stop after the initial development evaluation (batching equivalence check)')
    ap.add_argument('--gpus', type=int, default=None, help='Qwen: 1 loads on one device, else balanced')
    ap.add_argument('--data-root', type=Path, default=None,
                    help='Local copies of the calibration report and development directories '
                         '(<root>/<model>/calibration, <root>/<model>/<development name>); default: cluster paths')
    ap.add_argument('--transformers-deviation', action='store_true',
                    help='Run although the installed transformers differs from the calibration\'s; both are recorded')
    ap.add_argument('--skip-ce-backward', action='store_true',
                    help='--objective kl only: skip the CE backward in scoring (KL scores do not depend on it)')
    ap.add_argument('--shadow-native', action='store_true',
                    help='Also run every development evaluation on the native mixfp4 kernel (b8x64 build, '
                         'repro_local/realquant/native_dev.py). Logged only: the fake evaluation decides')
    ap.add_argument('--init-map', type=Path, default=None,
                    help='Start from this map (map.pt of a run with the same unit) instead of all E2M1')
    ap.add_argument('--dev-backend', choices=('fake', 'native'), default='fake',
                    help='Evaluator whose development KL makes every backtracking decision '
                         '(native: mixfp4 b8x64 kernel; the fake start value is still logged)')
    ap.add_argument('--eval-backend', choices=('fake', 'native'), default='fake',
                    help='Backend of the final WikiText-2 / C4 evaluation')
    ap.add_argument('--evaluate-map', action='append', default=[], metavar='LABEL=PATH',
                    help='Evaluate saved maps on WikiText-2 / C4 without calibrating (PATH may also be '
                         'fourover6 or bf16); repeatable')
    ap.add_argument('--deterministic', action='store_true',
                    help='torch.use_deterministic_algorithms(True); needs CUBLAS_WORKSPACE_CONFIG=:4096:8')
    ap.add_argument('--memory-mode', choices=('legacy', 'lean'), default='legacy',
                    help='lean: both candidates only in the native packed format (one store, the fake path '
                         'decodes from it), native weights built per evaluation and freed after it, and no '
                         'resident BF16 weight for the quantized matrices (each is decoded on use and '
                         'recomputed for the backward). The values are bitwise those of legacy')
    ap.add_argument('--record-dev-values', action='store_true',
                    help='Save every development evaluation\'s per-document CE and KL (dev_values.pt)')
    ap.add_argument('--dump-round0-scores', action='store_true',
                    help='Save round 0\'s per-unit CE/KL score mean and SE (round0_scores.pt)')
    ap.add_argument('--stop-after-scoring', action='store_true',
                    help='Stop after round 0\'s scoring pass (batching equivalence check, with --dump-round0-scores)')
    ap.add_argument('--fused-act-quant', action='store_true',
                    help='Fused (Triton) FourOverSix activation quantizers: per-token rows for scoring (quantize_rows) '
                         'and per-document for the fake evaluator (quant_per_document); bitwise identical, the first '
                         '64 calls of each are checked against the reference')
    ap.add_argument('--chunked-loss', action='store_true',
                    help='CE/KL (development evaluation) and the scoring KL backward over chunks of >= 2 documents '
                         '(chunked_loss.py): no FP32 [batch, T, vocab] tensor for the whole batch; bitwise identical')
    ap.add_argument('--tile-score-kernel', action='store_true',
                    help='B1: fused per-sequence tile scores in the scoring backward (repro_local/realquant/tile_score.py; '
                         'lean mode); FP32 summation order differs from the legacy hook, nothing else')
    ap.add_argument('--single-pass-epilogue', action='store_true',
                    help='Native evaluator: write bf16(D * gs) in one pass (torch.mul into a bf16 output); bitwise identical')
    ap.add_argument('--profile', type=Path, default=None, metavar='JSON',
                    help='Profile one fake development evaluation, one native one (native runs) and one scoring '
                         'pass with torch.profiler, write the GPU-time breakdown by region to JSON, and stop')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, cols = UNITS[args.unit]
    qwen = args.model == 'qwen27b'
    lean = args.memory_mode == 'lean'
    assert not args.tile_score_kernel or lean, '--tile-score-kernel reads the lean candidate store'
    assert not (args.chunked_loss and not args.skip_ce_backward and not args.evaluate_map), \
        '--chunked-loss scores the KL objective only (--skip-ce-backward)'
    assert not args.skip_ce_backward or args.objective == 'kl', '--skip-ce-backward needs --objective kl'
    if args.deterministic:
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8'), 'set CUBLAS_WORKSPACE_CONFIG=:4096:8'
        torch.use_deterministic_algorithms(True)
    assert not (args.shadow_native and args.dev_backend == 'native'), 'the shadow logs native next to fake decisions'
    maps_to_evaluate = dict(spec.partition('=')[::2] for spec in args.evaluate_map)
    evaluate_only = bool(maps_to_evaluate)
    assert args.eval_backend == 'fake' or 'bf16' not in maps_to_evaluate.values(), 'bf16 is evaluated with fake only'
    assert 'nvfp4' not in maps_to_evaluate.values() or lean, 'the NVFP4 map is evaluated in lean mode'
    # BF16 alone needs no candidates: the model is evaluated as loaded (weights verified by hash)
    bf16_only = evaluate_only and set(maps_to_evaluate.values()) == {'bf16'}
    calibration, development = data_paths(args.model, args.data_root)
    prior = json.loads((calibration / 'report.json').read_text())
    deviations = []
    if transformers.__version__ != prior['transformers_version']:
        assert args.transformers_deviation, (transformers.__version__, prior['transformers_version'])
        deviations.append(dict(transformers_installed=transformers.__version__,
                               transformers_calibration=prior['transformers_version']))
    args.out.mkdir(parents=True, exist_ok=False)
    extra = {}
    if args.dev_backend != 'fake' or args.eval_backend != 'fake' or evaluate_only:
        extra['backends'] = dict(dev=args.dev_backend, eval=args.eval_backend, evaluate_only=evaluate_only)
    if args.deterministic:
        extra['deterministic'] = dict(use_deterministic_algorithms=True,
                                      cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    report = dict(**extra, status='running', job_id=os.environ.get('SLURM_JOB_ID'),
                  host=dict(hostname=socket.gethostname(),
                            gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                            torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__),
                  data_root=None if args.data_root is None else str(args.data_root), deviations=deviations,
                  skip_ce_backward=args.skip_ce_backward, memory_mode=args.memory_mode,
                  speedups=dict(fused_act_quant=args.fused_act_quant, single_pass_epilogue=args.single_pass_epilogue,
                                tile_score_kernel=args.tile_score_kernel, chunked_loss=args.chunked_loss,
                                pytorch_cuda_alloc_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF')),
                  model=args.model, unit=args.unit, objective=args.objective,
                  eval_batch=args.eval_batch, score_batch=args.score_batch,
                  significant_steps=args.significant_steps, warm_start=args.warm_start,
                  filter_k=args.filter_k, max_tries=args.max_tries,
                  acceptance={'both': 'mean dev CE and mean dev KL must both decrease', 'ce': 'mean dev CE must decrease',
                              'kl': 'mean dev KL must decrease'}[args.objective],
                  source_sha256={p: digest_file(p) for p in ('run_multiround.py', 'quantize/quantizer.py',
                                                             'quantize/causal_four_over_six.py',
                                                             'repro_local/realquant/native_dev.py',
                                                             'repro_local/realquant/candidate_store.py',
                                                             'quantize/fused_fourover6.py',
                                                             'repro_local/realquant/tile_score.py',
                                                             'chunked_loss.py')},
                  rounds=[])
    save(args.out, report)

    def record_out_of_memory(kind, error, trace):
        # On a CUDA out-of-memory exit, keep the per-phase memory peaks and the allocator summary in the report.
        if issubclass(kind, torch.OutOfMemoryError):
            try:
                monitor.close()
                report['out_of_memory'] = dict(message=str(error).split('\n')[0], phases=monitor.summary(),
                                               allocator_summary=torch.cuda.memory_summary(abbreviated=True))
                report['status'] = 'out_of_memory'
                save(args.out, report)
            except Exception:
                pass
        sys.__excepthook__(kind, error, trace)
    sys.excepthook = record_out_of_memory
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
    if evaluate_only:
        fit, dev, report['development'] = [], [], None
    else:
        fit, _ = math_code_data(tok, prior['fit'])
        fit = [b for source in ('math', 'code') for b in fit[source]]
        dev, report['development'] = load_development(development)
    device = model.get_input_embeddings().weight.device
    monitor.enter('teacher_precompute')
    dev_teacher = []
    for r in dev:
        logits = model(input_ids=r['ids'].to(device), use_cache=False).logits
        dev_teacher.append(logits[:, :-1].float().log_softmax(-1).bfloat16().cpu())
        del logits
    teacher = []
    for ids in fit:
        logits = model(input_ids=ids.to(device), use_cache=False).logits
        teacher.append(logits[:, :-1].float().log_softmax(-1).bfloat16().cpu())
    # Both candidates are stored in deployment format (4-bit codes + FP8 scales) and
    # decoded per module on use; pack() verifies bitwise equality with the reference
    # quantizers, and any module that fails keeps dequantized BF16 copies instead.
    monitor.enter('candidate_packing')
    packed, dense, sel = {}, {}, {}
    native = store = None
    if lean or args.shadow_native or args.dev_backend == 'native' or args.eval_backend == 'native':
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'repro_local' / 'realquant'))
    if lean:
        # ONE candidate store, in the native packed format; the fake path decodes from it
        from candidate_store import CandidateStore
        store = CandidateStore(rows, cols)
    if args.shadow_native or args.dev_backend == 'native' or args.eval_backend == 'native':
        assert len(dev) % args.eval_batch == 0
        from native_dev import NativeDev
        native = NativeDev(modules, rows, cols, tokens=args.eval_batch * dev[0]['ids'].shape[1] if dev else 2048,
                           store=store)
        native.single_pass_epilogue = args.single_pass_epilogue
    pristine = {}
    for n, m in modules.items():
        assert sha(m.weight) == prior['matrices'][n]['source_sha256'], n
        if bf16_only:
            continue
        if lean and 'nvfp4' in maps_to_evaluate.values():
            store.add_nvfp4(n, m.weight, quant_nvfp4(m.weight, 4, 16))
        b = quant_nvfp4_4over6(m.weight, 4, 16)
        a = quant_mix_4_6(m.weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        p = pack(m.weight, b, a)
        if p is None:
            dense[n] = (b, a)
        elif not lean:
            packed[n] = p
        # both native candidates must decode to exactly what decode_base / decode_alt return
        if lean:
            store.add(n, m.weight, *((b, a) if p is None else (decode_base(p), decode_alt(p))))
        elif native is not None:
            native.add(n, m.weight, *((b, a) if p is None else (decode_base(p), decode_alt(p))))
        o, k = m.weight.shape
        assert k % cols == 0
        sel[n] = torch.zeros(-(-o // rows), k // cols, dtype=torch.bool, device=m.weight.device)
        if 'bf16' in maps_to_evaluate.values():
            pristine[n] = m.weight.detach().to('cpu', copy=True)
        if lean:
            # the store holds the candidates: free this matrix's BF16 weight (the placeholder keeps its dtype)
            m.weight = torch.nn.Parameter(torch.empty(0, dtype=m.weight.dtype, device=m.weight.device),
                                          requires_grad=False)
        else:
            m.weight.copy_(b)
        del a, b, p
    if lean:
        dense_fallback, dense = sorted(dense), {}      # the store verified these against (b, a) directly
    torch.cuda.empty_cache()
    if lean:
        report['candidate_storage'] = dict(store='native packed (lean)', modules=len(store.cand),
                                           dense_fallback_modules=dense_fallback,
                                           store_gib=store.nbytes() / 2 ** 30)
    else:
        report['candidate_storage'] = dict(packed_modules=len(packed), dense_fallback_modules=sorted(dense),
                                           packed_gib=sum(nbytes(p) for p in packed.values()) / 2 ** 30)
    print('CANDIDATES ' + json.dumps(report['candidate_storage']), flush=True)

    def base(n):
        if lean:
            return store.decode(n, which='base')
        return dense[n][0] if n in dense else decode_base(packed[n])

    def alt(n):
        if lean:
            return store.decode(n, which='alt')
        return dense[n][1] if n in dense else decode_alt(packed[n])

    def apply(n):
        if lean:
            return                          # lean forwards decode sel's weight on use
        b = base(n)
        modules[n].weight.copy_(torch.where(expand(sel[n], rows, cols, b.shape[0]), alt(n), b))

    weight_source = ['map']                 # lean: 'map' (the store under sel), 'nvfp4' or 'bf16' (pristine)

    def lean_weight(n):
        # the weight of module n for a key (source, map): the store decoded under the map, the store's
        # NVFP4 candidate, or pristine
        def weight_for(key):
            if key[0] == 'bf16':
                return pristine[n].to(device)
            with region('lean weight decode'):
                if key[0] == 'nvfp4':
                    return store.decode(n, which='nvfp4')
                return store.decode(n, key[1])
        return weight_for

    if args.tile_score_kernel:
        from tile_score import tile_scores
    if lean and not bf16_only:
        from candidate_store import lean_forward
        for n, m in modules.items():
            m.forward = lean_forward(lean_weight(n), lambda n=n: (weight_source[0], sel[n]), m.bias)

    if args.init_map is not None:
        start_map = torch.load(args.init_map, map_location='cpu', weights_only=True)
        assert list(start_map) == list(sel), 'the map was made for other modules'
        for n in sel:
            sel[n].copy_(start_map[n].to(sel[n].device))
            apply(n)
        report['init_map'] = dict(path=str(args.init_map), sha256=digest_file(args.init_map),
                                  e0m3_units=sum(int(s.sum()) for s in sel.values()))
    def installed(n):
        # legacy: the weight apply() installed; lean: the weight the lean forward decodes for sel
        return store.decode(n, sel[n]) if lean else modules[n].weight

    if native is not None or (lean and not bf16_only):
        # The packed native weight of a map must decode to the weight apply() installs, bitwise:
        # the start map, one random mixed map, and the start map again after restoring it. Lean: the
        # Triton decode of each map must equal the PyTorch select + decode_packed path bitwise.
        verify = (lambda: native.verify_map(sel, installed)) if native is not None else (lambda: store.verify_map(sel))
        check = dict(start_map_mismatches=verify())
        if lean and native is not None:
            check['lean_start_map_mismatches'] = store.verify_map(sel)
        start_sel = {n: s.clone() for n, s in sel.items()}
        mixer = torch.Generator(device=device).manual_seed(0)
        for n in sel:
            sel[n].copy_(torch.rand(sel[n].shape, generator=mixer, device=sel[n].device) < 0.5)
            apply(n)
        check['mixed_map_units'] = sum(int(s.sum()) for s in sel.values())
        check['mixed_map_mismatches'] = verify()
        if lean and native is not None:
            check['lean_mixed_map_mismatches'] = store.verify_map(sel)
        for n in sel:
            sel[n].copy_(start_sel[n])
            apply(n)
        check['restored_start_mismatches'] = verify()
        if lean and native is not None:
            check['lean_restored_start_mismatches'] = store.verify_map(sel)
        assert not any(v for key, v in check.items() if key.endswith('mismatches')), check
        if native is not None:
            record = dict(kernel=native.kern.cfg, library=str(native.kern.path), tokens_per_forward=native.tokens,
                          verification=check)
            if args.shadow_native:
                report['shadow_native'] = dict(record, tries=[])
            else:
                report['native'] = record
            print('NATIVE ' + json.dumps(check), flush=True)
        if lean:
            report['lean'] = dict(verification=check)
            print('LEAN ' + json.dumps(check), flush=True)

    eval_handles = []
    act_kind = ['four_over_six']            # fake evaluation activation rule: FourOverSix, or NVFP4 (its map)

    act_checks = [0]
    row_checks = [0]
    current_batch = [1]
    input_layouts = {}

    def per_document_act(module, inputs):
        # Tensor-wide activation scales are computed per document, exactly as with
        # one document per forward pass, however many documents are batched. The
        # vectorized quantizer is checked bitwise against quant_nvfp4_4over6 on the
        # first calls of every run.
        x = inputs[0]
        batch = current_batch[0]
        if batch > 1:
            input_layouts.setdefault(str((tuple(x.shape), batch)), 0)
            input_layouts[str((tuple(x.shape), batch))] += 1
        if batch == 1 or (x.dim() >= 3 and x.shape[0] == batch):
            view = x
        elif x.shape[0] % batch == 0 and x.shape[0] // batch > 1:
            # Some modules receive the batch flattened to (documents * tokens, features);
            # restore the document axis so each document keeps its own scale.
            view = x.reshape(batch, -1, x.shape[-1])
        else:
            raise RuntimeError(f'Cannot recover the document axis of a {tuple(x.shape)} input at batch {batch}')
        if act_kind[0] == 'nvfp4':
            # NVFP4 baseline: quant_nvfp4 with one tensor-wide scale per document
            docs = view if view.dim() >= 3 else view[None]
            return (torch.stack([quant_nvfp4(d, 4, 16) for d in docs]).reshape(x.shape), *inputs[1:])
        if args.fused_act_quant:
            with region('act_quant_doc (fake evaluation)'):
                q = fourover6(view, documents=view.shape[0] if view.dim() >= 3 else 1)
            if act_checks[0] < 64:
                assert torch.equal(q.view(torch.int16), quant_per_document(view).view(torch.int16)) and check_act(view), \
                    'fused per-document activation quantizer differs from quant_per_document / quant_nvfp4_4over6'
                act_checks[0] += 1
            return (q.reshape(x.shape), *inputs[1:])
        if act_checks[0] < 64:
            assert check_act(view), 'vectorized activation quantizer differs from quant_nvfp4_4over6'
            act_checks[0] += 1
        with region('act_quant_doc (fake evaluation)'):
            q = quant_per_document(view)
        return (q.reshape(x.shape), *inputs[1:])

    def per_sequence_losses(lp, ids, t):
        # lp, t: (B, T-1, V) log-probabilities; CE and KL averaged over each sequence's tokens.
        ce = F.nll_loss(lp.transpose(1, 2), ids[:, 1:].to(lp.device), reduction='none').mean(-1)
        kl = (t.exp() * (t - lp)).sum(-1).mean(-1)
        return ce, kl

    def eval_hooks(on):
        nonlocal eval_handles
        for h in eval_handles:
            h.remove()
        eval_handles = [m.register_forward_pre_hook(per_document_act) for m in modules.values()] if on else []

    def dev_eval():
        eval_hooks(True)
        ce, kl = [], []
        for start in range(0, len(dev), args.eval_batch):
            chunk = dev[start:start + args.eval_batch]
            ids = torch.cat([r['ids'] for r in chunk]).to(device)
            current_batch[0] = ids.shape[0]
            if args.chunked_loss:
                logits = model(input_ids=ids, use_cache=False).logits
                c, k = chunked_loss.per_sequence_losses(logits, ids, dev_teacher[start:start + args.eval_batch], logits.device)
                ce.extend(c.tolist()); kl.extend(k.tolist())
                del logits
                continue
            lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            with region('teacher host-to-device'):
                t = torch.cat(dev_teacher[start:start + args.eval_batch]).to(lp.device).float()
            with region('logits log_softmax + CE/KL'):
                c, k = per_sequence_losses(lp, ids, t)
            ce.extend(c.tolist()); kl.extend(k.tolist())
            del lp, t
        current_batch[0] = 1
        eval_hooks(False)
        return dict(ce=sum(ce) / len(ce), kl=sum(kl) / len(kl), ce_nll=ce, kl_values=kl)

    def native_dev_eval():
        # dev_eval with every scoped Linear on the native kernel instead of the activation hooks
        with region('native weight build'):
            native.install(sel)
        ce, kl = [], []
        for start in range(0, len(dev), args.eval_batch):
            chunk = dev[start:start + args.eval_batch]
            ids = torch.cat([r['ids'] for r in chunk]).to(device)
            native.documents = ids.shape[0]
            if args.chunked_loss:
                logits = model(input_ids=ids, use_cache=False).logits
                c, k = chunked_loss.per_sequence_losses(logits, ids, dev_teacher[start:start + args.eval_batch], logits.device)
                ce.extend(c.tolist()); kl.extend(k.tolist())
                del logits
                continue
            lp = model(input_ids=ids, use_cache=False).logits[:, :-1].float().log_softmax(-1)
            with region('teacher host-to-device'):
                t = torch.cat(dev_teacher[start:start + args.eval_batch]).to(lp.device).float()
            with region('logits log_softmax + CE/KL'):
                c, k = per_sequence_losses(lp, ids, t)
            ce.extend(c.tolist()); kl.extend(k.tolist())
            del lp, t
        native.remove()
        return dict(ce=sum(ce) / len(ce), kl=sum(kl) / len(kl), ce_nll=ce, kl_values=kl)

    def final_eval(backend, batches, label, nvfp4=False):
        # run_multiround's WikiText-2 / C4 evaluation: one 2048-token window per forward, with the
        # per-window FourOverSix activation of per_document_act (fake), the native kernel with one
        # activation scale per window (native), or no quantization (bf16, pristine weights installed).
        # nvfp4: the NVFP4 baseline, NVFP4 weights and per-window tensor-wide NVFP4 activations.
        out = {}
        if backend == 'fake':
            act_kind[0] = 'nvfp4' if nvfp4 else 'four_over_six'
            eval_hooks(True)
        elif backend == 'native':
            if nvfp4:
                native.set_activation('nvfp4_rows')
                native.install_fixed(lambda n: store.nvfp4[n])
            else:
                native.install(sel)
            native.documents = 1
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
            out[key] = dict(nll=values, ppl=float(torch.exp(losses.sum() / (len(values) * 2048))))
            print(f'PPL {label} {key} {out[key]["ppl"]:.6f}', flush=True)
        if backend == 'fake':
            eval_hooks(False)
            act_kind[0] = 'four_over_six'
        elif backend == 'native':
            native.remove()
            native.set_activation('four_over_six_rows')
        return out

    def difference(nat, fake):
        out = {}
        for key, values in (('kl', 'kl_values'), ('ce', 'ce_nll')):
            d = [x - y for x, y in zip(nat[values], fake[values])]
            out[key] = dict(mean_native_minus_fake=nat[key] - fake[key],
                            mean_abs_per_document=sum(abs(x) for x in d) / len(d),
                            max_abs_per_document=max(abs(x) for x in d))
        return out

    def improves(new, old):
        keys = ('ce', 'kl') if args.objective == 'both' else (args.objective,)
        if not args.significant_steps:
            return all(new[key] < old[key] for key in keys)
        for key, values in (('ce', 'ce_nll'), ('kl', 'kl_values')):
            if key not in keys:
                continue
            diff = torch.tensor(new[values], dtype=torch.float64) - torch.tensor(old[values], dtype=torch.float64)
            if float(diff.mean() + 2 * diff.std(unbiased=True) / math.sqrt(len(diff))) >= 0:
                return False
        return True

    def score():
        """Per-unit mean and SE of CE and KL directional scores for flipping each unit now."""
        sums = {n: [torch.zeros(sel[n].shape, dtype=torch.float64, device=sel[n].device) for _ in range(4)] for n in modules}
        phase = [0]

        def act(module, inputs):
            x = inputs[0]
            with region('act_quant_rows (scoring)'):
                if args.fused_act_quant:
                    q = fourover6_rows(x.detach())
                    if row_checks[0] < 64:
                        assert torch.equal(q.view(torch.int16), quantize_rows(x.detach()).view(torch.int16)), \
                            'fused per-token activation quantizer differs from quantize_rows'
                        row_checks[0] += 1
                else:
                    q = quantize_rows(x.detach())
            # straight-through: the same expression either way (it also turns a -0 of q into +0)
            return (q + (x - x.detach()), *inputs[1:])

        def make_hook(n):
            def forward(module, inputs, output):
                x = inputs[0].detach()
                x = x.reshape(x.shape[0], -1, x.shape[-1]) if x.dim() >= 3 else x.reshape(1, -1, x.shape[-1])

                def backward(dy):
                    # Each sequence's loss reaches only its own slice of dy, so
                    # dy[i]^T x[i] is exactly sequence i's weight gradient.
                    dy = dy.detach().reshape(x.shape[0], -1, dy.shape[-1])
                    if args.tile_score_kernel:
                        # B1: every sequence's tile scores in one launch, added into the FP64 sums in sequence order
                        with region('hook: B1 fused tile scores'):
                            s = sums[n]
                            tile_scores(dy, x, store.cand[n], sel[n], rows, cols, store._luts(dy.device),
                                        s[2 * phase[0]], s[2 * phase[0] + 1])
                        return
                    with region('hook: candidate decode + D'):
                        b = base(n)
                        d = (alt(n).float() - b.float()) * torch.where(expand(sel[n], rows, cols, b.shape[0]), -1., 1.)
                        del b
                    s = sums[n]
                    for i in range(x.shape[0]):
                        # the same operations as reduce((dy[i]^T x[i]) * d), with the same tensor lifetimes
                        with region('hook: G = dy^T x (FP32 matmul)'):
                            g = dy[i].float().T @ x[i].float()
                        with region('hook: G*D, tile reduction, FP64 accumulation'):
                            gd = g * d
                            del g
                            value = reduce(gd, rows, cols).double()
                            del gd
                            s[2 * phase[0]] += value
                            s[2 * phase[0] + 1] += value.square()
                output.register_hook(backward)
            return forward

        handles = [m.register_forward_pre_hook(act) for m in modules.values()]
        handles += [m.register_forward_hook(make_hook(n)) for n, m in modules.items()]
        with torch.enable_grad():
            for start in range(0, len(fit), args.score_batch):
                ids = torch.cat(fit[start:start + args.score_batch]).to(device)
                embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
                if args.chunked_loss:
                    # the KL backward's logit gradient chunk by chunk, then the model's backward from the logits
                    logits = model(inputs_embeds=embeds, use_cache=False).logits
                    grad = chunked_loss.kl_logit_gradient(logits, teacher[start:start + args.score_batch], logits.device)
                    phase[0] = 1
                    logits.backward(grad)
                    del embeds, logits, grad
                    continue
                if profile_regions.ON[0]:
                    logits = model(inputs_embeds=embeds, use_cache=False).logits
                    with region('logits log_softmax + CE/KL'):
                        lp = logits[:, :-1].float().log_softmax(-1)
                        del logits
                else:
                    lp = model(inputs_embeds=embeds, use_cache=False).logits[:, :-1].float().log_softmax(-1)
                with region('teacher host-to-device'):
                    t = torch.cat(teacher[start:start + args.score_batch]).to(lp.device).float()
                with region('logits log_softmax + CE/KL'):
                    ce, kl = per_sequence_losses(lp, ids, t)
                if not args.skip_ce_backward:
                    phase[0] = 0; ce.sum().backward(retain_graph=True)
                phase[0] = 1; kl.sum().backward()
                del embeds, lp, t, ce, kl
        for h in handles:
            h.remove()
        n_seq = len(fit)
        stats = {}
        for n, (cs, cq, ks, kq) in sums.items():
            out = []
            for total, square in ((cs, cq), (ks, kq)):
                mean = total / n_seq
                se = ((square - n_seq * mean.square()).clamp_min(0) / (n_seq - 1)).sqrt() / math.sqrt(n_seq)
                out.append((mean, se))
            stats[n] = out
        return stats

    def validate_data():
        if args.model in PUBLISHED:
            validate_evaluation_data(report['data'], json.loads(Path(PUBLISHED[args.model]).read_text())['data'])
            report['data_validation'] = PUBLISHED[args.model]
        else:
            report['data_validation'] = 'no published record for this model; windows recorded by token hash'

    if evaluate_only:
        monitor.enter('final_evaluation')
        batches, report['data'] = data(tok, prior, 2048)
        validate_data()
        report['evaluations'] = {}
        for label, path in maps_to_evaluate.items():
            entry = report['evaluations'][label] = dict(path=path, backend='fake' if path == 'bf16' else args.eval_backend)
            if path == 'bf16':
                if bf16_only:
                    pass                                    # the model as loaded
                elif lean:
                    weight_source[0] = 'bf16'
                else:
                    for n, m in modules.items():
                        m.weight.copy_(pristine[n].to(m.weight.device))
                entry['evaluation'] = final_eval('bf16', batches, label)
                if lean:
                    weight_source[0] = 'map'
                elif not bf16_only:
                    for n in sel:
                        apply(n)
                save(args.out, report)
                continue
            if path == 'nvfp4':
                # the store's NVFP4 candidate: its Triton decode must equal the PyTorch decode bitwise
                entry['nvfp4_decode_mismatches'] = [
                    n for n in store.nvfp4 if not torch.equal(store.decode(n, which='nvfp4').view(torch.int16),
                                                              store.reference_decode(n, which='nvfp4').view(torch.int16))]
                assert not entry['nvfp4_decode_mismatches'], label
                if native is not None:
                    native.checked = 0
                if args.eval_backend == 'fake':
                    weight_source[0] = 'nvfp4'
                entry['evaluation'] = final_eval(args.eval_backend, batches, label, nvfp4=True)
                weight_source[0] = 'map'
                if native is not None:
                    entry['native_activation_checks'] = native.checked
                save(args.out, report)
                continue
            if path == 'fourover6':
                for n in sel:
                    sel[n].zero_()
                    apply(n)
            else:
                saved_map = torch.load(path, map_location='cpu', weights_only=True)
                assert list(saved_map) == list(sel), label
                for n in sel:
                    chosen = saved_map[n].to(sel[n].device)
                    if chosen.shape != sel[n].shape:
                        # A 256x64 or 16x64 map in an 8x64 process: every coarse tile is whole 8-row tiles (32 or 2),
                        # so the element mask, hence the weight, is unchanged (checked).
                        o = prior['matrices'][n]['shape'][0]
                        coarse = [u for u in ('256x64', '16x64') if chosen.shape == (-(-o // UNITS[u][0]), sel[n].shape[1])]
                        assert args.unit == '8x64' and len(coarse) == 1, label
                        big = UNITS[coarse[0]][0]
                        fine = chosen.repeat_interleave(big // rows, 0)[:sel[n].shape[0]]
                        assert torch.equal(expand(fine, rows, cols, o), expand(chosen, big, cols, o)), (label, n)
                        entry['converted_from'] = coarse[0]
                        entry[f'e0m3_units_{coarse[0]}'] = entry.get(f'e0m3_units_{coarse[0]}', 0) + int(chosen.sum())
                        chosen = fine
                    sel[n].copy_(chosen)
                    apply(n)
                entry['sha256'] = digest_file(path)
            entry['e0m3_units'] = sum(int(s.sum()) for s in sel.values())
            if lean:
                entry['lean_map_mismatches'] = store.verify_map(sel)
                assert not entry['lean_map_mismatches'], label
            if native is not None:
                entry['native_map_mismatches'] = native.verify_map(sel, installed)
                assert not entry['native_map_mismatches'], label
                native.checked = 0          # the first 64 activation calls of every map are checked
            entry['evaluation'] = final_eval(args.eval_backend, batches, label)
            if native is not None:
                entry['native_activation_checks'] = native.checked
            save(args.out, report)
        monitor.close()
        phases = monitor.summary()
        report['resources'] = dict(total_seconds=time.time() - started,
                                   gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                                   gpu_peak_allocated_gib=phases['gpu_peak_allocated_gib'],
                                   gpu_peak_reserved_gib=phases['gpu_peak_reserved_gib'],
                                   cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                                   phases=phases)
        report['status'] = 'complete'
        save(args.out, report)
        return
    if args.profile is not None:
        # One fake development evaluation, one native one (native runs) and one scoring pass under
        # torch.profiler; the GPU time of every kernel goes to its innermost named region.
        phases = ['phase: fake development evaluation'] + (['phase: native development evaluation']
                                                          if native is not None else []) + ['phase: scoring pass']
        regions = ['act_quant_rows (scoring)', 'act_quant_doc (fake evaluation)', 'hook: candidate decode + D',
                   'hook: G = dy^T x (FP32 matmul)', 'hook: G*D, tile reduction, FP64 accumulation',
                   'hook: B1 fused tile scores',
                   'lean weight decode', 'logits log_softmax + CE/KL', 'teacher host-to-device',
                   'native weight build', 'native: activation quantization', 'native: FP4 GEMM',
                   'native: epilogue']
        profile_regions.ON[0] = True
        wall = {}
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for phase, fn in zip(phases, [dev_eval] + ([native_dev_eval] if native is not None else []) + [score]):
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.profiler.record_function(phase):
                    fn()
                    torch.cuda.synchronize()
                wall[phase] = time.time() - t0
        profile_regions.ON[0] = False
        trace = args.profile.with_suffix('.trace.json')
        prof.export_chrome_trace(str(trace))
        result, counted = profile_regions.breakdown_trace(trace, phases, regions)
        result['kernels_counted'] = counted
        for phase in phases:
            result[phase]['wall_seconds_with_profiler'] = wall[phase]
        result['trace'] = str(trace)
        args.profile.write_text(json.dumps(dict(model=args.model, unit=args.unit, memory_mode=args.memory_mode,
                                                eval_batch=args.eval_batch, score_batch=args.score_batch,
                                                breakdown=result), indent=1) + '\n')
        print('PROFILE ' + json.dumps({p: dict(gpu_ms_total=result[p]['gpu_ms_total'],
                                              wall=result[p]['wall_seconds_with_profiler']) for p in phases}), flush=True)
        report['status'] = 'profile_complete'
        save(args.out, report)
        return
    monitor.enter('initial_dev_eval')
    t_fake = time.time()
    current = dev_eval()
    t_fake = time.time() - t_fake
    report['initial_dev'] = current
    if args.dev_backend == 'native':
        # The fake start value is logged; from here on the native development KL decides.
        report['fake_initial_dev'] = current
        monitor.enter('native_dev_evaluation')
        current = native_dev_eval()
        report['initial_dev'] = current
        monitor.enter('initial_dev_eval')
    dev_values = dict(initial=dict(ce=torch.tensor(current['ce_nll'], dtype=torch.float64),
                                   kl=torch.tensor(current['kl_values'], dtype=torch.float64)),
                      fake_initial=None, tries=[])
    if 'fake_initial_dev' in report:
        dev_values['fake_initial'] = dict(ce=torch.tensor(report['fake_initial_dev']['ce_nll'], dtype=torch.float64),
                                          kl=torch.tensor(report['fake_initial_dev']['kl_values'], dtype=torch.float64))
    if args.shadow_native:
        monitor.enter('native_dev_evaluation')
        t_native = time.time()
        native_current = native_dev_eval()
        report['shadow_native']['initial'] = dict(
            native_current, fake_seconds=t_fake, native_seconds=time.time() - t_native,
            native_rebuild_seconds=native.rebuild_seconds, activation_checks=native.checked,
            difference=difference(native_current, current))
        print('NATIVE initial ' + json.dumps(report['shadow_native']['initial']['difference']), flush=True)
        monitor.enter('initial_dev_eval')
    save(args.out, report)
    timing = dict(setup_seconds=time.time() - started)
    if args.check_start:
        report['input_layouts'] = input_layouts
        print('LAYOUTS ' + json.dumps(input_layouts), flush=True)
        report['status'] = 'check_start_complete'
        save(args.out, report)
        print(f'START {args.objective} dev CE {current["ce"]:.6f} KL {current["kl"]:.6f}', flush=True)
        return
    print(f'START {args.objective} dev CE {current["ce"]:.6f} KL {current["kl"]:.6f}', flush=True)
    names = list(modules)
    previous_accepted = None
    for rnd in range(args.max_rounds):
        if time.time() - started > args.budget_hours * 3600:
            report['stopped'] = 'time budget'; break
        monitor.enter('scoring')
        t0 = time.time()
        stats = score()
        if rnd == 0 and args.dump_round0_scores:
            torch.save({n: dict(ce_mean=cm.cpu(), ce_se=cse.cpu(), kl_mean=km.cpu(), kl_se=kse.cpu())
                        for n, ((cm, cse), (km, kse)) in stats.items()}, args.out / 'round0_scores.pt')
            report['round0_scores_sha256'] = digest_file(args.out / 'round0_scores.pt')
        if args.stop_after_scoring:
            if args.record_dev_values:
                torch.save(dev_values, args.out / 'dev_values.pt')
            report['score_seconds'] = time.time() - t0
            monitor.close()
            phases = monitor.summary()
            report['resources'] = dict(gpu_peak_allocated_gib=phases['gpu_peak_allocated_gib'],
                                       gpu_peak_reserved_gib=phases['gpu_peak_reserved_gib'],
                                       cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                                       phases=phases)
            report['status'] = 'scores_complete'
            save(args.out, report)
            print('STOP after round 0 scoring', flush=True)
            return
        bounds, flat_mean_ce, flat_mean_kl, owners = [], [], [], []
        for i, n in enumerate(names):
            (cm, cse), (km, kse) = stats[n]
            ub = {'ce': cm + args.filter_k * cse, 'kl': km + args.filter_k * kse}
            u = (torch.maximum(ub['ce'], ub['kl']) if args.objective == 'both' else ub[args.objective]).reshape(-1)
            bounds.append(u.to(device)); flat_mean_ce.append(cm.reshape(-1).to(device)); flat_mean_kl.append(km.reshape(-1).to(device))
            owners.append(torch.full_like(u, i, dtype=torch.int32, device=device))
        bounds = torch.cat(bounds); owners = torch.cat(owners)
        flat_mean_ce = torch.cat(flat_mean_ce); flat_mean_kl = torch.cat(flat_mean_kl)
        offsets = np.cumsum([0] + [sel[n].numel() for n in names])
        candidates = (bounds < 0).nonzero().squeeze(-1)
        candidates = candidates[bounds[candidates].argsort()]
        score_seconds = time.time() - t0
        entry = dict(round=rnd, candidates=int(candidates.numel()), score_seconds=score_seconds, tries=[])
        if candidates.numel() == 0:
            entry['accepted'] = 0; report['rounds'].append(entry); report['stopped'] = 'no candidates'; break
        size, accepted = int(candidates.numel()), None
        if args.warm_start and previous_accepted:
            size = min(size, 2 * previous_accepted)
        for _ in range(args.max_tries):
            monitor.enter('dev_evaluation')
            chosen = candidates[:size]
            touched = {}
            who = owners[chosen]
            for i in who.unique().tolist():
                touched[names[i]] = chosen[who == i] - int(offsets[i])
            for n, idx in touched.items():
                flat = sel[n].view(-1); flat[idx.to(flat.device)] ^= True; apply(n)
            t_fake = time.time()
            new = native_dev_eval() if args.dev_backend == 'native' else dev_eval()
            t_fake = time.time() - t_fake
            if args.shadow_native:
                monitor.enter('native_dev_evaluation')
                t_native = time.time()
                native_new = native_dev_eval()
                t_native = time.time() - t_native
                monitor.enter('dev_evaluation')
            pce, pkl = float(flat_mean_ce[chosen].sum()), float(flat_mean_kl[chosen].sum())
            entry['tries'].append(dict(size=size, predicted_ce=pce, predicted_kl=pkl,
                                       dev_delta_ce=new['ce'] - current['ce'], dev_delta_kl=new['kl'] - current['kl']))
            dev_values['tries'].append(dict(round=rnd, try_index=len(entry['tries']) - 1, size=size,
                                            ce=torch.tensor(new['ce_nll'], dtype=torch.float64),
                                            kl=torch.tensor(new['kl_values'], dtype=torch.float64)))
            print(f'ROUND {rnd} try size={size} pred CE {pce:+.6f} KL {pkl:+.6f} | dev dCE {new["ce"] - current["ce"]:+.6f} '
                  f'dKL {new["kl"] - current["kl"]:+.6f}', flush=True)
            fake_accepts = improves(new, current)
            if args.shadow_native:
                # The native state mirrors fake's: its "current" moves only when fake accepts.
                native_accepts = improves(native_new, native_current)
                d_fake, d_native = new['kl'] - current['kl'], native_new['kl'] - native_current['kl']
                report['shadow_native']['tries'].append(dict(
                    round=rnd, try_index=len(entry['tries']) - 1, size=size,
                    fake_new_kl=new['kl'], fake_current_kl=current['kl'], fake_new_ce=new['ce'],
                    fake_current_ce=current['ce'], fake_accepts=fake_accepts,
                    native_new_kl=native_new['kl'], native_current_kl=native_current['kl'],
                    native_new_ce=native_new['ce'], native_current_ce=native_current['ce'],
                    native_accepts=native_accepts, agree=native_accepts == fake_accepts,
                    delta_fake=d_fake, delta_native=d_native, discrepancy=abs(d_native - d_fake),
                    margin=abs(d_fake), fake_seconds=t_fake, native_seconds=t_native,
                    native_rebuild_seconds=native.rebuild_seconds,
                    fake_kl_values=new['kl_values'], fake_ce_nll=new['ce_nll'],
                    native_kl_values=native_new['kl_values'], native_ce_nll=native_new['ce_nll']))
                print(f'SHADOW round {rnd} size={size} fake dKL {d_fake:+.6f} {"accept" if fake_accepts else "reject"} | '
                      f'native dKL {d_native:+.6f} {"accept" if native_accepts else "reject"} | '
                      f'{"agree" if native_accepts == fake_accepts else "DISAGREE"} | '
                      f'fake {t_fake:.1f}s native {t_native:.1f}s', flush=True)
                if fake_accepts:
                    native_current = native_new
            if fake_accepts:
                accepted = size; current = new; break
            for n, idx in touched.items():
                flat = sel[n].view(-1); flat[idx.to(flat.device)] ^= True; apply(n)
            if size == 1:
                break
            size = max(1, size // 2)
        entry['accepted'] = accepted or 0
        previous_accepted = accepted
        entry['e0m3_units'] = sum(int(s.sum()) for s in sel.values())
        entry['dev_ce'], entry['dev_kl'] = current['ce'], current['kl']
        entry['round_seconds'] = time.time() - t0
        report['rounds'].append(entry)
        torch.save({n: s.cpu() for n, s in sel.items()}, args.out / 'map.pt')
        if args.record_dev_values:
            torch.save(dev_values, args.out / 'dev_values.pt')
        save(args.out, report)
        print(f'ROUND {rnd} accepted={accepted} e0m3={entry["e0m3_units"]} dev CE {current["ce"]:.6f} KL {current["kl"]:.6f} '
              f'{entry["round_seconds"]:.0f}s', flush=True)
        if not accepted:
            report['stopped'] = 'no step lowers the development objective'; break
    report['final_dev'] = current
    timing['optimization_seconds'] = time.time() - started - timing['setup_seconds']
    if args.record_dev_values:
        torch.save(dev_values, args.out / 'dev_values.pt')
        report['dev_values_sha256'] = digest_file(args.out / 'dev_values.pt')
    if native is not None:
        report['native_builds'] = dict(count=native.builds, seconds=native.build_seconds_total)
    report['final_e0m3_units'] = sum(int(s.sum()) for s in sel.values())
    report['map_sha256'] = digest_file(args.out / 'map.pt') if (args.out / 'map.pt').exists() else None
    save(args.out, report)
    # One evaluation of the final map on the released PPL windows.
    monitor.enter('final_evaluation')
    batches, report['data'] = data(tok, prior, 2048)
    validate_data()
    if args.eval_backend == 'native':
        report['evaluation'] = final_eval('native', batches, f'multiround_{args.unit}_{args.objective}')
        save(args.out, report)
    else:
        eval_hooks(True)
        report['evaluation'] = {}
        for domain, sequences in batches.items():
            values = []
            for ids in sequences:
                ids = ids.to(device)
                logits = model(input_ids=ids, use_cache=(domain == 'wiki')).logits
                values.append(float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1).to(logits.device))))
                del logits
            losses = torch.tensor(values, dtype=torch.float32) * 2048
            key = 'c4' if domain == 'c4_paper' else domain
            report['evaluation'][key] = dict(nll=values, ppl=float(torch.exp(losses.sum() / (len(values) * 2048))))
            print(f'PPL multiround_{args.unit}_{args.objective} {key} {report["evaluation"][key]["ppl"]:.6f}', flush=True)
            save(args.out, report)
        eval_hooks(False)
    timing['evaluation_seconds'] = time.time() - started - timing['setup_seconds'] - timing['optimization_seconds']
    timing['total_seconds'] = time.time() - started
    monitor.close()
    phases = monitor.summary()
    # Peak statistics are reset at every phase start; the phases cover the whole run, so the
    # maximum over phases is the run-wide torch.cuda.max_memory_allocated/reserved.
    report['resources'] = dict(
        timing, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        gpu_peak_allocated_gib=phases['gpu_peak_allocated_gib'],
        gpu_peak_reserved_gib=phases['gpu_peak_reserved_gib'],
        cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
        scoring_passes=len(report['rounds']),
        development_evaluations=sum(len(r['tries']) for r in report['rounds']) + 1,
        phases=phases)
    print('RESOURCES ' + json.dumps(report['resources']), flush=True)
    report['status'] = 'complete'
    save(args.out, report)


if __name__ == '__main__':
    main()

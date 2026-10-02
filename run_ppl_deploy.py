"""B2: WikiText-2 / C4 perplexity through the SM120 deployment path, convention (c), on the released protocol windows.

    python run_ppl_deploy.py --model llama8b --data-root DATA_ROOT \\
        --evaluate BF16=bf16 \\
        --evaluate FourOverSix-fake=fake:four_over_six --evaluate NVFP4-fake=fake:nvfp4 \\
        --evaluate tc-8x64-fake=fake:map:ARTIFACT.mixfp4map \\
        --evaluate FourOverSix=native:ARTIFACT_FO6 --evaluate NVFP4=native:ARTIFACT_NVFP4 --evaluate tc-8x64=native:ARTIFACT \\
        --out DIR

Windows: run_baseline_protocol_audit.data(tok, prior, 2048), the released protocol (the WikiText-2 test split cut into
2048-token windows, and 256 C4 validation crops). They are validated against the published record
(validate_evaluation_data) where one exists, and compared by token hash with sm120/eval/reference. One window per
forward (batch 1). The per-window value is run_multiround.py's: the mean cross-entropy over the window's 2047
predicted tokens (float32 logits), with use_cache as there. PPL = exp of the mean over windows. So every number pairs
window by window with run_multiround.py's evaluations (conventions (a)).

Policies, convention (c) (per-token activation scales):
  bf16                  the model as loaded (no quantization).
  fake:four_over_six | fake:nvfp4 | fake:map:<.mixfp4map> | fake:weights:<state.pt> | fake:format:<rule>:<unit>
                        fake quant, the like-for-like reference for the kernel:
                        - the fake-quant weight installed in BF16 (FourOverSix, NVFP4, or a map's tiles);
                        - every quantized Linear input quantized per token and dequantized (FourOverSix rows for
                          four_over_six and maps, NVFP4 rows for nvfp4);
                        - then a BF16 GEMM (sm120/eval/common.FakeQuant).
                        fake:weights:<state.pt> is FourOverSix of trained weights (run_cost_distill.py --arm qat
                        state.pt, {module: BF16 weight}) instead of the model's, the reference for a QAT artifact
                        (results/unified_baselines).
                        fake:format:<rule>:<unit> (results/paper_extra/A): the weights of IF4 (rule if4, Cook et al.)
                        or MixFP4 (rule zou, Zou et al.), quantize/adaptive_formats.py, chosen per 16-element block
                        (unit 1x16) or elected per tile (8x64, 16x64, 256x64); FourOverSix per-token activations.
                        The record holds the share of blocks in the uniform format, overall and per projection.
  fake:w4a4:<rule>      (results/main_ppl) W4A4 under IF4 (if4) or MixFP4 (zou): the rule's 1x16 weights as
                        fake:format:<rule>:1x16, and every quantized Linear input quantized by the same rule per
                        16-element block with a per-token tensor scale (quantize/adaptive_formats.quantize_rows); the
                        record holds the activations' uniform-format share (activation_format).
  native:<artifact>[:<kernel>]
                        NativeLinear: every quantized Linear on the SM120 kernel (the per-token activation quantizer,
                        then the FP4 GEMM with a one-rounding epilogue). Default kernel, by the artifact's type block:
                        16x64 (and 256x64 exported as 16x64 granules) 'auto', the width-selecting weights-on-A mixed
                        set; 8x64 'n8k64_wB' (weights on B; 'auto' cannot execute 8-row granules); E2M1-only
                        (FourOverSix, NVFP4) 'auto_stock', the stock weights-on-A set.
Fake and BF16 policies run first. Native policies replace the Linears, so they run last. Every native policy is checked
for coverage: in every forward, every quantized Linear ran natively. The dense GEMMs of one forward are recorded
(sm120/eval/ppl.py gemm_audit).
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent
SM120 = REPO / 'sm120'
sys.path.insert(0, str(SM120))
from mixfp4_sm120 import model as NM  # noqa: E402
from run_baseline_protocol_audit import data  # noqa: E402
from run_c4_frozen import digest_file  # noqa: E402
from run_conditional_format import sha  # noqa: E402
from run_math_code_calibration import load_model  # noqa: E402
from run_multiround import PUBLISHED, data_paths  # noqa: E402
from run_task_reorder_eval import validate_evaluation_data  # noqa: E402


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse(spec):
    label, _, rest = spec.partition('=')
    parts = rest.split(':')
    if parts[0] == 'bf16':
        return dict(label=label, kind='bf16')
    if parts[0] == 'fake' and parts[1] in ('four_over_six', 'nvfp4'):
        return dict(label=label, kind='fake', weight=parts[1])
    if parts[0] == 'fake' and parts[1] == 'map':
        return dict(label=label, kind='fake', weight='map', map=':'.join(parts[2:]))
    if parts[0] == 'fake' and parts[1] == 'weights':
        return dict(label=label, kind='fake', weight='four_over_six', state=':'.join(parts[2:]))
    if parts[0] == 'fake' and parts[1] == 'format':
        return dict(label=label, kind='fake', weight='format', rule=parts[2], unit=parts[3])
    if parts[0] == 'fake' and parts[1] == 'w4a4':
        return dict(label=label, kind='fake', weight='format', rule=parts[2], unit='1x16', act='rule')
    if parts[0] == 'native':
        return dict(label=label, kind='native', artifact=parts[1], kernel=parts[2] if len(parts) > 2 else None)
    raise SystemExit(f'bad --evaluate {spec!r}')


@torch.no_grad()
def evaluate(model, batches, device, label, first_forward=None):
    out, forwards = {}, 0
    for domain, sequences in batches.items():
        values = []
        for i, ids in enumerate(sequences):
            ids = ids.to(device)
            logits = model(input_ids=ids, use_cache=(domain == 'wiki')).logits
            values.append(float(F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                                ids[:, 1:].reshape(-1).to(logits.device))))
            del logits
            forwards += 1
            if i == 0 and first_forward is not None:
                first_forward(domain, forwards)
            if not math.isfinite(values[-1]):
                raise SystemExit(f'non-finite NLL: {label} {domain} window {i}')
        losses = torch.tensor(values, dtype=torch.float32) * 2048
        key = 'c4' if domain == 'c4_paper' else domain
        out[key] = dict(nll=values, ppl=float(torch.exp(losses.sum() / (len(values) * 2048))))
        print(f'PPL {label} {key} {out[key]["ppl"]:.6f}', flush=True)
    return out


def load_for_evaluation(key, prior, C):
    """(model, {name: quantized Linear}) as run_multiround.py loads it, checked against sm120's scope and the record."""
    if C.MODELS[key]['loader'] == 'qwen3_5_conditional':
        from transformers import Qwen3_5ForConditionalGeneration
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            prior['source'], revision=prior['revision'], dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map='cuda', output_loading_info=True)
        assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
        model.eval().requires_grad_(False)
        modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                   and 'language_model' in n and 'head' not in n}
        assert list(modules) == list(prior['matrices'])
    else:
        model, modules = load_model(prior, False)
        model.set_attn_implementation('sdpa')
    assert list(modules) == list(C.scope(model, key)), 'our scope differs from sm120 model.scope'
    wrong = [n for n, m in modules.items() if sha(m.weight) != prior['matrices'][n]['source_sha256']]
    assert not wrong, f'weights differ from the calibration record: {wrong[:3]}'
    return model, modules


@torch.no_grad()
def install_format(fq, pol, C):
    """fake:format:<rule>:<unit>: IF4 / MixFP4 weights (quantize/adaptive_formats.py) with FourOverSix per-token
    activations, installed on FakeQuant's modules like its own policies. Returns the record fields."""
    from quantize import adaptive_formats as AF
    from quantize.causal_four_over_six import quantize_rows
    assert pol['rule'] in AF.RULES, pol['rule']
    tile = tuple(int(v) for v in pol['unit'].split('x'))
    fq.remove()
    h = hashlib.sha256()
    total, per_proj, zero, sq = dict(blocks=0, uniform=0), {}, 0, 0.0
    mix = {}                                   # within-tile mixing of the per-block choices (1x16 arms only)
    for n, m in fq.modules.items():
        q, st = AF.quantize(fq.pristine[n].to(m.weight.device), pol['rule'], tile)
        m.weight.copy_(q)
        h.update(n.encode())
        h.update(q.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        proj = per_proj.setdefault(n.rsplit('.', 1)[-1], dict(blocks=0, uniform=0))
        for d in (total, proj):
            d['blocks'] += st['blocks']
            d['uniform'] += st['uniform_blocks']
        zero += st['zero_scale_blocks']
        sq += st['sq_error']
        for mtile, x in st.get('mixing', {}).items():        # not `tile`: that is the policy's own unit
            for key in ('all', n.rsplit('.', 1)[-1]):
                agg = mix.setdefault(mtile, {}).setdefault(key, dict(tiles=0, mixed=0, minority_sum=0.0))
                for k2 in agg:
                    agg[k2] += x[k2]
    if pol.get('act') == 'rule':
        # results/main_ppl: W4A4 under the rule; every quantized Linear input by the rule per 16-block, tensor scale
        # per token (AF.quantize_rows)
        assert pol['rule'] in AF.ACT_RULES and tile == (1, 16), pol
        pol['_act_stats'] = AF.RowStats()
        act = f"{pol['rule']}_rows (per 16-block selection, per-token tensor scale)"
        for m in fq.modules.values():
            fq.handles.append(m.register_forward_pre_hook(
                lambda mod, inp, _r=pol['rule'], _st=pol['_act_stats']:
                    (C._chunked(lambda t: AF.quantize_rows(t, _r, _st), inp[0]), *inp[1:])))
    else:
        act = 'four_over_six_rows'
        for m in fq.modules.values():
            fq.handles.append(m.register_forward_pre_hook(lambda mod, inp: (C._chunked(quantize_rows, inp[0]), *inp[1:])))
    frac = lambda d: d['uniform'] / d['blocks']  # noqa: E731
    return dict(backend='fake (c)', installed_weight_sha256=h.hexdigest(), activation=act,
                format=dict(rule=pol['rule'], unit=pol['unit'], tile=list(tile), uniform_fraction=frac(total),
                            uniform_fraction_by_projection={k: frac(v) for k, v in per_proj.items()},
                            blocks=total['blocks'], uniform_blocks=total['uniform'], zero_scale_blocks=zero,
                            weight_sq_error=sq,
                            mixing={t: {k: dict(v, mixed_fraction=v['mixed'] / v['tiles'],
                                                mean_minority_share=v['minority_sum'] / v['mixed'] if v['mixed'] else None)
                                        for k, v in by.items()} for t, by in mix.items()} or None))


@torch.no_grad()
def installed_error(fq):
    """The total squared error of the installed weights of the quantized Linears against the model's (float64)."""
    total = 0.0
    for n, m in fq.modules.items():
        total += float(((m.weight.double() - fq.pristine[n].to(m.weight.device).double()) ** 2).sum())
    return total


def install_fake(fq, pol, key, modules, C):
    """bf16 or fake (c) through sm120/eval/common.FakeQuant; returns the record fields."""
    if pol['kind'] == 'bf16':
        fq.install('bf16')
        return dict(backend='bf16')
    if pol['weight'] == 'format':
        return install_format(fq, pol, C)
    out = dict(backend='fake (c)')
    masks = type_block = None
    if pol['weight'] == 'map':
        header, masks, digest = C.read_map_for(key, pol['map'], modules)
        type_block = tuple(header['type_block'])
        out.update(map_sha256=digest, type_block=list(type_block), e0m3_tiles=header['totals']['selected_tiles'])
    if pol.get('state'):
        # trained weights (QAT): FakeQuant quantizes what it holds as the source weights, so they stand in for them
        state = torch.load(pol['state'], map_location='cpu', weights_only=True)
        assert list(state) == list(fq.pristine), 'the weight state\'s modules differ from the quantized scope'
        assert all(state[n].shape == w.shape and state[n].dtype == w.dtype for n, w in fq.pristine.items())
        saved, fq.pristine = fq.pristine, state
        try:
            out['installed_weight_sha256'] = fq.install(pol['weight'], masks, type_block)
        finally:
            fq.pristine = saved
        out.update(trained_weights=pol['state'], trained_weights_file_sha256=digest_file(pol['state']),
                   trained_modules_changed=sum(not torch.equal(state[n], w) for n, w in saved.items()))
        del state
        return out
    out['installed_weight_sha256'] = fq.install(pol['weight'], masks, type_block)
    return out


def install_native(model, pol, key, C):
    """NativeLinear (c) from an artifact; the kernel by its type block unless given. Returns the record fields."""
    meta = json.loads(Path(pol['artifact'], 'artifact.json').read_text())
    out = {}
    if pol['kernel'] is None:
        pol['kernel'] = out['kernel'] = {None: 'auto_stock', (16, 64): 'auto', (8, 64): 'n8k64_wB'}[
            tuple(meta['type_block']) if meta.get('type_block') else None]
    rep = NM.install(model, pol['artifact'], kernel=pol['kernel'], loader=C.MODELS[key]['loader'])
    out.update(backend='NativeLinear (c)', install=rep.as_dict(), artifact_weights_sha256=meta['weights_sha256'],
               artifact_map_sha256=(meta.get('map') or {}).get('sha256'), artifact_weight_kind=meta['weight_kind'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--evaluate', action='append', required=True, metavar='LABEL=SPEC')
    ap.add_argument('--transformers-deviation', action='store_true',
                    help='accept a transformers version other than the calibration record\'s (as run_multiround.py)')
    ap.add_argument('--limit-windows', type=int, default=None, help='smoke tests only: the first N windows per corpus')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    C = load_path('sm120_eval_common', SM120 / 'eval' / 'common.py')
    P = load_path('sm120_eval_ppl', SM120 / 'eval' / 'ppl.py')
    policies = [parse(s) for s in args.evaluate]
    policies = [p for p in policies if p['kind'] != 'native'] + [p for p in policies if p['kind'] == 'native']
    calibration, _ = data_paths(args.model, args.data_root)
    prior = json.loads((calibration / 'report.json').read_text())
    if transformers.__version__ != prior['transformers_version']:
        assert args.transformers_deviation, (transformers.__version__, prior['transformers_version'])
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', convention='(c) per-token activation scales', model=args.model,
                  calibration_record=str(calibration / 'report.json'), evaluate=args.evaluate,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                  host=dict(hostname=platform.node(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                            cuda=torch.version.cuda, transformers=transformers.__version__),
                  sm120_model=C.MODELS[args.model], evaluations={})
    save = lambda: (args.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')  # noqa: E731
    t0 = time.time()
    model, modules = load_for_evaluation(args.model, prior, C)
    report['load_seconds'] = time.time() - t0
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    batches, report['data'] = data(tok, prior, 2048)
    if args.model in PUBLISHED:
        validate_evaluation_data(report['data'], json.loads(Path(PUBLISHED[args.model]).read_text())['data'])
        report['data_validation'] = PUBLISHED[args.model]
    else:
        report['data_validation'] = 'no published record for this model; windows recorded by token hash'
    reference = {}
    for key, domain in (('wiki', 'wiki'), ('c4', 'c4_paper')):
        ref = C.REFERENCE / f'{args.model}_windows_{key}.json'
        reference[key] = (json.loads(ref.read_text())['token_sha256'] == report['data'][domain]['token_sha256']) if ref.exists() else None
    report['windows_equal_sm120_reference'] = reference
    assert all(v is not False for v in reference.values()), reference
    if args.limit_windows:
        batches = {d: b[:args.limit_windows] for d, b in batches.items()}
        report['limit_windows'] = args.limit_windows
    report['data_seconds'] = time.time() - t0 - report['load_seconds']
    save()
    device = model.get_input_embeddings().weight.device
    fq = C.FakeQuant(modules)
    for pol in policies:
        entry = report['evaluations'][pol['label']] = dict(pol)
        t1 = time.time()
        if pol['kind'] in ('bf16', 'fake'):
            entry.update(install_fake(fq, pol, args.model, modules, C))
            entry['installed_weight_sq_error'] = installed_error(fq)
            first = None
        else:
            if fq.modules:                  # the first native policy: the BF16 Linears and their CPU copies go
                fq.release()
                modules = None
                torch.cuda.empty_cache()
            entry.update(install_native(model, pol, args.model, C))
            NM.reset_counters(model)
            entry['coverage_first_forward'] = {}

            def first(domain, forwards, _entry=entry):
                # after the domain's first window: every quantized Linear ran natively once per forward so far
                cov = NM.coverage(model)
                _entry['coverage_first_forward'][domain] = {k: v for k, v in cov.items() if k != 'remaining_bf16_linears'}
                if cov['native_called'] != cov['native'] or cov['calls'] != cov['native'] * forwards:
                    raise SystemExit(f'native coverage incomplete: {cov}')
        entry['install_seconds'] = time.time() - t1
        t2 = time.time()
        entry['evaluation'] = evaluate(model, batches, device, pol['label'], first)
        entry['evaluation_seconds'] = time.time() - t2
        if pol.get('_act_stats') is not None:
            entry['activation_format'] = pol['_act_stats'].as_dict()
            entry.pop('_act_stats', None)
        if pol['kind'] == 'native':
            cov = NM.coverage(model)
            windows = sum(len(v) for v in batches.values())
            entry['coverage_total'] = dict(calls=cov['calls'], expected_calls=cov['native'] * windows,
                                           remaining_bf16_linears=cov['remaining_bf16_linears'])
            assert cov['calls'] == cov['native'] * windows, entry['coverage_total']
            entry['gemm_audit'] = P.gemm_audit(model, batches['wiki'][0], device)
            if entry['install'].get('kernel_set') is not None:
                entry['install']['kernel_set'] = next(m.kernel_set for m in NM.native_modules(model).values()).describe()
        save()
    report['status'] = 'complete'
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    report['seconds'] = time.time() - t0
    save()


if __name__ == '__main__':
    main()

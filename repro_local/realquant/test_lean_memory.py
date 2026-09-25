"""Part A unit checks for run_multiround.py --memory-mode lean, on Llama-3.1-8B's 224 matrices (GPU).

1. The ONE candidate store: for every matrix, the Triton decode of the native-format base and
   alternative equals decode_base / decode_alt of quantize/packed_candidates (the legacy fake
   store) bitwise as int16, signed zeros included (pack_candidates checks the PyTorch decode
   the same way when the store is filled).
2. The per-layer decoded weight: for the start map (all E2M1), a random mixed map (run_multiround's
   generator: CUDA seed 0, p = 0.5), the restored start map and the committed final maps of that
   unit, the lean decode equals the weight legacy apply() installs,
   torch.where(expand(map), decode_alt, decode_base), bitwise, at 256x64 and 8x64. The PyTorch
   reference path (select + decode_packed) is compared too.
3. The recompute-for-backward forward (candidate_store.lean_forward): one deterministic forward and
   backward of the whole model on two development documents, with scoring's straight-through
   activation pre-hooks, on a mixed map. Logits and the input-embedding gradient must equal the
   legacy run (resident weights, nn.Linear) bitwise. Peak GPU memory of both passes is recorded.

CUBLAS_WORKSPACE_CONFIG=:4096:8 python repro_local/realquant/test_lean_memory.py DATA_ROOT OUT_JSON MAP...
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_store import CandidateStore, lean_forward  # noqa: E402
from quantize.causal_four_over_six import quantize_rows  # noqa: E402
from quantize.packed_candidates import decode_alt, decode_base, pack  # noqa: E402
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6  # noqa: E402
from run_math_code_calibration import load_model  # noqa: E402
from run_multiround import UNITS, data_paths, expand, load_development  # noqa: E402


def same(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def negative_zeros(t):
    return int(((t == 0) & torch.signbit(t)).sum())


@torch.no_grad()
def main():
    data_root, out = Path(sys.argv[1]), Path(sys.argv[2])
    committed = [Path(p) for p in sys.argv[3:]]
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    calibration, development = data_paths('llama8b', data_root)
    prior = json.loads((calibration / 'report.json').read_text())
    model, modules = load_model(prior, False)
    model.set_attn_implementation('sdpa')
    device = model.get_input_embeddings().weight.device
    result = dict(modules=len(modules), candidates={}, maps={}, autograd={})

    # 1. the store vs the legacy fake store
    t0 = time.time()
    legacy, store = {}, CandidateStore(256, 64)
    neg = dict(base=0, alt=0)
    for n, m in modules.items():
        w = m.weight
        b = quant_nvfp4_4over6(w, 4, 16)
        a = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        p = pack(w, b, a)
        assert p is not None, n
        legacy[n] = p
        base, alt = decode_base(p), decode_alt(p)
        store.add(n, w, base, alt)                  # raises unless the PyTorch decode is bitwise
        assert same(store.decode(n, which='base'), base), n
        assert same(store.decode(n, which='alt'), alt), n
        assert same(store.reference_decode(n, which='base'), base), n
        assert same(store.reference_decode(n, which='alt'), alt), n
        neg['base'] += negative_zeros(base)
        neg['alt'] += negative_zeros(alt)
        del a, b, base, alt
    result['candidates'] = dict(bitwise_modules=len(store.cand), negative_zeros_in_base=neg['base'],
                                negative_zeros_in_alt=neg['alt'], store_gib=store.nbytes() / 2 ** 30,
                                legacy_fake_store_gib=sum(v.numel() * v.element_size() for p in legacy.values()
                                                          for v in p.values() if torch.is_tensor(v)) / 2 ** 30,
                                seconds=time.time() - t0)
    print('CANDIDATES ' + json.dumps(result['candidates']), flush=True)

    def legacy_apply(n, s, rows, cols):
        b = decode_base(legacy[n])
        return torch.where(expand(s, rows, cols, b.shape[0]), decode_alt(legacy[n]), b)

    # 2. the per-layer decoded weight vs apply() at both units
    for unit in ('256x64', '8x64'):
        rows, cols = UNITS[unit]
        store.rows, store.cols = rows, cols
        sel = {n: torch.zeros(-(-m.weight.shape[0] // rows), m.weight.shape[1] // cols, dtype=torch.bool,
                              device=device) for n, m in modules.items()}
        maps = {'start': {n: s.clone() for n, s in sel.items()}}
        mixer = torch.Generator(device=device).manual_seed(0)
        maps['mixed'] = {n: torch.rand(s.shape, generator=mixer, device=device) < 0.5 for n, s in sel.items()}
        maps['restored'] = {n: s.clone() for n, s in maps['start'].items()}
        for path in committed:
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if all(saved[n].shape == sel[n].shape for n in sel):
                maps[str(path)] = {n: saved[n].to(device) for n in sel}
        for label, chosen in maps.items():
            bad_apply, bad_ref = [], []
            for n in modules:
                got = store.decode(n, chosen[n])
                if not same(got, legacy_apply(n, chosen[n], rows, cols)):
                    bad_apply.append(n)
                if not same(got, store.reference_decode(n, chosen[n])):
                    bad_ref.append(n)
            e0m3 = sum(int(s.sum()) for s in chosen.values())
            result['maps'][f'{unit} {label}'] = dict(e0m3_units=e0m3, apply_mismatches=bad_apply,
                                                     reference_mismatches=bad_ref)
            print(f'MAP {unit} {label}: {e0m3} E0M3 units, apply mismatches {len(bad_apply)}, '
                  f'reference mismatches {len(bad_ref)}', flush=True)
            assert not bad_apply and not bad_ref

    # 3. forward + backward through the whole model: resident weights vs lean_forward
    store.rows, store.cols = UNITS['8x64']
    mixer = torch.Generator(device=device).manual_seed(1)
    chosen = {n: torch.rand(-(-m.weight.shape[0] // 8), m.weight.shape[1] // 64, generator=mixer,
                            device=device) < 0.3 for n, m in modules.items()}
    for n, m in modules.items():
        m.weight.copy_(legacy_apply(n, chosen[n], 8, 64))
    del legacy
    torch.cuda.empty_cache()
    docs, _ = load_development(development)
    ids = torch.cat([docs[0]['ids'], docs[1]['ids']]).to(device)

    def act(module, inputs):
        x = inputs[0]
        return (quantize_rows(x.detach()) + (x - x.detach()), *inputs[1:])

    handles = [m.register_forward_pre_hook(act) for m in modules.values()]
    runs = {}
    original = {n: m.forward for n, m in modules.items()}
    for mode in ('legacy', 'lean'):
        if mode == 'lean':
            for n, m in modules.items():
                m.forward = lean_forward(lambda key, n=n: store.decode(n, key[0]), lambda n=n: (chosen[n],), m.bias)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        with torch.enable_grad():
            embeds = model.get_input_embeddings()(ids).detach().requires_grad_()
            logits = model(inputs_embeds=embeds, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
            loss.backward()
        torch.cuda.synchronize()
        runs[mode] = dict(logits=logits.detach().cpu(), grad=embeds.grad.detach().cpu(), loss=float(loss),
                          peak_above_start_gib=(torch.cuda.max_memory_allocated() - base_alloc) / 2 ** 30)
        del embeds, logits, loss
    for n, m in modules.items():
        m.forward = original[n]
    for h in handles:
        h.remove()
    result['autograd'] = dict(documents=2, tokens=int(ids.numel()),
                              logits_bitwise=same(runs['legacy']['logits'], runs['lean']['logits']),
                              input_gradient_bitwise=same(runs['legacy']['grad'], runs['lean']['grad']),
                              loss_legacy=runs['legacy']['loss'], loss_lean=runs['lean']['loss'],
                              peak_above_start_gib=dict(legacy=runs['legacy']['peak_above_start_gib'],
                                                        lean=runs['lean']['peak_above_start_gib']),
                              map_e0m3_units_8x64=sum(int(s.sum()) for s in chosen.values()))
    print('AUTOGRAD ' + json.dumps(result['autograd']), flush=True)
    assert result['autograd']['logits_bitwise'] and result['autograd']['input_gradient_bitwise']
    result['passed'] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    print('PASS', flush=True)


if __name__ == '__main__':
    main()

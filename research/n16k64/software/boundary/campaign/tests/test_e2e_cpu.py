"""Spec test 12 (CPU mini end-to-end): tiny Llama -> CE/KL scores -> N16 map -> reload/verify -> install -> PPL -> schema record."""
import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from campaign import calibrate as C
from campaign import evaluate_ppl as E
from campaign import mapio as MIO
from campaign import models as MOD
from campaign import policies as P
from campaign import records as REC
from campaign import tiles as T

FINDINGS = {}


def record(k, v):
    FINDINGS[k] = v
    if os.environ.get('E2E_FINDINGS'):
        json.dump(FINDINGS, open(os.environ['E2E_FINDINGS'], 'w'), indent=1, sort_keys=True, default=str)


@pytest.fixture(scope='module')
def tiny():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=320, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval().requires_grad_(False)
    MOD.REGISTRY['tiny_test'] = dict(model_id='test/tiny-llama', revision='0' * 40, family='test', panel='test', loader='causal_lm',
                                     n8_total=None, n16_total=None)
    return model


def test_cpu_end_to_end(tiny, tmp_path, monkeypatch):
    model = tiny
    modules = MOD.scope(model, 'tiny_test')
    assert all(m.weight.shape[0] % 16 == 0 and m.weight.shape[1] % 64 == 0 for m in modules.values())
    pristine = {n: m.weight.detach().clone() for n, m in modules.items()}
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(0, 320, (1, 64), generator=g) for _ in range(8)]
    domains = ['math'] * 4 + ['code'] * 4
    teacher = []
    with torch.no_grad():
        for s in seqs:
            teacher.append(model(input_ids=s, use_cache=False).logits[:, :-1].float().reshape(-1, 320).log_softmax(-1))
    alt, stats = C.build_candidates(modules, alt_on_gpu=True)
    # record direct N16 reductions inside the production hook path
    direct = []
    real = T.directional_scores

    def spy(grad, d, tb):
        direct.append((real(grad.double(), d.double(), T.N16), T.tile_sum((grad.double() * d.double()).abs(), T.N16)))
        return real(grad, d, tb)
    monkeypatch.setattr(T, 'directional_scores', spy)
    sc = C.run_scoring(model, modules, seqs, domains, lambda i: teacher[i], alt, raw='full', sample_fraction=0.5, subset_moments=True)
    monkeypatch.setattr(T, 'directional_scores', real)
    names = sc['names']
    # direct-vs-aggregated identity on real gradients: calls arrive as (CE: reversed modules, KL: reversed modules) per sequence
    rev = list(reversed(names))
    per_seq = len(rev) * 2
    worst = 0.0
    for i in range(len(seqs)):
        for j, n in enumerate(rev):
            d_ce, abs_sum = direct[i * per_seq + j]
            agg = T.aggregate_n8_to_n16(sc['raw_full'][n]['ce'][i].double().reshape(1, -1), *sc['shapes'][n]).reshape(-1)
            worst = max(worst, float(((d_ce - agg).abs() / abs_sum.clamp_min(1e-30)).max()))
    # float32 tile sums of 512 products can round by ~512*eps32 relative to the absolute-term mass
    record('hook_direct_float64_vs_aggregated_float32_children_max_abs_error_over_abs_term_mass', worst)
    assert worst < 512 * 1.2e-7
    # moments from stored raw equal the streamed moments
    for n in names:
        o, k = sc['shapes'][n]
        raw_ce16 = T.aggregate_n8_to_n16(sc['raw_full'][n]['ce'].double(), o, k)
        raw_kl16 = T.aggregate_n8_to_n16(sc['raw_full'][n]['kl'].double(), o, k)
        m = T.Moments.from_raw(raw_ce16, raw_kl16)
        assert torch.equal(m.ce_sum, sc['full16'][n].ce_sum) and torch.equal(m.kl_sq, sc['full16'][n].kl_sq)
    # map election and canonical write/reload
    masks = {n: T.elect(sc['full16'][n], 0, 'ce_kl').reshape(sc['shapes'][n][0] // 16, sc['shapes'][n][1] // 64) for n in names}
    ident = dict(model_id='test/tiny-llama', revision='0' * 40, tokenizer_revision='0' * 40, model_class=type(model).__name__)
    header = MIO.build_header(protocol_id='aligned-primary', policy=dict(name='n16_k0_smoke', rule='ce_kl', k=0), model=ident,
                              type_block=(16, 64), masks=masks, weight_shapes={n: sc['shapes'][n] for n in names},
                              source_manifest_sha256='smoke', calibration_manifest_sha256='smoke')
    digest, path = MIO.write_map(tmp_path / 'tiny.mixfp4map', header, masks)
    # install via the production installer (restores pristine weights first)
    with torch.no_grad():
        for n, m in modules.items():
            m.weight.copy_(pristine[n])
    inst = P.Installer('tiny_test', model, modules, cache='cpu', protocol_id='aligned-primary', campaign_root=None)
    plan = [dict(name='four_over_six', kind='four_over_six'), dict(name='n16_k0_smoke', kind='map', map_path=path, map_sha256=digest,
             map_policy='n16_k0_smoke', type_block=[16, 64]), dict(name='bf16', kind='bf16')]
    results = {}
    for pol in plan:
        info = inst.install(pol)
        nll = []
        with torch.no_grad():
            for s in seqs:
                st = E.token_stats(model(input_ids=s, use_cache=False).logits[0, :-1], s[0, 1:])
                nll.append(float(st['nll'].double().mean()))
        results[pol['name']] = dict(ppl=math.exp(sum(nll) / len(nll)), checksum=info['installed_weight_sha256'],
                                    selected=info.get('selected_tiles'))
    inst.remove()
    with pytest.raises(MIO.MapVerificationError):
        inst.install(dict(plan[1], map_policy='n16_k3'))
    with pytest.raises(MIO.MapVerificationError):
        inst.install(dict(plan[1], type_block=[8, 64]))
    assert all(math.isfinite(v['ppl']) for v in results.values())
    assert results['n16_k0_smoke']['selected'] == header['totals']['selected_tiles']
    record('e2e_policies', {k: dict(finite=math.isfinite(v['ppl']), selected=v['selected']) for k, v in results.items()})
    # RESULT_SCHEMA-conforming record from a synthetic CPU run directory
    run = tmp_path / 'runs' / 'V14_cpu_mini_e2e'
    (run / 'preflight').mkdir(parents=True)
    (run / 'launch_record.json').write_text(json.dumps(dict(run_id='V14_cpu_mini_e2e', matrix_id='V14', protocol_id='correctness',
        requested=dict(gpus=0), status='complete', source_manifest_sha256='smoke', host='cpu', wall_seconds=1.0, gpu_hours=0.0)))
    (run / 'job_status.json').write_text(json.dumps(dict(status='complete', preflight_records=[])))
    (run / 'job_result.json').write_text(json.dumps(dict(
        protocol_id='correctness', protocol_freeze_sha256='smoke',
        source=dict(model_id='test/tiny-llama', model_revision='0' * 40, tokenizer_revision='0' * 40, model_class='LlamaForCausalLM',
                    module_manifest_sha256=MOD.module_manifest(modules)[0], source_manifest_sha256='smoke'),
        environment=dict(hostname='cpu', container_or_lock_sha256='smoke', driver=None, cuda=None, torch=torch.__version__,
                         transformers='x', datasets='x', lm_eval=None, attention_backend='sdpa', activation_quantizer='four_over_six_rows'),
        data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[dict(name=p['name'], weight_format=p['kind'], activation_format='rows', scale_block=16, type_block=p.get('type_block'),
                       map_path=p.get('map_path'), map_sha256=p.get('map_sha256'), selected_tiles=results[p['name']]['selected'],
                       total_tiles=None, map_reloaded_for_evaluation=p['kind'] == 'map') for p in plan],
        results=dict(raw_outputs=[], summary={k: v['ppl'] for k, v in results.items()}, uncertainty={}, attempted_endpoints=['ppl'],
                     missing_endpoints=[]), logs=[], failures=[])))
    rec, errors = REC.assemble_run_record(run)
    record('schema_errors', errors)
    assert not errors, errors

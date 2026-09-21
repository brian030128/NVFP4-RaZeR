"""Build, hash and immutably write PROTOCOL_FREEZE.json (V02 / V60).

Must run before any confirmatory-model quantized quality result exists. It refuses to run if any
confirmatory calibration/evaluation run directory already exists.
"""
import hashlib
import json
import os
import time
from pathlib import Path

from campaign import models as MOD

CR = Path(os.environ['CAMPAIGN_ROOT'])


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    runs = CR / 'runs'
    conf = [k for k, v in MOD.REGISTRY.items() if v['panel'] == 'confirmatory']
    offending = [d.name for d in runs.iterdir() if d.is_dir() and any(f'_{k}_' in d.name or d.name.endswith(f'_{k}') for k in conf)
                 and not d.name.startswith('V14_smoke_')]
    if offending:
        raise SystemExit(f'confirmatory runs exist before freeze: {offending}')
    smoke = {}
    for k in MOD.REGISTRY:
        cands = sorted(runs.glob(f'V14_smoke_{k}_*'))
        ok = [c for c in cands if (c / 'launch_record.json').exists() and json.loads((c / 'launch_record.json').read_text()).get('status') == 'complete']
        if not ok:
            raise SystemExit(f'no completed compatibility smoke for {k}')
        rep = json.loads((ok[-1] / 'smoke' / 'smoke_report.json').read_text())
        smoke[k] = dict(run=ok[-1].name, calibration_manifest_sha256=rep['calibration_manifest_sha256'],
                        calibration_manifest_file_sha256=sha_file(ok[-1] / 'smoke' / 'calibration_manifest_seed0.json'),
                        module_manifest_sha256=rep['checks']['module_manifest_sha256'], modules=rep['checks']['modules'],
                        n8_total=rep['checks']['n8_total'], n16_total=rep['checks']['n16_total'],
                        windows=rep['checks']['windows'], max_position_embeddings=rep.get('max_position_embeddings'),
                        quality_values_recorded=rep.get('quality_values_recorded'))
    tasks = json.loads((CR / 'provenance' / 'LM_EVAL_TASK_FETCH_offline.json').read_text())
    online = json.loads((CR / 'provenance' / 'LM_EVAL_TASK_FETCH_online.json').read_text())
    task_digests = {t: dict(docs=v['docs'], docs_sha256=v['docs_sha256'], dataset_path=v['dataset_path'], dataset_name=v['dataset_name'],
                            split=v['split'], version=v['version'], hub_revision_at_fetch=online['tasks'].get(t, {}).get('hub_revision'))
                    for t, v in tasks['tasks'].items()}
    offline_equal_online = all(online['tasks'][t]['docs_sha256'] == v['docs_sha256'] for t, v in tasks['tasks'].items())
    freeze = {
        'freeze_version': 1,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'campaign_root': str(CR),
        'input_zip_sha256': (CR / 'provenance' / 'INPUT_ZIP.sha256').read_text().split()[0],
        'owner_decisions': json.loads((CR / 'provenance' / '01_owner_decisions.json').read_text()),
        'quality_results_opened_before_freeze': dict(
            confirmatory_models='none (compatibility/memory/cost smoke tests only; no quantized quality value stored or printed)',
            development_models=('archived historical reports only, plus two cost-planning timing runs on Qwen3-4B with plain FourOverSix (no MixFP4 map): '
                                'V14_lmeval_timing_qwen4b (40 examples per task) and V14_gsm8k_timing_qwen4b (48 examples); their metrics were not inspected; '
                                'no aligned-protocol MixFP4 map or MixFP4 quality result exists before this freeze')),
        'scope': dict(
            claim='fake-quantized (BF16 dequantized matmul) algorithmic results for tile-aligned FourOverSix/E0M3 type selection under N8K64/N16K64 granularity',
            excluded=['native FP4/E0M3 Tensor Core or kernels', 'SM120', 'kernel throughput / latency / speedup', 'N8 ~13% and N16 ~1.5% overhead re-measurement (external team estimates only)',
                      'area / power / RTL / hardware co-design', 'KV-cache quantization', 'lm_head / embedding / norm quantization'],
            hardware='RTX A6000 (primary) and RTX 6000 Ada (portability only); at most 3 homogeneous GPUs per run; campaign-local lease + Docker device isolation (D01)'),
        'panels': {
            'development': {k: dict(model_id=MOD.REGISTRY[k]['model_id'], revision=MOD.REGISTRY[k]['revision'], tokenizer_revision=MOD.REGISTRY[k]['revision'],
                                    family=MOD.REGISTRY[k]['family'], role='development / transfer evidence, not independent confirmation') for k in ('llama8b', 'qwen4b', 'qwen27b')},
            'confirmatory': {k: dict(model_id=MOD.REGISTRY[k]['model_id'], revision=MOD.REGISTRY[k]['revision'], tokenizer_revision=MOD.REGISTRY[k]['revision'],
                                     family=MOD.REGISTRY[k]['family']) for k in ('mistral7b', 'phi4', 'olmo2_13b')},
            'replacements': [dict(original='google/gemma-2-9b', replacement='allenai/OLMo-2-1124-13B', reason='HTTP 403: gated license not accepted for the available token',
                                  decided_before_quality=True, decision='provenance/01_owner_decisions.json#D02')],
            'replacement_rule_if_compatibility_fails_before_first_quality_result': ['ibm-granite/granite-3.3-8b-base@cfb7adb44a974653cbb2ff883653971c54dba578',
                                                                                  'tiiuae/Falcon3-10B-Base@34bb99a889fe0426412da3dd2b46e6f64c8fd003',
                                                                                  '01-ai/Yi-1.5-9B@80d5471b1eae28beae33e06eadbd4b48e74d4ce1'],
            'after_first_confirmatory_quality_result': 'no replacement; failures are reported as missing endpoints',
            'representative_confirmatory_model': 'mistral7b',
            'portability_and_determinism_model': 'qwen4b',
            'compatibility_smoke': smoke,
        },
        'protocols': {
            'historical': dict(purpose='provenance and anchors only', calibration='archived run_math_code_calibration: eager attention, causal per-token FourOverSix activations with STE, CE/KL/sampled-label backwards',
                               evaluation='archived run_kse_paper: SDPA, tensor-wide FourOverSix activation factor, WikiText use_cache=True, C4 use_cache=False',
                               environments=dict(llama8b='hist (transformers 4.57.3)', qwen4b='hist (transformers 4.57.3)', qwen27b='main (transformers 5.16.1)'),
                               deviations='campaign/historical.py D-H1..D-H5'),
            'aligned-primary': dict(
                calibration=dict(teacher='pristine BF16, no activation quantization', student='FourOverSix weights on every scoped Linear; causal per-token FourOverSix activations (quantize_rows) with identity STE',
                                 attention='sdpa (PyTorch default kernel dispatch) in calibration, PPL, accuracy and generation', dtype='bfloat16', tf32=False,
                                 objectives=['mean next-token CE', 'batchmean KL(teacher || student)'], score='per sequence, per N8 tile float32 <dLoss/dW, E0M3 - FourOverSix>',
                                 n16='float64 per-sequence sum of vertically adjacent N8 children, then float64 moments; child SEs never combined; masks never OR/AND-ed'),
                evaluation=dict(bf16='pristine weights, no activation quantization', nvfp4='quant_nvfp4 weights + per-token NVFP4 activations (nvfp4_rows)',
                                w4a4_mixfp4_family='FourOverSix / E0M3 / exact maps with causal per-token FourOverSix activations', kv_cache='unquantized', lm_head='unquantized',
                                map_installation='maps re-read from disk; SHA-256, model/tokenizer revision, module names/shapes, type block, protocol id, policy name and archived source manifest verified'),
                environment=dict(lock='env/main.lock.txt', lock_sha256=sha_file(CR / 'env' / 'main.lock.txt'), image='ubuntu@sha256:829f6df217bcbae2b371026e81711d1a787c61b2967ad09d015063663ebafbf7',
                                 torch='2.9.0+cu128', transformers='5.16.1', datasets='4.8.5', lm_eval='0.4.11')),
        },
        'primary_policy': dict(name='n16_k3', weight_baseline='FourOverSix E2M1 (quant_nvfp4_4over6)', alternative='E0M3 signed uniform alpha=1',
                               scale_block=16, type_block=[16, 64], k=3, rule='elect tile iff max(mean_CE + 3 SE_CE, mean_KL + 3 SE_KL) < 0', count_cap=None,
                               calibration='seed0: 64 OpenWebMath + 64 CodeParrot sequences, 512 tokens'),
        'policies': dict(
            primary_evaluation=['bf16', 'nvfp4', 'four_over_six', 'all_e0m3', 'n8_k3', 'n16_k3'],
            accuracy=['bf16', 'four_over_six', 'n8_k3', 'n16_k3'],
            secondary_sensitivity=['n16_k2', 'n16_k4', 'n16_k5', 'n16_k6', 'n8_k2', 'n8_k4', 'n8_k5', 'n8_k6'],
            objective_ablation=['n16_k3_ce_only', 'n16_k3_kl_only', 'n16_k3_mean_only (k irrelevant: both means < 0)', 'n16_k3 (CE intersect KL)'],
            selector_controls=['n16_random_s0..s4 (per-layer count-matched to n16_k3, Generator(1000+s))', 'n16_weight_mse (per-layer count-matched)',
                               'n16_magnitude (per-layer count-matched tile sum w^2)', 'n16_change_norm (extra heuristic)', 'n16_density_matched_n8k3 (global ascending U3, round(N8k3 tiles/2))',
                               'all-false == four_over_six checksum', 'all-true == all_e0m3 checksum'],
            baselines=['razer_wonly_shared_act', 'razer_native_rows', 'nover6_wonly_shared_act', 'nover6_native_rows'],
            baselines_note='released RaZeR W4A4 = nvfp4_razer_e3m3 (outlier 8) weights + nvfp4_razer_e4m3 activations; released e4m3 selection expression preserved, not corrected'),
        'calibration': dict(
            seed0=dict(development='archived manifests (document SHA-256 + token offsets), token hashes verified', confirmatory='archived builder rule run_pooled_scale.shared_data applied to each tokenizer',
                       manifest_sha256={k: v['calibration_manifest_sha256'] for k, v in smoke.items()}),
            independent_draws=dict(ids=['draw1', 'draw2', 'draw3', 'draw4'], rule='campaign/data.keyed_draw: sha256(drawN:doc) order over the same pinned OpenWebMath/CodeParrot files, excluding seed0 and all lower draws, >=512 tokens, keyed offset',
                                   models=['llama8b', 'qwen4b', 'mistral7b'], resolutions=['n8_k3', 'n16_k3']),
            size=dict(models=['qwen4b', 'mistral7b'], settings=['16+16 (math16+code16 prefixes of seed0)', '32+32', '64+64 (primary)']),
            domain=dict(models=['qwen4b', 'mistral7b'], settings=['math64 only', 'code64 only', 'balanced 32+32', 'held-out mixture: 32 arXiv + 32 GovReport test documents (keyed draw heldout)']),
            storage=dict(raw_sample='per module keyed-random 1% of N16 parents (cap 4096, floor 32) with both N8 children, all sequences', full_raw='qwen4b seed0 if /home free >= 80 GiB at launch',
                         moments='float64 sums/squares/cross for full set; math/code prefix moments for size/domain models')),
        'evaluation': dict(
            ppl=dict(wikitext='Salesforce/wikitext wikitext-2-raw-v1 test @b08601e0, all full non-overlapping 2048-token windows of the joined text',
                     c4='allenai/c4 en/c4-validation.00000-of-00008 @1588ec45, 256 documents Random(0) with replacement, 2048-token crops',
                     aggregation='PPL = exp(sum token NLL / tokens); per-window and per-token arrays saved', teacher_diagnostics='KL(teacher||student), top-1 agreement, entropy, confidence, token accuracy, ECE (10 bins)',
                     teacher_mode=dict(instance=['qwen4b', 'llama8b', 'mistral7b'], cache_first_32_windows_per_corpus=['phi4', 'olmo2_13b'], cache_first_16_windows_per_corpus=['qwen27b'],
                                       note='NLL, entropy, confidence and token accuracy on every window; KL/top-1 agreement on all windows (instance) or the predeclared cached subset'),
                     clusters=dict(c4='document SHA-256', wikitext='article containing the first token of the window; sensitivity: contiguous blocks of 5 windows')),
            long_context=dict(corpus='emozilla/pg19-test @c5e39bf3 books in file order', windows={'4096': 64, '8192': 32}, clusters='book', models=['llama8b', 'qwen4b', 'mistral7b'],
                              policies=['bf16', 'four_over_six', 'n8_k3', 'n16_k3']),
            accuracy=dict(harness='lm-eval 0.4.11 HFLM on the in-memory model, batch_size 16, log_samples, seeds (0,1234,1234,1234)', tasks={'arc_easy': 'acc_norm', 'arc_challenge': 'acc_norm',
                          'hellaswag': 'acc_norm', 'openbookqa': 'acc_norm', 'boolq': 'acc', 'winogrande': 'acc', 'piqa': 'acc_norm', 'mmlu': 'acc (57 subjects, lm-eval group aggregate)'},
                          num_fewshot=0, representative_suite=['arc_challenge', 'piqa', 'winogrande', 'boolq'], task_documents=task_digests, offline_docs_equal_online=offline_equal_online),
            generation=dict(task='lm-eval gsm8k v3.0', num_fewshot=5, decoding='task defaults: greedy (do_sample false), until [Question:, </s>, <|im_end|>], max_gen_toks 256',
                            primary_metric='exact_match,flexible-extract', secondary_metric='exact_match,strict-match', models=['qwen4b', 'llama8b', 'mistral7b'],
                            policies=['bf16', 'four_over_six', 'n8_k3', 'n16_k3'])),
        'endpoints': dict(
            primary=[f'{m}:{d}:dlogppl(n16_k3 - four_over_six)' for m in ('mistral7b', 'phi4', 'olmo2_13b') for d in ('wiki', 'c4')],
            primary_estimator='token-weighted mean NLL difference with 95% paired cluster-bootstrap percentile CI (B=10000, seed 20260911)',
            secondary=['per-task and 8-task macro accuracy difference (n16_k3 - four_over_six), paired example bootstrap', 'gsm8k exact_match difference',
                       'retained fraction R = sum(dlogppl n16_k3) / sum(dlogppl n8_k3) over confirmatory model x dataset (only if the N8 sum < 0)',
                       'same endpoints on the development panel (labelled development)'],
            exploratory='everything else'),
        'success_criteria': dict(
            sesoi=dict(ppl_noninferiority_margin_dlogppl=0.004987541511038968, ppl_margin_meaning='log(1.005): +0.5% perplexity', accuracy_macro_margin_pp=-0.5,
                       note='campaign-defined margins (not team-agreed engineering tolerances); fixed before any aligned or confirmatory quality result'),
            minimum_pass='all 6 primary endpoints have upper 95% CI < margin AND each confirmatory model: 8-task macro accuracy difference lower 95% CI > -0.5 pp AND fewer than 3 tasks with a paired-bootstrap CI entirely below 0',
            strong_pass_quality_only='minimum_pass AND R >= 0.5 AND pooled N8 k3 dlogppl < 0 (overhead component out of scope)',
            investigate='pooled mean of primary endpoints < 0 but minimum_pass fails',
            stop_redirect='R <= 0 on at least 2 of 3 confirmatory models, or pooled primary upper 95% CI >= margin',
            multiplicity='Holm across the 6 primary one-sided non-inferiority tests (H0: dlogppl >= margin), family alpha 0.05; unadjusted CIs also reported',
            k_rule='k=3 is primary; k in {2,4,5,6} is sensitivity only and may not replace k=3',
            qwen_smoothing_rule=dict(
                applies_to=['qwen4b n8_k3 vs four_over_six', 'qwen4b n16_k3 vs four_over_six'],
                capability_gain='mean NLL difference CI < 0 AND (8-task macro accuracy difference lower CI > 0 OR GSM8K difference lower CI > 0) AND top-1 agreement with BF16 does not decrease (CI upper >= 0)',
                confidence_smoothing='mean NLL difference CI < 0 AND macro accuracy and GSM8K differences have CIs including 0 or below 0 AND mean predictive entropy increases (CI > 0)',
                otherwise='mixed / inconclusive; PPL below BF16 is never reported as capability exceeding BF16',
                ece='10 equal-width confidence bins, top-1 token confidence vs token accuracy')),
        'statistics': dict(
            bootstrap=dict(B_primary=10000, B_exploratory=2000, seed=20260911, type='paired percentile over clusters'),
            selection_multiplicity=dict(iut='per tile p_IUT = max(p_CE, p_KL), one-sided, normal and t(n-1)', fdr=['BH q=0.05,0.10', 'BY q=0.05,0.10 (arbitrary dependence)'],
                                        null_bounds=['m*Phi(-3) (IUT, any dependence)', 'sum Phi2(-3,-3; rho_hat_tile)'],
                                        sign_flip=dict(R=1000, unit='calibration sequence (same flip for CE, KL and every sampled tile)', sample='stored stratified tile sample')),
            calibration_vs_evaluation_variance='between-draw variance of endpoints over 5 draws vs mean within-draw bootstrap variance',
            ce_kl_correlation='per-tile Pearson over sequences (distribution) and across-tile correlation of t statistics'),
        'first_order_fidelity': dict(models=['llama8b', 'qwen4b', 'mistral7b'], resolutions=['n8', 'n16'],
                                     strata=dict(selected_k3=40, near_threshold=40, rejected=40, random=40), layer_strata='early/middle/late thirds reported', sampling_seed=20260912,
                                     near_threshold_definition='0 <= U3 < 10th percentile of positive U3', rejected_definition='U3 > median of positive U3',
                                     batched_prefix_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 'all selected'], actual='float64 mean over the 128 seed0 calibration sequences of CE/KL with only those tiles switched, W4A4 causal eval mode (no STE)',
                                     metrics=['Pearson', 'Spearman', 'sign precision/recall', 'binned calibration', 'false-positive rate among selected', 'interaction error vs sum of singles and vs predicted']),
        'numerical_tolerances': dict(
            historical=dict(tierA='bitwise equality', tierB=dict(bf16_fit_nll_max_abs=2e-3, bf16_fit_nll_mean_abs=5e-4, w4a4_fit_ce_max_abs=5e-2, w4a4_fit_ce_mean_abs=1e-2,
                                                                 fit_kl_max_abs=2e-2, fit_kl_mean_abs=5e-3, ppl_rel_bf16=1e-3, ppl_rel_w4a4=5e-3, k3_count_rel=0.05, fixed256_jaccard=0.8),
                            tierC='archived paired effect sign reproduced and |regenerated - archived dlogppl| <= max(0.25*|archived|, 0.002)',
                            required_anchors=['four_over_six PPL (3 development models)', 'archived fixed-256 map PPL', 'N8 k3 PPL', 'N8 k2 score identity', 'fixed-256 map reconstruction'],
                            on_failure='stop runs that would be compared with archived numbers for that model; investigate; never widen; aligned protocol is not compared with archived numbers and proceeds',
                            basis='cross-architecture (archived H100/H200 vs A6000); W4A4 fake quantization amplifies kernel noise (V14 causality diagnostic)'),
            cross_gpu=dict(map='tierA identical map SHA; tierB Jaccard >= 0.9 with every flipped tile |U3| reported', ppl_rel_w4a4=5e-3, paired_effect='sign equal and |difference| <= max(0.25*|A6000 effect|, 0.002)'),
            determinism=dict(tierA='identical score-stream SHA-256 and map SHA-256 on repeat', tierB='map Jaccard >= 0.99; all threshold flips reported')),
        'gpu_policy': dict(max_gpus=3, homogeneous=True, preflight='before/phase/after; fail closed; co-tenancy invalidates and reschedules', scheduler='campaign_local_lease + docker device cgroup (D01)'),
        'source_state_at_freeze': None,
    }
    from campaign.launcher import source_manifest
    freeze['source_state_at_freeze'] = source_manifest()[0]
    out = CR / 'freeze'
    out.mkdir(exist_ok=True)
    path = out / 'PROTOCOL_FREEZE.json'
    if path.exists():
        raise SystemExit('PROTOCOL_FREEZE.json already exists (immutable)')
    data = (json.dumps(freeze, indent=1, sort_keys=True) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    digest = hashlib.sha256(data).hexdigest()
    (out / 'PROTOCOL_FREEZE.sha256').write_text(f'{digest}  PROTOCOL_FREEZE.json\n')
    os.chmod(out / 'PROTOCOL_FREEZE.sha256', 0o444)
    print(json.dumps(dict(path=str(path), sha256=digest, created_utc=freeze['created_utc'])))


if __name__ == '__main__':
    main()

"""Create the detached-hash protocol lock for the PPL-improvement extension.

This job is deliberately CPU-only and refuses to overwrite an existing lock.  It may
run only after the read-only parent and prospectively reserved held-out input audits
have passed, and before any extension quality endpoint has been evaluated.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path

from campaign import runtime


CR = Path(os.environ["CAMPAIGN_ROOT"])
PR = Path(os.environ["CAMPAIGN_PARENT_ROOT"])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def file_identity(path: Path, logical_path: str | None = None) -> dict:
    return {"path": logical_path or str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def assert_preconditions() -> tuple[dict, dict, dict]:
    input_path = CR / "runs/P00_input_audit_attempt3/input_audit/READ_ONLY_INPUT_AUDIT.json"
    heldout_path = CR / "runs/P00_heldout_input_audit_attempt2/heldout_input_audit/HELDOUT_INPUT_AUDIT.json"
    reservation_path = CR / "provenance/HELDOUT_PANEL_RESERVATION.json"
    input_audit, heldout_audit, reservation = map(read_json, (input_path, heldout_path, reservation_path))
    if not input_audit["g0_input_audit"]["passed"]:
        raise RuntimeError("authoritative read-only input audit did not pass")
    if not heldout_audit["all_compatibility_passed"] or heldout_audit["model_count"] < 2:
        raise RuntimeError("prospectively reserved held-out compatibility audit did not pass")
    if sha(reservation_path) != (CR / "provenance/HELDOUT_PANEL_RESERVATION.sha256").read_text().split()[0]:
        raise RuntimeError("held-out reservation sidecar mismatch")
    if heldout_audit["reservation_sha256"] != sha(reservation_path):
        raise RuntimeError("held-out audit did not use the current reservation")

    # A protocol lock after extension quality has been observed would be retrospective.
    allowed = {"P00_INPUT_AUDIT", "P00_PROTOCOL_LOCK"}
    opened = []
    for record_path in sorted((CR / "runs").glob("*/run_record.json")):
        record = read_json(record_path)
        matrix_id = record.get("matrix_id")
        evaluation = record.get("evaluation") or {}
        policies = record.get("policies") or []
        raw = evaluation.get("raw_ppl")
        if matrix_id not in allowed or raw is not None or policies:
            opened.append({"run_id": record.get("run_id"), "matrix_id": matrix_id,
                           "raw_ppl_present": raw is not None, "policy_count": len(policies)})
    if opened:
        raise RuntimeError(f"extension quality artifacts exist before protocol lock: {opened}")
    if (CR / "freeze/PROTOCOL_EXTENSION.json").exists() or (CR / "freeze/PROTOCOL_EXTENSION.sha256").exists():
        raise RuntimeError("protocol lock already exists; overwrite is forbidden")
    return input_audit, heldout_audit, reservation


def main() -> None:
    input_audit, heldout_audit, reservation = assert_preconditions()
    locked_at = utc()
    handoff = CR / "handoff/agent_handoff"
    matrix_path = handoff / "EXPERIMENT_MATRIX.csv"
    matrix_rows = list(csv.DictReader(matrix_path.open()))
    launch = read_json(runtime.run_dir / "launch_record.json")
    parent_freeze = PR / "freeze/PROTOCOL_FREEZE.json"
    parent_validation = PR / "runs/V83_validate_artifacts_attempt7/artifact_validation/ARTIFACT_VALIDATION.json"
    parent_manifest = PR / "runs/V83_validate_artifacts_attempt7/artifact_validation/ARTIFACT_MANIFEST.sha256"
    parent_final = PR / "runs/V90_final_reports_attempt7/final"
    reservation_sha = sha(CR / "provenance/HELDOUT_PANEL_RESERVATION.json")
    practical = math.log(1.005)

    protocol = {
        "schema_version": "1.1",
        "campaign_name": CR.name,
        "status": "LOCKED",
        "locked_at_utc": locked_at,
        "relationship_to_prior_campaign": "post_hoc_robustness_and_method_development_extension",
        "claim_label_required": "k=2 was selected after the completed parent sweep and is post-hoc; only the prospectively reserved held-out validation after winner freeze may be called held-out",
        "detached_hash_rule": {
            "authoritative_hash_file": "freeze/PROTOCOL_EXTENSION.sha256",
            "rule": "SHA-256 is over the exact bytes of freeze/PROTOCOL_EXTENSION.json; the digest is detached to avoid a self-referential field",
            "overwrite": "forbidden; any scientifically necessary change is a hash-chained amendment and cannot retroactively change an opened endpoint",
        },
        "parent_campaign": {
            "logical_path": "research_runs/mixfp4_n16k64_full_validation_20260911T065444Z",
            "protocol_freeze_sha256": sha(parent_freeze),
            "authoritative_artifact_validation_attempt": "V83_validate_artifacts_attempt7",
            "artifact_manifest_sha256": sha(parent_manifest),
            "artifact_manifest_entries": 78705,
            "run_directories": 338,
            "current_runs_with_problems": 0,
            "matrix_coverage": {"complete": 39, "partial": 0, "missing": 0},
            "authoritative_final_reports_attempt": "V90_final_reports_attempt7",
            "identities": {
                "artifact_validation": file_identity(parent_validation, "parent:runs/V83_validate_artifacts_attempt7/artifact_validation/ARTIFACT_VALIDATION.json"),
                "deliverable_index": file_identity(parent_final / "deliverable_index.json", "parent:runs/V90_final_reports_attempt7/final/deliverable_index.json"),
                "risk_audit": file_identity(parent_final / "FINAL_SUBMISSION_RISK_AUDIT.md", "parent:runs/V90_final_reports_attempt7/final/FINAL_SUBMISSION_RISK_AUDIT.md"),
                "decision": file_identity(parent_final / "N16_DECISION.md", "parent:runs/V90_final_reports_attempt7/final/N16_DECISION.md"),
                "statistics": file_identity(parent_final / "STATISTICAL_VALIDITY_REPORT.md", "parent:runs/V90_final_reports_attempt7/final/STATISTICAL_VALIDITY_REPORT.md"),
                "reproduction": file_identity(parent_final / "REPRODUCTION_REPORT.md", "parent:runs/V90_final_reports_attempt7/final/REPRODUCTION_REPORT.md"),
            },
            "preservation": "read-only mount; all negative, OOM, failed, superseded, and invalid/co-tenancy attempts remain in the parent and are included in final disclosure",
        },
        "input_locks": {
            "agent_handoff_zip_sha256": "810483fcc621b4801081a8afbb0d12ebb9a0d317110bc91984b76b472d188e4a",
            "original_implementation_handoff_sha256": "b4ba1a4b1af25dfa1dfc5e07429760af7ae0974918c5b43a8a12670bd48c2ce4",
            "parent_input_fetch_manifest_sha256": "151115d748d6841bb166f3d30702161794a9c22463c45ad6c9e5e2871600cd72",
            "read_only_input_audit": file_identity(CR / "runs/P00_input_audit_attempt3/input_audit/READ_ONLY_INPUT_AUDIT.json", "runs/P00_input_audit_attempt3/input_audit/READ_ONLY_INPUT_AUDIT.json"),
            "read_only_input_audit_passed": input_audit["g0_input_audit"]["passed"],
            "heldout_reservation_sha256": reservation_sha,
            "heldout_input_audit": file_identity(CR / "runs/P00_heldout_input_audit_attempt2/heldout_input_audit/HELDOUT_INPUT_AUDIT.json", "runs/P00_heldout_input_audit_attempt2/heldout_input_audit/HELDOUT_INPUT_AUDIT.json"),
            "heldout_compatibility_passed": heldout_audit["all_compatibility_passed"],
            "experiment_matrix_sha256": sha(matrix_path),
            "experiment_matrix_rows": len(matrix_rows),
            "result_schema_sha256": sha(handoff / "RESULT_SCHEMA.json"),
            "fair_comparison_protocol_sha256": sha(handoff / "FAIR_COMPARISON_PROTOCOL.md"),
            "statistical_analysis_plan_sha256": sha(handoff / "STATISTICAL_ANALYSIS_PLAN.md"),
            "gpu_resource_policy_sha256": sha(handoff / "GPU_RESOURCE_POLICY.md"),
            "claim_and_stopping_gates_sha256": sha(handoff / "CLAIM_AND_STOPPING_GATES.md"),
            "paper_risk_register_sha256": sha(handoff / "PAPER_RISK_REGISTER.md"),
        },
        "source_state_at_lock": {
            "source_manifest_sha256": launch["source_manifest_sha256"],
            "exact_parent_identical_anchor_files": {
                "campaign/evaluate_ppl.py": sha(CR / "source/NVFP4-RaZeR-main/campaign/evaluate_ppl.py"),
                "campaign/quant.py": sha(CR / "source/NVFP4-RaZeR-main/campaign/quant.py"),
                "campaign/data.py": sha(CR / "source/NVFP4-RaZeR-main/campaign/data.py"),
            },
            "analysis_fixes": {
                "campaign/stats.py": sha(CR / "source/NVFP4-RaZeR-main/campaign/stats.py"),
                "finite_monte_carlo_pvalue": "(extreme_count+1)/(B+1)",
                "accuracy_rng": "stable named NumPy SeedSequence task streams; Python hash() forbidden",
            },
            "per_run_identity": "every attempt records its own immutable source manifest; later implementation work cannot alter this lock or prior run manifests",
        },
        "research_roles": {
            "development_models": ["llama8b", "qwen4b", "mistral7b"],
            "breadth_replication_models": ["qwen27b", "phi4", "olmo2_13b"],
            "new_held_out_models": [m["key"] for m in reservation["models"]],
            "new_held_out_exact_revisions": {m["key"]: {"repository": m["repository"], "revision": m["revision"], "family": m["family"]} for m in reservation["models"]},
            "heldout_no_swap_rule": reservation["no_swap_rule"],
            "qwen_family_reporting": "qwen4b is a known large-effect development outlier; always report non-Qwen aggregates and never treat Qwen models as exchangeable independent families",
        },
        "scope": {
            "allowed": "software BF16-dequantized fake-quant quality experiments for W4A4 methods on A6000 and RTX 6000 Ada",
            "out_of_scope": [
                "native FP4 or E0M3 Tensor Core execution", "N8 approximately 13 percent overhead",
                "N16 approximately 1.5 percent overhead", "kernel speedup", "latency or throughput claims",
                "area", "power", "SM120-specific claims",
            ],
            "forbidden_claim_rule": "no fake-quant timing or A6000/Ada result may be presented as native execution or deployment overhead evidence",
        },
        "primary_k2_rule": {
            "tile_shape": [16, 64], "k": 2.0,
            "criterion": "max(mean_CE + k*SE_CE, mean_KL + k*SE_KL) < 0",
            "n16_construction": "sum the two adjacent N8-row per-sequence CE and KL scores first, then recompute mean, sample SD, SE, and the election; never OR/AND N8 masks",
            "role": "post-hoc robustness extension",
        },
        "quantization_semantics": {
            "shared_w4a4_arms": ["four_over_six", "all_e0m3", "n8", "n16"],
            "activation": "same causal per-token FourOverSix row quantizer unless a separately labeled activation-ablation arm is used",
            "weights": "same scale and module scope except selected E2M1/E0M3 tile format",
            "scope": "all text-model torch.nn.Linear weights except lm_head; embeddings, norms, KV cache and lm_head excluded",
            "fake_quant_path": "BF16-dequantized fake-quant matmul",
            "comparators": ["bf16", "nvfp4", "four_over_six", "all_e0m3", "n8_k3", "n16_k3", "n8_k2", "n16_k2", "razer_wonly_shared_act", "razer_native_rows", "nover6_wonly_shared_act", "nover6_native_rows"],
            "missing_cell_labels": ["not_run", "unsupported", "OOM", "invalid_cotenancy", "resource_blocked", "stopped_by_gate", "unexpected_failure"],
            "identity_match_required": ["model repository/revision and all weight/config/tokenizer hashes", "calibration document/crop/token hashes and order", "evaluation document/window/token hashes", "module scope, exclusions, padding and tile layout", "attention backend, dtype, batch, cache and runtime", "quantizer source identity", "exact map hash and pre-map model checksum"],
        },
        "calibration": {
            "domains": ["OpenWebMath", "CodeParrot-clean"], "sequence_length": 512,
            "canonical": {"math_sequences": 64, "code_sequences": 64, "draws": ["seed0", "draw1", "draw2", "draw3", "draw4"]},
            "draw_rule": "parent campaign/data.keyed_draw deterministic SHA-256 keyed ordering; documents/crops are mutually exclusive across seed0 and draw1-draw4 and excluded from evaluation",
            "larger_nested_draws": {
                "128+128": "the exact seed0 64+64 prefix plus deterministic key size128_v1, excluding seed0 and draw1-draw4, until 128 per domain",
                "256+256": "the exact 128+128 prefix plus deterministic key size256_v1, excluding every preceding calibration sample, until 256 per domain",
            },
            "stored_statistics": "float64 per-sequence sums, sums of squares and cross-products sufficient to recompute means, sample variances and SE; raw retained where feasible",
        },
        "evaluation": {
            "ppl": {
                "corpora": ["wikitext", "c4"], "sequence_length": 2048,
                "wikitext": "parent-pinned Salesforce/wikitext wikitext-2-raw-v1 test revision b08601e04326c79dfdd32d625aee71d232d685c3; all full non-overlapping windows of joined text",
                "c4": "parent-pinned allenai/c4 en validation shard revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2; exact parent Random(0) 256-document 2048-token crops",
                "aggregation": "raw PPL=exp(total token NLL/total predicted tokens); store per-window token count, NLL sum, source cluster and pairing key",
                "clusters": {"wikitext": "article containing first token", "c4": "document SHA-256"},
            },
            "accuracy": {
                "tasks": {"arc_easy": "acc_norm", "arc_challenge": "acc_norm", "hellaswag": "acc_norm", "openbookqa": "acc_norm", "boolq": "acc", "winogrande": "acc", "piqa": "acc_norm", "mmlu": "acc over 57 subjects"},
                "num_fewshot": 0, "batch_size": 16, "log_samples": True,
                "harness": "lm-eval 0.4.11 HFLM with parent offline documents and seeds (0,1234,1234,1234)",
                "macro": "unweighted mean of the eight task-level accuracy differences; MMLU is one task after pooling its frozen 57 subjects",
            },
            "gsm8k": {
                "task": "lm-eval gsm8k v3.0", "num_fewshot": 5,
                "decoding": "greedy do_sample=false; until [Question:, </s>, <|im_end|>]; max_gen_toks=256",
                "primary_metric": "exact_match,flexible-extract", "secondary_metric": "exact_match,strict-match",
                "retain": "all prompts, generations, extracted answers and paired item outcomes",
            },
            "pg19": {
                "corpus": "emozilla/pg19-test revision c5e39bf32e33f9111323aa68d7d9000d22722035",
                "lengths": [4096, 8192], "minimum_documents": 20,
                "selection": "for each tokenizer take the first 20 books in frozen file order with at least 8193 tokens; use nested prefix windows at both lengths; do not replace based on quality",
                "inference": "resample source books; fewer than 20 valid books is exploratory point-estimate-only",
            },
            "portability": {
                "models": ["llama8b", "granite8b"], "devices": ["NVIDIA RTX A6000", "NVIDIA RTX 6000 Ada Generation"],
                "windows": "first 16 frozen WikiText windows and first 16 frozen C4 windows",
                "map": "generate/freeze on clean A6000 and evaluate the byte-identical map on both devices; no Ada regeneration substitution",
                "aggregate_ppl_relative_tolerance": 0.005,
                "paired_effect_tolerance": "same sign and absolute A6000-vs-Ada dlogPPL difference <= max(0.25*abs(A6000 effect),0.002)",
                "per_window_portability_claim": "forbidden",
            },
        },
        "gates_and_thresholds": {
            "practical_ppl_material_regression_dlogppl": practical,
            "practical_ppl_definition": "log(1.005); a >=0.5% relative PPL regression is material",
            "development_safety_cell_dlogppl": 0.001,
            "development_meaningful_gain_dlogppl": -0.0005,
            "G0_integrity": "input hashes, exact authoritative attempt resolution, model/map/evaluator identities, GPU preflight, and both exact-map anchors must pass; no tolerance widening",
            "G1_k2_breadth": "Holm over six N16 k2-vs-FourOverSix endpoints; beneficial point estimates on both corpora for at least two of three breadth models; no cell >= practical material-regression threshold; report aggregate with and without Qwen",
            "G2_k2_vs_k3": "Holm over six breadth endpoints and frozen aggregate; no family-level material regression; differences smaller than the practical threshold are scientifically equivalent even if statistically detectable",
            "G3_density_selector": "at equal selected-weight budget N16 candidate must beat median random and show consistent advantage over deterministic controls; otherwise classify natural gain as density-driven",
            "G4_n16_vs_n8": "report natural and selected-weight-matched contrasts; worse N16 quality can only support an unmeasured engineering trade-off, never a favorable overhead claim",
            "G5_calibration_256_continuation": {
                "estimand": "for N16 k2, per development model average of WikiText and C4 dlogPPL(128+128 minus 64+64)",
                "continue_if": "at least 2 of 3 model averages <= -0.0005 and every individual model/corpus cell <= +0.001",
                "otherwise": "all P32 cells stopped_by_gate; report P31 fully",
            },
            "G6_selector_replacement": {
                "full_map": "at equal budget, median across six development cells dlogPPL(candidate minus old selector) <= -0.0005 and worst cell <= +0.001",
                "tile_sign": "exact-binomial 95% lower confidence bound > 0.5 and sign-accuracy point gain over old selector >= 0.15",
                "tile_rank": "cluster/stratum bootstrap 95% lower bound for Spearman rho > 0; also report Kendall tau",
                "claim": "causal selector replacement requires all full-map, sign and rank conditions; PPL-only success is empirical selection quality only",
            },
            "G7_quantizer": "candidate must retain exact four-bit E2M1/E0M3 value formats and W4A4 scope and fully disclose scale metadata/high-precision exceptions/unfused operations; hidden high-precision work, noncausal calibration, or incompatible MMA scope is rejected or separately labeled",
            "G8_heldout_downstream": "global winner hash-frozen before held-out quality; evaluate both reserved families; positive general-quality claim also requires frozen accuracy non-inferiority after correction",
            "G9_sota": "only if one frozen method beats the strongest fair baseline on a broad non-cherry-picked panel with corrected inference; any missing baseline cells or material RaZeR wins prohibit broad SOTA wording",
        },
        "development_search": {
            "k": {
                "candidates": [0.5, 1.0, 1.5, 2.0, 2.5], "anchors": [2.0, 3.0, "all_e0m3"],
                "panel": ["llama8b", "qwen4b", "mistral7b"], "tile_shape": [16, 64], "per_model_tuning": "forbidden",
                "eligibility": "all six cells dlogPPL(candidate minus N16 k2 seed0) <= +0.001",
                "lexicographic_selection": ["smallest median dlogPPL versus N16 k2", "if within 0.0001, smallest worst-cell dlogPPL", "if still tied, fewer total selected weights", "if still tied, larger k"],
                "fallback": "N16 k2 seed0",
            },
            "density_controls": {
                "target": "exact selected-weight count of N16 k2, both globally and separately per layer/module",
                "random_seeds": [2026091401, 2026091402, 2026091403, 2026091404, 2026091405],
                "deterministic": ["weight_mse", "magnitude", "change_norm", "activation_weighted"],
                "n8_matching": "one N16K64 tile equals 1024 weights and one N8K64 tile equals 512; select exactly twice the N8 tile count where divisible",
                "rounding": "floor each stratum then allocate remaining tiles by largest fractional remainder; ties by module name then row then K index; report residual",
                "eligibility_and_ties": "exact same module/weight eligibility mask; finite scores before nonfinite; score then module name,row,K; no repeated random draw",
            },
            "draw_aggregation": {
                "candidates": ["seed0", "pooled_per_sequence_moments", "draw_mean", "draw_median", "consensus_3_of_5"],
                "pooled_rule": "pool per-sequence sufficient statistics then recompute mean/sample variance/SE; never average masks",
                "forms": ["natural_density", "fixed_budget_matched_to_seed0_k2"],
                "eligibility": "median six-cell dlogPPL versus seed0 <= -0.0005 and worst cell <= +0.001",
                "selection": ["smallest median dlogPPL", "if within 0.0001, smallest worst cell", "if tied, highest median leave-one-draw-out map Jaccard", "if tied, fewer selected weights", "if none eligible, seed0"],
            },
            "selector_candidates": {
                "old": "first_order_ce_kl",
                "activation_weighted_error": "per-tile L1 E0M3-minus-E2M1 weight error weighted by calibration input-channel RMS, averaged across development calibration only",
                "layer_output_reconstruction": "per-tile change in mean squared linear-layer output reconstruction error on cached development calibration inputs",
                "diagonal_hessian_gptq_proxy": "per-tile squared quantization-error change weighted by the diagonal empirical input Gram/Hessian proxy E[x_j^2] with damping 0.01*mean(diagonal)",
                "combined": "not primary; may be attempted only after component results, with frozen coefficients fit on development tile interventions and no evaluation-window outcomes",
                "fixed_budget": True,
                "staging": "derive/store scores first; evaluate one map per named candidate plus old selector and strongest deterministic heuristic, not a Cartesian product with scale/clipping",
            },
            "tile_fidelity": {
                "sample_tiles_per_model": 60, "models": ["llama8b", "qwen4b", "mistral7b"],
                "freeze_before_effects": True,
                "strata": ["layer quartile", "module type", "old/candidate score quintile", "predicted sign"],
                "sampling": "deterministic weighted allocation without replacement; redistribute empty-stratum quota lexicographically and store inclusion probabilities",
                "outcomes": ["CE", "KL", "layer reconstruction"],
            },
            "format_preserving_quantizer": {
                "weight_scale_multipliers": [0.95, 1.0, 1.05],
                "weight_rule": "multiply the existing per-scale-block absmax-derived dequantization scale; identical E2M1/E0M3 codes, scope and scale-block count",
                "activation_candidates": ["baseline_absmax", "row_mse_grid_0.90_0.95_1.00", "row_percentile_99.9", "row_percentile_99.99", "row_percentile_100"],
                "activation_rule": "causal per-token row only; MSE grid chooses among [0.90,0.95,1.00] times row absmax using that row; percentiles use only the current row; four-bit FourOverSix values retained",
                "independent_pass": "median six-cell dlogPPL versus existing scale <= -0.0005 and worst cell <= +0.001",
                "selection": ["smallest median dlogPPL", "if within 0.0001, smallest worst cell", "least extra metadata/operations"],
                "combination": "combine only independently passing best weight and activation choices with the frozen selector; evaluate baseline plus single-axis screens before the combination",
                "local_transform": "P62 stopped_by_gate unless a pre-result algebraic/folding/MMA-scope proof shows no hidden high-precision state and accounts for every metadata bit and operation",
            },
        },
        "statistics": {
            "ppl_estimand": "paired token-weighted dlogPPL = mean_NLL(method)-mean_NLL(comparator)",
            "bootstrap": {"replicates": 10000, "interval": "paired cluster percentile 95%", "master_seed": 20260914, "rng": "numpy PCG64 via stable named SeedSequence streams"},
            "finite_mc_pvalue": "(extreme_count+1)/(B+1); exact p=0 forbidden",
            "multiplicity": {
                "method": "Holm within each frozen family",
                "families": {
                    "breadth_quality": "six N16 k2-vs-FourOverSix endpoints on qwen27b, phi4, olmo2_13b",
                    "threshold_improvement": "six N16 k2-vs-N16 k3 endpoints on the same panel",
                    "tile_shape": "six selected-weight-matched N16-vs-N8 endpoints on the same panel",
                    "heldout_quality": "four winner-vs-FourOverSix endpoints on granite8b and falcon3_10b",
                    "accuracy": "eight task-level paired winner-vs-FourOverSix endpoints for each tested model; model panels reported separately",
                    "strongest_baseline": "opened only after preceding gates, baseline selected without N16/winner outcome peeking",
                },
            },
            "draws": "report every draw, mean, median, range, worst, between-draw sample SD, t_4 interval, beneficial fraction, pairwise map Jaccard and count variability; within-evaluation and between-draw uncertainty remain separate",
            "accuracy": {"sesoi_pp": -0.5, "noninferiority": "paired macro lower 95% CI > -0.5 percentage points after frozen correction", "rng_stream": "independent deterministic stable task-name SeedSequence"},
            "tile_fidelity": "exact binomial sign CI; Spearman and Kendall with stratified bootstrap; predicted-vs-observed calibration; sampling weights",
            "post_selection": "no naive development winner confidence interval is labeled confirmatory; confirmatory inference starts only after winner hash freeze on unopened held-out endpoints",
        },
        "anchor_reproduction": {
            "device": "one clean NVIDIA RTX A6000", "order": ["first_four_wikitext_windows", "full_wikitext"],
            "map_regeneration": "forbidden", "policies": ["four_over_six", "n16_k3_exact_map"],
            "acceptance": {"each_policy_relative_ppl": 0.005, "paired_effect": "same sign and |new-parent dlogPPL| <= max(0.25*abs(parent dlogPPL),0.002)", "failure": "stop large matrix; diagnose; do not widen"},
            "models": {
                "llama8b": {
                    "map_logical_path": "parent:runs/V30_calib_llama8b_seed0_attempt1/maps/llama8b_seed0_n16_k3.mixfp4map",
                    "map_sha256": "0920f55ddc053a5f5a8b0d046d0b74a62d4abafcad5e36051d7682da961e1f2b", "selected_tiles": 1781, "selected_weights": 1823744,
                    "evaluation_manifest_sha256": "8d72dbf23db218671856f620bd49c2226db829880673f106078e375a8ca35197",
                    "parent": {"first4": {"four_over_six": {"mean_nll": 2.0610900399817313, "raw_ppl": 7.854526894041561}, "n16_k3": {"mean_nll": 2.0593648955835695, "raw_ppl": 7.840988382271653}, "dlogppl": -0.0017251443981618486}, "full": {"four_over_six": {"mean_nll": 1.9279499304938719, "raw_ppl": 6.8754007343996815}, "n16_k3": {"mean_nll": 1.92326091483662, "raw_ppl": 6.843237338969587}, "dlogppl": -0.004689015657251883}},
                },
                "mistral7b": {
                    "map_logical_path": "parent:runs/V61_calib_mistral7b_seed0_attempt2/maps/mistral7b_seed0_n16_k3.mixfp4map",
                    "map_sha256": "0c3d822a18d0480ca2aeff0abd78cf6e4ca3155bb207156a400fde124e1cf13d", "selected_tiles": 4179, "selected_weights": 4279296,
                    "evaluation_manifest_sha256": "f15f8bad484a93abafa08985a924e255d5f2cdd535bfb9c1d8785487b258cafe",
                    "parent": {"first4": {"four_over_six": {"mean_nll": 1.8626170782621378, "raw_ppl": 6.440570210901774}, "n16_k3": {"mean_nll": 1.8609998059798598, "raw_ppl": 6.430162473558694}, "dlogppl": -0.0016172722822780372}, "full": {"four_over_six": {"mean_nll": 1.708977147744794, "raw_ppl": 5.5233090581954585}, "n16_k3": {"mean_nll": 1.7050881772071924, "raw_ppl": 5.50187078542319}, "dlogppl": -0.0038889705376015105}},
                },
            },
        },
        "gpu_policy": {
            "allowed_models": ["NVIDIA RTX A6000", "NVIDIA RTX 6000 Ada Generation"],
            "campaign_max_concurrent_gpus": 3, "mixed_gpu_models_within_run": False,
            "foreign_compute_process_allowed": False, "fail_closed": True,
            "ownership_check_interval_seconds": 20, "maximum_allowed_interval_seconds": 60,
            "checks": ["host prelaunch PID/owner/index/UUID/model/memory/process snapshot", "in-container visible UUID cross-check", "20-second host sidecar", "every phase boundary", "immediately before result finalization", "host and container postflight"],
            "foreign_process_action": "stop only this campaign run, preserve evidence, mark invalid_cotenancy, safely release and requeue; never signal another user's process",
            "no_gpu_lease_for_cpu_jobs": True,
        },
        "winner_freeze": {
            "required_before": ["P71_HELDOUT_PPL", "P72_ACCURACY", "P73_GSM8K", "P74_PG19", "P75_PORTABILITY"],
            "contents": "exact source manifest, maps/hashes, model-independent configuration, scale/clip/selector/aggregation/k, module scope, calibration manifests, evaluation plan, comparators and analysis hash",
            "candidate_changes_after_opening_heldout": "forbidden",
            "fallback": "N16 k2 seed0 with existing scale and activation if no development candidate passes its frozen gate",
        },
        "claim_classification": {
            "hierarchy": ["general positive MixFP4 selection method", "robust post-hoc k2 quality extension without new mechanism", "density-driven PPL improvement", "N16/N8 engineering trade-off study without measured deployment benefit", "map-level robustness plus tile-level negative result", "workshop/negative-result study"],
            "native_or_performance_claims": "prohibited",
            "selection_rule": "choose the narrowest classification supported by all gates; never omit contrary evidence",
        },
        "carryover_bundle_repairs": {
            "finite_monte_carlo_plus_one": "fixed and tested in extension source",
            "independent_accuracy_rng": "fixed and tested in extension source",
            "required_in_new_bundle": ["repair redacted protocol-freeze sidecar", "repair redacted amendment-chain entries 10,11,32,37,52", "replace Windows-unsafe angle-bracket lock filenames with safe logical names", "state that evaluation CIs are conditional on the stored map and separately report draw variability"],
            "parent_mutation": "forbidden",
        },
        "quality_results_opened_before_lock": {
            "extension_results": [],
            "known_parent_development_and_posthoc_results": "the parent reports and handoff EXISTING_*.csv tables were already viewed; they remain development/post-hoc and cannot become prospective",
            "heldout_quality": [],
        },
        "matrix": [{"matrix_id": row["matrix_id"], "phase": int(row["phase"]), "requiredness": row["requiredness"], "gate": row["gate"]} for row in matrix_rows],
    }

    freeze_dir = CR / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = freeze_dir / "PROTOCOL_EXTENSION.json"
    runtime.atomic_json(protocol_path, protocol)
    digest = sha(protocol_path)
    sidecar = freeze_dir / "PROTOCOL_EXTENSION.sha256"
    sidecar.write_text(f"{digest}  PROTOCOL_EXTENSION.json\n")
    protocol_path.chmod(0o444)
    sidecar.chmod(0o444)

    amendment = {
        "schema_version": "1.0", "event": "protocol_genesis", "logged_utc": locked_at,
        "protocol_sha256": digest, "previous_entry_sha256": None,
        "entry_sha256_rule": "SHA-256 of canonical JSON excluding entry_sha256",
    }
    amendment["entry_sha256"] = canonical_sha(amendment)
    amendments_path = CR / "provenance/PROTOCOL_EXTENSION_AMENDMENTS.jsonl"
    amendments_path.write_text(json.dumps(amendment, sort_keys=True, separators=(",", ":")) + "\n")

    runtime.atomic_json(runtime.run_dir / "job_result.json", {
        "protocol_id": "ppl-improvement-extension-v1", "protocol_freeze_sha256": digest,
        "source": {"model_id": "protocol-extension", "model_revision": digest,
                   "tokenizer_revision": "not_applicable", "model_class": "not_applicable",
                   "module_manifest_sha256": "not_applicable",
                   "source_manifest_sha256": launch["source_manifest_sha256"]},
        "environment": runtime.environment(),
        "data": {"calibration_manifest_sha256": None, "evaluation_manifest_sha256": None,
                 "token_hashes": {}, "overlap_audit": None},
        "policies": [],
        "results": {"raw_outputs": [str(protocol_path), str(sidecar), str(amendments_path)],
                    "summary": {"protocol_sha256": digest, "status": "LOCKED",
                                "extension_quality_results_before_lock": 0,
                                "heldout_reservation_sha256": reservation_sha},
                    "uncertainty": {}, "attempted_endpoints": ["P00_PROTOCOL_LOCK"],
                    "missing_endpoints": []},
        "logs": [], "failures": [],
    })
    print(json.dumps({"protocol_path": str(protocol_path), "protocol_sha256": digest,
                      "locked_at_utc": locked_at}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

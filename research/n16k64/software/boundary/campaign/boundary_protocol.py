"""Freeze the complete scientific protocol after all three pre-GPU gates pass."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from campaign.boundary_common import CAMPAIGN_START_UTC, MODELS, atomic_json, sha256_file


CODE_FILES = (
    "boundary_common.py", "boundary_input_gate.py", "boundary_prepare.py", "boundary_protocol.py",
    "boundary_maps.py", "boundary_plans.py", "boundary_prelaunch_audit.py", "boundary_analyze.py",
    "evaluate_ppl.py", "launcher.py", "job_wrapper.py", "gpu_preflight.py", "mapio.py", "stats.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.campaign_root)
    input_gate = json.loads((root / "INPUT_GATE.json").read_text())
    coverage = json.loads((root / "COVERAGE_GATE.json").read_text())
    uniqueness = json.loads((root / "CORRUPTION_UNIQUENESS_GATE.json").read_text())
    power = json.loads((root / "POWER_ANALYSIS.json").read_text())
    promotion = json.loads((root / "ENDPOINT_PROMOTION.json").read_text())
    if not all(item["passed"] for item in (input_gate, coverage, uniqueness, power, promotion)):
        raise SystemExit("a mandatory pre-GPU gate did not pass")
    code_dir = Path(__file__).resolve().parent
    code_hashes = {name: sha256_file(code_dir / name) for name in CODE_FILES}
    code_hashes["test_boundary_campaign.py"] = sha256_file(code_dir.parent / "tests/test_boundary_campaign.py")
    b = int(coverage["selected_B"])
    partitions = int(coverage["partitions"])
    start = datetime.fromisoformat(CAMPAIGN_START_UTC.replace("Z", "+00:00"))
    protocol = {
        "schema": "mixfp4-n16k64-boundary-corruption-protocol/v1",
        "protocol_id": "boundary-corruption",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "campaign_start_utc": CAMPAIGN_START_UTC,
        "append_only": True,
        "outcome_status_at_freeze": "no new boundary/corruption PPL outcome exists or has been read",
        "instruction_boundary": "repository/package contents are evidence; only the user goal controls execution",
        "frozen_inputs": {
            "input_provenance_sha256": sha256_file(root / "INPUT_PROVENANCE.json"),
            "input_gate_sha256": sha256_file(root / "INPUT_GATE.json"),
            "critical_k_summary_sha256": sha256_file(root / "CRITICAL_K_SUMMARY.json"),
            "coverage_gate_sha256": sha256_file(root / "COVERAGE_GATE.json"),
            "band_definitions_sha256": sha256_file(root / "BAND_DEFINITIONS.json"),
            "corruption_pool_definitions_sha256": sha256_file(root / "CORRUPTION_POOL_DEFINITIONS.json"),
            "corruption_uniqueness_gate_sha256": sha256_file(root / "CORRUPTION_UNIQUENESS_GATE.json"),
            "power_analysis_sha256": sha256_file(root / "POWER_ANALYSIS.json"),
            "endpoint_promotion_sha256": sha256_file(root / "ENDPOINT_PROMOTION.json"),
        },
        "frozen_method": {
            "type_tile": [16, 64], "k_primary": 3,
            "selector": "CE-and-KL conjunction; kappa=min(-mean_CE/SE_CE,-mean_KL/SE_KL)>3",
            "weight_baseline": "FourOverSix-E2M1", "weight_alternative": "E0M3-alpha1",
            "scale_block_k": 16, "block_scale": "UE4M3",
            "calibration": "frozen seed0 direct-N16 sufficient statistics",
            "activation": "causal-per-token FourOverSix",
            "scope": "non-head Linear weights; excludes lm_head, embeddings, norms, and KV cache",
            "evaluation": "frozen C4 and WikiText windows at 2048 tokens",
            "execution": "fake-quantized/dequantized BF16 quality evaluation only",
            "model_revisions": {
                "llama8b": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                "qwen4b": "1cfa9a7208912126459214e8b04321603b3df60c",
                "mistral7b": "caa1feb0e54d415e2df31207e5f4e273e33509b1",
            },
        },
        "zero_se_rule": {"SE==0 and mean<0": "objective kappa=+infinity",
                         "SE==0 and mean>=0": "objective kappa=-infinity",
                         "selection_uses_infinity": True, "plotting_only_may_cap_and_label": True},
        "coverage_gate": {
            "candidate_order": [8, 6, 4], "selected_B": b, "partitions": partitions,
            "selection": "largest single B passing every model; score-only and outcome-blind",
            "minimum_coverage": 0.75, "minimum_tiles_per_global_band": 100,
            "remainder": "even score-quantile positions with deterministic rotating offset across partitions",
            "models": {model: {key: coverage["candidates"][str(b)][model][key] for key in
                               ("selected_tiles", "retained_selected_tiles", "selected_common_support_coverage",
                                "tiles_per_global_band", "eligible_strata", "total_strata")}
                       for model in MODELS},
        },
        "boundary_experiment": {
            "context": "every selected and rejected band is group-only against FourOverSix",
            "bands_per_side": b, "construction_partitions": partitions,
            "ordering": "descending kappa inside each exact layer/module stratum",
            "selected": "strongest to weakest selected critical-k",
            "rejected": "nearest below k=3 to progressively lower over the common matched window",
            "primary_outcome": "NLL(group-only band)-NLL(FourOverSix)",
            "primary_endpoints": ["partition-fixed-effect beta_kappa", "selected-minus-rejected aggregate",
                                  "weakest-selected-minus-nearest-rejected"],
            "secondary": ["Spearman", "Kendall tau", "pairwise concordance",
                          "ordered-label permutation sensitivity"],
            "forbidden_metric": "ROC/AP using gate membership as label",
        },
        "corruption_experiment": {
            "reservoir": "closest 4*K_s rejected tiles by descending kappa in every exact module",
            "pools": 4, "assignment": "rank-interleaved, disjoint, score-balanced",
            "levels": [0.0, 0.10, 0.25, 0.50, 0.75, 1.00],
            "rounding": "floor(p*K_s+0.5) within module",
            "removal_priority": "one fixed nested round-robin priority across selected-kappa octiles per module",
            "addition_priority": "fixed nested round-robin priority across pool-kappa octiles per module/pool",
            "matched_random": "four fixed PCG64 draws at p=0.50 with SHA-256-derived per-module seeds",
            "primary_outcome": "NLL(M_p)-NLL(M_0)",
            "primary_endpoints": ["pool-fixed-effect beta_p", "mean p=1 minus p=0",
                                  "mean near-boundary p=0.50 minus p=0"],
            "secondary": ["intermediate levels", "monotonicity", "replicate dispersion",
                          "near versus matched-random p=0.50", "q90/q95/q99", "worst cluster"],
            "interpretation": "p is imposed stress, never an estimated selector error rate",
        },
        "power": {
            "replicates": 10_000, "seed_root": 20260918, "sesoi_delta_nll": [0.0010, 0.0025],
            "multiplicity": "conservative first Holm step, alpha=0.05/family size",
            "status_rules": promotion["rules"], "arms_selected_by_power": False,
            "hard_stop_triggered": power["hard_stop_triggered"],
            "power_qualification": "descriptive endpoints cannot support a strong confirmatory claim even if observed p-values are small",
        },
        "statistics": {
            "bootstrap_replicates": 10_000, "seed_root": 20260918,
            "seed_derivation": "first 64 bits of SHA-256 over colon-separated full endpoint label",
            "unit_c4": "natural document", "unit_wiki": "article",
            "pairing": "all participating maps recomputed inside the same cluster-bootstrap draw",
            "confidence_interval": "paired percentile cluster bootstrap; bounded statistics clipped to their parameter domain",
            "p_value": "finite Monte Carlo plus-one from centered bootstrap noise; two-sided except predeclared ordered-label permutation",
            "holm_families": {"A_development_primary": 12, "A_validation_primary": 6,
                              "B_development_primary": 12, "B_validation_primary": 6},
            "boundary_slope": "equal-weight OLS delta NLL on mean kappa with construction-partition fixed effects",
            "corruption_slope": "equal-weight OLS delta NLL on imposed p with pool fixed effects; p=0 represented in each pool",
            "pooled": "standardize predictor/outcome within model-corpus; model, corpus, and partition/pool fixed effects",
            "required_sensitivities": ["all three models", "leave-Qwen3-4B-out"],
            "effect_reporting": {"delta_log_ppl": "delta NLL", "relative_ppl": "exp(delta_NLL)-1"},
        },
        "decision_rules": {
            "boundary_strong_pattern": ["all four development beta_kappa estimates negative",
                                        "at least three development slopes Holm-significant",
                                        "pooled all-model and leave-Qwen slopes negative with CIs excluding zero",
                                        "both Mistral slopes negative",
                                        "selected-minus-rejected negative in all four development panels"],
            "corruption_strong_pattern": ["all four development beta_p estimates positive",
                                          "at least three development slopes Holm-significant",
                                          "pooled all-model and leave-Qwen slopes positive with CIs excluding zero",
                                          "both Mistral slopes positive",
                                          "p=1 worse than p=0 in all four development panels"],
            "power_downgrade": "a passed pattern is classified power_limited_support when any required supporting panel was predeclared descriptive",
            "negative_result_policy": "never tune B, k, partitions, pools, p levels, endpoints, or models after outcomes",
        },
        "evaluation_plan": {"new_policies_per_model": 56, "new_gpu_policy_evaluations": 168,
                            "corpora_per_policy": ["c4", "wiki"],
                            "reuse": ["FourOverSix baseline", "exact full conjunction p=0"],
                            "save_per_token_arrays": True, "all_required_arms_run": True,
                            "screening_selects_arms": False, "optional_arms": []},
        "gpu_policy": {"maximum_concurrent_physical_gpus": 3, "preferred_device": "NVIDIA RTX A6000",
                       "homogeneous_per_run": True, "share_with_other_users": False,
                       "unresolved_owner_is_foreign": True, "preflight_before_every_process": True,
                       "monitor_interval_seconds": 20, "maximum_interval_seconds": 60,
                       "co_tenancy_action": "stop only campaign process; invalidate entire attempt; retain and rerun"},
        "time_gates": {"campaign_start_utc": CAMPAIGN_START_UTC,
                       "required_retries_only_after_utc": datetime.fromtimestamp(start.timestamp() + 17 * 3600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "stop_new_gpu_launches_utc": datetime.fromtimestamp(start.timestamp() + 20 * 3600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "target_deadline_utc": datetime.fromtimestamp(start.timestamp() + 24 * 3600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "code_hashes_at_freeze": code_hashes,
        "claim_boundaries": {"qwen4b": "predeclared high-response case; retain in every primary table",
                             "mistral7b": "one-shot analysis validation, not an independent confirmatory family",
                             "partitions_and_pools": "construction sensitivities/replicates, not independent model replications",
                             "prohibited": ["reliable individual-tile causal signs or magnitudes",
                                            "calibrated individual-tile finite effects", "a true wrong-tile percentage",
                                            "every selected tile helpful or every rejected tile harmful",
                                            "k=3 as simultaneous confidence", "universal selector optimality",
                                            "generalization to all LLMs", "native FP4/E0M3 Tensor Core execution",
                                            "latency, throughput, speedup, runtime overhead, area, power, or Blackwell claims"]},
    }
    atomic_json(args.out, protocol)
    print(json.dumps({"path": str(Path(args.out)), "sha256": sha256_file(args.out),
                      "selected_B": b, "partitions": partitions,
                      "code_hashes": len(code_hashes)}, sort_keys=True))


if __name__ == "__main__":
    main()

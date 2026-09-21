#!/usr/bin/env python3
"""Render candid, fixed-template mechanism reports from frozen analysis JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def effect_cells(result: dict) -> str:
    lo, hi = result["ci95"]
    holm = result.get("holm_adjusted_p")
    hp = "—" if holm is None else f"{holm:.6g}"
    return (f'{result["estimate"]:+.7f} | [{lo:+.7f}, {hi:+.7f}] | '
            f'{100 * math.expm1(result["estimate"]):+.4f}% | {hp}')


def ranking_table(results: dict) -> list[str]:
    lines = [
        "## Ranking-not-calibration primary contrasts",
        "",
        "| Role | Model | Corpus | Contrast | delta-log-PPL | 95% CI | Relative PPL | Holm p |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for model, mr in results["models"].items():
        role = "one-shot validation" if model == "mistral7b" else "development"
        for domain, dr in mr["domains"].items():
            for endpoint, label in (("group_only:strongest-weakest", "group-only strongest − weakest"),
                                    ("full_context_marginal:strongest-weakest", "marginal strongest − weakest")):
                lines.append(f"| {role} | {model} | {domain} | {label} | {effect_cells(dr['ranking'][endpoint])} |")
    lines.append("")
    return lines


def veto_table(results: dict) -> list[str]:
    lines = [
        "## CE/KL-veto primary contrasts",
        "",
        "Positive add-back effects are worse NLL. The matched-random contrast controls composition and approving-margin bins.",
        "",
        "| Role | Model | Corpus | Veto class | Contrast | delta-log-PPL | 95% CI | Relative PPL | Holm p |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for model, mr in results["models"].items():
        role = "one-shot validation" if model == "mistral7b" else "development"
        for domain, dr in mr["domains"].items():
            for label, block in dr["veto"].items():
                for name, text in (("actual_vs_full", "actual add-back − conjunction"),
                                   ("actual_vs_matched_random", "actual add-back − matched random")):
                    lines.append(f"| {role} | {model} | {domain} | {label} | {text} | {effect_cells(block[name])} |")
    lines.append("")
    lines += [
        "### Veto tail differences: actual add-back minus matched random",
        "",
        "| Role | Model | Corpus | Veto class | Endpoint | Difference | 95% CI | Holm p |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for model, mr in results["models"].items():
        role = "one-shot validation" if model == "mistral7b" else "development"
        for domain, dr in mr["domains"].items():
            for label, block in dr["veto"].items():
                for endpoint, value in block["tails"].items():
                    lo, hi = value["ci95"]
                    lines.append(f'| {role} | {model} | {domain} | {label} | {endpoint} | '
                                 f'{value["estimate"]:+.7f} | [{lo:+.7f}, {hi:+.7f}] | '
                                 f'{value["holm_adjusted_p"]:.6g} |')
    lines.append("")
    return lines


def interaction_table(results: dict) -> list[str]:
    lines = [
        "## Attention/MLP interaction primary contrasts",
        "",
        "| Role | Model | Corpus | Contrast | delta-log-PPL | 95% CI | Relative PPL | Holm p |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for model, mr in results["models"].items():
        role = "one-shot validation" if model == "mistral7b" else "development"
        for domain, dr in mr["domains"].items():
            for name, label in (("actual_residual", "actual interaction residual"),
                                ("actual_minus_matched_random", "actual − matched-random residual")):
                lines.append(f"| {role} | {model} | {domain} | {label} | {effect_cells(dr['interaction'][name])} |")
    lines.append("")
    return lines


def absolute_table(results: dict) -> list[str]:
    lines = [
        "## Absolute PPL context",
        "",
        "Absolute PPL is descriptive context; the inferential endpoint is paired delta NLL (delta-log-PPL).",
        "",
        "| Role | Model | Corpus | FourOverSix PPL | Full-map PPL | Full − baseline delta-log-PPL | Relative PPL |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for model, mr in results["models"].items():
        role = "one-shot validation" if model == "mistral7b" else "development"
        for domain, dr in mr["domains"].items():
            base = dr["absolute_ppl"]["four_over_six"]
            full = dr["absolute_ppl"]["full"]
            delta = math.log(full) - math.log(base)
            lines.append(f"| {role} | {model} | {domain} | {base:.7f} | {full:.7f} | {delta:+.7f} | {100 * math.expm1(delta):+.4f}% |")
    lines.append("")
    return lines


def fidelity_table(score: dict) -> list[str]:
    lines = [
        "## Existing individual-tile counterevidence",
        "",
        "These frozen attempt7 finite-effect results are the counterpoint to any aggregate result; they are not overwritten by this campaign.",
        "",
        "| Model | n | Sign precision | Sign recall | Pearson | Spearman | Resolvable @1.96SE | Resolvable @3SE | Full-map actual/predicted CE ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, block in score["models"].items():
        f = block["individual_tile_existing_frozen_evidence"]
        lines.append(f'| {model} | {f["n"]} | {f["sign_precision"]:.4f} | {f["sign_recall"]:.4f} | '
                     f'{f["pearson"]:.4f} | {f["spearman"]:.4f} | {f["resolvable_fraction_1p96se"]:.4f} | '
                     f'{f["resolvable_fraction_3se"]:.4f} | {f["full_map_actual_to_predicted_ce_ratio"]:.4f} |')
    lines.append("")
    return lines


def verdict_text(results: dict, score: dict) -> str:
    verdict = results["hypothesis_classification"]
    lines = [
        "# Mechanism verdict",
        "",
        "This is a fixed-gate verdict for aggregate ranking, CE/KL veto risk control, and attention/MLP interaction. "
        "A failed gate is a negative result, not a reason to redefine an endpoint.",
        "",
        "## Verdict summary",
        "",
        "| Question | Classification | Frozen development gate passed? |",
        "|---|---|---:|",
    ]
    for key in ("ranking_not_calibration", "ce_kl_veto", "attention_mlp_interaction"):
        item = verdict[key]
        lines.append(f'| {key.replace("_", " ")} | {item["classification"]} | {"yes" if item["passed"] else "no"} |')
    lines += ["", "## Direct answer", ""]
    if verdict["ranking_not_calibration"]["passed"]:
        lines.append("The strict development gate supports the proposed reconciliation at group level: the score carries useful aggregate ranking information even though the frozen finite-effect evidence does not calibrate individual tiles reliably.")
    else:
        lines.append("The strict development gate does not establish the proposed ranking-based reconciliation. The campaign therefore cannot explain the model-level gain by asserting reliable score-margin ordering, even at the tested group scale.")
    if verdict["ce_kl_veto"]["passed"]:
        lines.append("The conjunction also passes the frozen CE/KL-veto protection gate against composition- and approving-margin-matched add-backs.")
    else:
        lines.append("The CE/KL conjunction does not pass the frozen veto-protection gate across both one-objective-pass classes; any favorable individual endpoint remains partial evidence only.")
    if verdict["attention_mlp_interaction"]["passed"]:
        lines.append("Attention and MLP effects are detectably non-additive under the frozen interaction gate, so a sum of isolated module effects is not an adequate account of the full map.")
    else:
        lines.append("The frozen interaction gate does not establish a stable attention/MLP non-additivity pattern across all development endpoints.")
    if "per_class_passed" in verdict["ce_kl_veto"]:
        detail = ", ".join(f"{name}={'pass' if passed else 'fail'}" for name, passed in verdict["ce_kl_veto"]["per_class_passed"].items())
        lines.append(f"Veto subclass gates: {detail}.")
    lines += ["", "The Mistral results below are one-shot analysis validation only; they do not convert a development finding into a fully independent model-family generalization claim.", ""]
    lines += absolute_table(results)
    lines += ranking_table(results)
    lines += veto_table(results)
    lines += interaction_table(results)
    lines += fidelity_table(score)
    lines += [
        "## Interpretation boundary",
        "",
        "The admissible interpretation is group-level and model-level: a selector may carry useful aggregate ranking or risk-control information even when its per-tile first-order estimate is noisy, biased, or interaction-confounded. "
        "No result here establishes reliable individual-tile causal signs or magnitudes.",
        "",
        "All evaluation is fake-quantized/dequantized BF16 quality evaluation. Native FP4/E0M3 Tensor Core execution, latency, speedup, area, and power claims remain prohibited and untested.",
        "",
    ]
    return "\n".join(lines)


def recommendation_text(results: dict) -> str:
    verdict = results["hypothesis_classification"]
    supported = [name for name, value in verdict.items() if value["passed"]]
    lines = ["# Next-step recommendation", ""]
    if supported:
        lines += [
            "Do not launch another broad selector search. If more compute is authorized, preregister a narrow replication containing only the strict-gate-supported mechanism contrast(s): " + ", ".join(supported) + ".",
            "",
            "The replication should preserve the same composition matching, natural-cluster paired inference, and held-out analysis rules. It should target uncertainty reduction and failure-boundary characterization, not best-PPL selection.",
        ]
    else:
        lines += [
            "Do not launch another broad selector search or claim a resolved mechanism. None of the three strict development gates passed; retain the null/negative evidence and revise the mechanistic hypothesis before allocating new GPU work.",
        ]
    lines += [
        "",
        "Regardless of gate outcome, do not claim individual-tile causality. Keep native-kernel execution and latency, speedup, area, or power as a separate project requiring dedicated kernels and measurements.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", required=True)
    args = ap.parse_args()
    root = Path(args.campaign_root).resolve()
    results = load(root / "MECHANISM_RESULTS.json")
    score = load(root / "SCORE_MARGIN_ANALYSIS.json")
    history = root / "freeze_history"
    history.mkdir(exist_ok=True)
    for name in ("MECHANISM_VERDICT.md", "NEXT_STEP_RECOMMENDATION.md"):
        source = root / name
        archived = history / f"{source.stem}_analysis_generated{source.suffix}"
        if source.exists() and not archived.exists():
            archived.write_bytes(source.read_bytes())
    (root / "PRIMARY_RESULTS_TABLES.md").write_text(
        "\n".join(["# Primary mechanism results tables", ""] + absolute_table(results) +
                  ranking_table(results) + veto_table(results) + interaction_table(results)), encoding="utf-8")
    (root / "MECHANISM_VERDICT.md").write_text(verdict_text(results, score), encoding="utf-8")
    (root / "NEXT_STEP_RECOMMENDATION.md").write_text(recommendation_text(results), encoding="utf-8")
    print(json.dumps({"rendered": ["PRIMARY_RESULTS_TABLES.md", "MECHANISM_VERDICT.md", "NEXT_STEP_RECOMMENDATION.md"]}))


if __name__ == "__main__":
    main()

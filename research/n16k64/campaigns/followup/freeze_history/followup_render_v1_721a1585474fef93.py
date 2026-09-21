"""Render frozen follow-up results into reviewer-facing reports and figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("llama8b", "qwen4b", "mistral7b")
DOMAINS = ("c4", "wiki")
DISPLAY = {"llama8b": "Llama-3.1-8B", "qwen4b": "Qwen3-4B", "mistral7b": "Mistral-7B-v0.3",
           "c4": "C4", "wiki": "WikiText"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt(value, digits=7):
    return "NA" if value is None else f"{value:+.{digits}f}"


def ci(result: dict) -> str:
    lo, hi = result.get("ci95", [None, None])
    return "[NA, NA]" if lo is None else f"[{lo:+.7f}, {hi:+.7f}]"


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def dose_figure(root: Path, dose: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True)
    for row, domain in enumerate(DOMAINS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            bins = dose["models"][model][domain]["bins"]
            x = np.arange(1, 9)
            for endpoint, color, marker, label in (
                ("full_context_marginal", "#155e75", "o", "full − full-minus-bin"),
                ("group_only_vs_baseline", "#b45309", "s", "group-only − baseline"),
            ):
                y = np.asarray([b[endpoint]["estimate"] for b in bins])
                lo = y - np.asarray([b[endpoint]["ci95"][0] for b in bins])
                hi = np.asarray([b[endpoint]["ci95"][1] for b in bins]) - y
                ax.errorbar(x, y, yerr=[lo, hi], color=color, marker=marker, lw=1.25, capsize=2.5, label=label)
            rho = dose["models"][model][domain]["statistics"]["metrics"]["margin_spearman"]["estimate"]
            ax.axhline(0, color="#333333", lw=.75)
            ax.set_title(f"{DISPLAY[model]} · {DISPLAY[domain]}\nmargin Spearman={rho:+.3f}")
            ax.set_xticks(x)
            ax.grid(axis="y", alpha=.25)
            if col == 0:
                ax.set_ylabel("delta-log-PPL (delta NLL)")
            if row == 1:
                ax.set_xlabel("ordered bin (1 strongest → 8 weakest)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Frozen k=3 conjunction: eight-bin group-level dose response", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, root / "figures/dose_response")


def objective_figure(root: Path, objective: dict) -> None:
    order = ("ce_natural", "kl_natural", "conjunction", "ce_matched_k", "kl_matched_k", "random_matched_k")
    labels = ("CE natural", "KL natural", "CE∧KL", "CE matched", "KL matched", "Random matched")
    colors = ("#64748b", "#64748b", "#0f766e", "#2563eb", "#7c3aed", "#9ca3af")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True)
    for row, domain in enumerate(DOMAINS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            policies = objective["models"][model][domain]["policies"]
            vals = np.asarray([policies[p]["delta_vs_four_over_six"]["estimate"] for p in order])
            lo = vals - np.asarray([policies[p]["delta_vs_four_over_six"]["ci95"][0] for p in order])
            hi = np.asarray([policies[p]["delta_vs_four_over_six"]["ci95"][1] for p in order]) - vals
            x = np.arange(len(order))
            ax.bar(x, vals, color=colors, alpha=.86)
            ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor="#111827", capsize=2.5, lw=.9)
            ax.axhline(0, color="#333333", lw=.75)
            ax.axvline(2.5, color="#94a3b8", ls="--", lw=.8)
            ax.set_title(f"{DISPLAY[model]} · {DISPLAY[domain]}")
            ax.set_xticks(x, labels, rotation=32, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=.22)
            if col == 0:
                ax.set_ylabel("delta-log-PPL vs FourOverSix")
    fig.suptitle("Protocol-aligned N16K64 k=3 objective ablation\nleft: natural budgets; right: matched per-stratum budgets", y=1.04, fontsize=14)
    fig.tight_layout()
    save(fig, root / "figures/objective_ablation")


def veto_figure(root: Path, veto: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    items = (("actual_vs_conjunction", "Actual − conjunction", "#b91c1c"),
             ("matched_random_vs_conjunction", "Random − conjunction", "#64748b"),
             ("actual_vs_matched_random", "Actual − random", "#7c3aed"))
    for ax, domain in zip(axes, DOMAINS):
        corpus = veto["corpora"][domain]
        values = np.asarray([corpus[key]["estimate"] for key, _, _ in items])
        lo = values - np.asarray([corpus[key]["ci95"][0] for key, _, _ in items])
        hi = np.asarray([corpus[key]["ci95"][1] for key, _, _ in items]) - values
        x = np.arange(len(items))
        ax.bar(x, values, color=[color for _, _, color in items], alpha=.85)
        ax.errorbar(x, values, yerr=[lo, hi], fmt="none", ecolor="#111827", capsize=3)
        ax.axhline(0, color="#333333", lw=.75)
        ax.set_xticks(x, [label for _, label, _ in items], rotation=25, ha="right")
        ax.set_title(DISPLAY[domain])
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("delta-log-PPL (positive is worse)")
    fig.suptitle("Mistral CE-vetoed / KL-approved add-back completion", fontsize=14)
    fig.tight_layout()
    save(fig, root / "figures/mistral_veto_completion")


def statistical_report(protocol_sha: str, dose: dict, objective: dict, veto: dict) -> str:
    lines = ["# Statistical report", "", f"- Frozen protocol SHA-256: `{protocol_sha}`",
             "- Inference unit: paired natural document/article cluster.",
             "- Bootstrap: 10,000 deterministic replicates; finite Monte Carlo plus-one p-values.",
             "- Multiplicity: Holm correction within the frozen A-development, A-validation, B-development, B-validation, and C-Mistral families.",
             "- delta-log-PPL is delta NLL; relative PPL is `exp(delta_NLL)-1`.", ""]
    lines += ["## Dose-response statistics", "",
              "| Role | Model | Corpus | Metric | Estimate | 95% CI | p | Holm p |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for family, rows in dose["family_rows"].items():
        role = "validation" if family.endswith("validation") else "development"
        for row in rows:
            lines.append(f"| {role} | {row['model']} | {row['domain']} | {row['endpoint']} | {fmt(row['estimate'])} | {ci(row)} | {row['p_two_sided_plus_one']:.6g} | {row['holm_adjusted_p']:.6g} |")
    lines += ["", "## Standardized pooled dose-response sensitivity", "",
              "| Analysis | Slope | 95% CI | p | Models |", "|---|---:|---:|---:|---|"]
    for name, row in dose["pooled_standardized"].items():
        lines.append(f"| {name} | {fmt(row['estimate'])} | {ci(row)} | {row['p_two_sided_plus_one']:.6g} | {', '.join(row['models'])} |")
    lines += ["", "## Matched-budget objective contrasts", "",
              "| Role | Model | Corpus | Contrast | delta-log-PPL | 95% CI | Relative PPL | p | Holm p |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for family, rows in objective["family_rows"].items():
        role = "validation" if family.endswith("validation") else "development"
        for row in rows:
            lines.append(f"| {role} | {row['model']} | {row['domain']} | {row['endpoint']} | {fmt(row['estimate'])} | {ci(row)} | {row['relative_ppl_change']:+.4%} | {row['p_two_sided_plus_one']:.6g} | {row['holm_adjusted_p']:.6g} |")
    lines += ["", "## Mistral veto-completion family", "",
              "| Corpus | Endpoint | Estimate | 95% CI | p | Holm p |", "|---|---|---:|---:|---:|---:|"]
    for row in veto["family_rows"]:
        lines.append(f"| {row['domain']} | {row['endpoint']} | {fmt(row['estimate'])} | {ci(row)} | {row['p_two_sided_plus_one']:.6g} | {row['holm_adjusted_p']:.6g} |")
    lines += ["", "## Interpretation boundary", "",
              "Correlations across composition-matched tile groups are aggregate ranking evidence. They do not establish reliable individual-tile signs, magnitudes, calibration, or causality. Natural-threshold objective maps have unequal selected counts and are descriptive rule-level comparisons; only matched-K contrasts isolate objective information.", ""]
    return "\n".join(lines)


def primary_tables(dose: dict, objective: dict, veto: dict) -> str:
    lines = ["# Primary results tables", "", "## Eight-bin marginal dose response", "",
             "| Model | Corpus | Bin | Tiles | CE predicted sum | Mean combined margin | Marginal delta-log-PPL | 95% CI | Group-only delta-log-PPL |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for domain in DOMAINS:
            for row in dose["models"][model][domain]["bins"]:
                lines.append(f"| {model} | {domain} | {row['bin']} | {row['tiles']} | {row['sum_ce_mean']:+.7g} | {row['mean_combined_election_margin']:+.7g} | {fmt(row['full_context_marginal']['estimate'])} | {ci(row['full_context_marginal'])} | {fmt(row['group_only_vs_baseline']['estimate'])} |")
    lines += ["", "## Dose-response rank summaries", "",
              "| Model | Corpus | CE Spearman | 95% CI | Margin Spearman | 95% CI | CE calibration slope | R² | Monotone adjacencies |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for domain in DOMAINS:
            metrics = dose["models"][model][domain]["statistics"]["metrics"]
            lines.append(f"| {model} | {domain} | {metrics['ce_spearman']['estimate']:+.4f} | {ci(metrics['ce_spearman'])} | {metrics['margin_spearman']['estimate']:+.4f} | {ci(metrics['margin_spearman'])} | {metrics['ce_slope']['estimate']:+.4f} | {metrics['ce_r_squared']['estimate']:.4f} | {metrics['ordered_monotonicity']['adjacent_nonnegative']}/7 |")
    lines += ["", "## Objective maps versus FourOverSix", "",
              "| Model | Corpus | Rule | Selected tiles | Absolute PPL | delta-log-PPL | 95% CI | Relative PPL |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for domain in DOMAINS:
            for name, row in objective["models"][model][domain]["policies"].items():
                delta = row["delta_vs_four_over_six"]
                lines.append(f"| {model} | {domain} | {name} | {row['selected_tiles']} | {row['absolute_ppl']:.7f} | {fmt(delta['estimate'])} | {ci(delta)} | {delta['relative_ppl_change']:+.4%} |")
    lines += ["", "## Mistral CE-vetoed / KL-approved completion", "",
              "| Corpus | Contrast | delta-log-PPL | 95% CI | Relative PPL | Holm p |", "|---|---|---:|---:|---:|---:|"]
    for domain in DOMAINS:
        corpus = veto["corpora"][domain]
        for label, key in (("actual − conjunction", "actual_vs_conjunction"), ("actual − matched random", "actual_vs_matched_random")):
            row = corpus[key]
            lines.append(f"| {domain} | {label} | {fmt(row['estimate'])} | {ci(row)} | {row['relative_ppl_change']:+.4%} | {row['holm_adjusted_p']:.6g} |")
        for endpoint, row in corpus["tails"].items():
            lines.append(f"| {domain} | {endpoint} | {fmt(row['estimate'])} | {ci(row)} | NA | {row['holm_adjusted_p']:.6g} |")
    return "\n".join(lines) + "\n"


def verdict(dose: dict, objective: dict, veto: dict) -> str:
    dc, oc, vc = dose["classification"], objective["classification"], veto["classification"]
    lines = ["# Follow-up verdict", "", "## Fixed-gate summary", "",
             "| Question | Classification | Gate passed? |", "|---|---|---:|",
             f"| Group-level eight-bin dose response | {dc['classification']} | {'yes' if dc['passed'] else 'no'} |",
             f"| Matched-budget conjunction over both single objectives | {oc['classification']} | {'yes' if oc['passed'] else 'no'} |",
             f"| Mistral CE-vetoed/KL-approved symmetry | {vc['classification']} | {'yes' if vc['passed'] else 'no'} |", "",
             "## Direct conclusions", ""]
    lines += [f"- Dose response: {dc['gate']}. Result: **{dc['classification']}**.",
              f"- Objective ablation: {oc['gate']}. Result: **{oc['classification']}**.",
              f"- Mistral veto completion: {vc['gate']}. Result: **{vc['classification']}**.", ""]
    lines += ["## Per-model evidence", ""]
    for model in MODELS:
        pieces = []
        for domain in DOMAINS:
            metrics = dose["models"][model][domain]["statistics"]["metrics"]
            ce = metrics["ce_spearman"]
            margin = metrics["margin_spearman"]
            ce_vs = objective["models"][model][domain]["matched_budget_contrasts"]["ce_matched_k_minus_conjunction"]
            kl_vs = objective["models"][model][domain]["matched_budget_contrasts"]["kl_matched_k_minus_conjunction"]
            pieces.append(f"{domain}: dose rho_CE={ce['estimate']:+.3f}, rho_margin={margin['estimate']:+.3f}; CE-matched−conjunction={ce_vs['estimate']:+.6f}, KL-matched−conjunction={kl_vs['estimate']:+.6f}")
        lines.append(f"- **{DISPLAY[model]}:** " + "; ".join(pieces) + ".")
    leave = dose["pooled_standardized"]
    lines += ["", "## Qwen outlier sensitivity", "",
              f"All-model standardized CE slope: {leave['ce_all_models']['estimate']:+.4f} {ci(leave['ce_all_models'])}; leave-Qwen3-4B-out: {leave['ce_leave_qwen4b_out']['estimate']:+.4f} {ci(leave['ce_leave_qwen4b_out'])}.",
              f"All-model standardized margin slope: {leave['margin_all_models']['estimate']:+.4f} {ci(leave['margin_all_models'])}; leave-Qwen3-4B-out: {leave['margin_leave_qwen4b_out']['estimate']:+.4f} {ci(leave['margin_leave_qwen4b_out'])}.",
              "Qwen3-4B remains in every primary per-model table; this sensitivity does not erase or downweight its result.", "",
              "## Claims that remain prohibited", "",
              "- reliable individual-tile causal signs, magnitudes, calibration, or causality;",
              "- k=3 as a simultaneous per-tile confidence guarantee;",
              "- universal CE or KL superiority from these models/corpora;",
              "- native FP4/E0M3 Tensor Core execution;",
              "- latency, speedup, runtime-overhead, area, power, or Blackwell-performance claims from fake quantization.", "",
              "All results are fake-quantized/dequantized BF16 quality evaluation. Mistral remains one-shot analysis validation, not a fully independent confirmatory family.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--protocol-sha256", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    dose = load(root / "DOSE_RESPONSE_RESULTS.json")
    objective = load(root / "OBJECTIVE_ABLATION_RESULTS.json")
    veto = load(root / "MISTRAL_VETO_COMPLETION.json")
    if any(doc["protocol_sha256"] != args.protocol_sha256 for doc in (dose, objective, veto)):
        raise SystemExit("result/protocol hash mismatch")
    (root / "figures").mkdir(exist_ok=True)
    dose_figure(root, dose)
    objective_figure(root, objective)
    veto_figure(root, veto)
    (root / "STATISTICAL_REPORT.md").write_text(statistical_report(args.protocol_sha256, dose, objective, veto) + "\n")
    (root / "PRIMARY_RESULTS_TABLES.md").write_text(primary_tables(dose, objective, veto))
    (root / "FOLLOWUP_VERDICT.md").write_text(verdict(dose, objective, veto))
    recommendation = ["# Next-step recommendation", "",
                      "Do not launch another broad selector or k search. Preserve the frozen per-model heterogeneity and null/negative findings.", "",
                      "If a follow-up is warranted, replicate only a predeclared contrast that passed its frozen gate, on an untouched family and independently frozen windows. Treat native-kernel execution and overhead as a separate systems project; this fake-quantization campaign cannot answer them.", ""]
    (root / "NEXT_STEP_RECOMMENDATION.md").write_text("\n".join(recommendation))
    print(json.dumps({"reports": ["STATISTICAL_REPORT.md", "PRIMARY_RESULTS_TABLES.md", "FOLLOWUP_VERDICT.md", "NEXT_STEP_RECOMMENDATION.md"],
                      "figures": ["dose_response", "objective_ablation", "mistral_veto_completion"]}, sort_keys=True))


if __name__ == "__main__":
    main()

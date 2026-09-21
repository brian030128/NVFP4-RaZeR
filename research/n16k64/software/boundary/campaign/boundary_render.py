"""Render frozen boundary/corruption results into figures and reviewer-facing reports."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from campaign.boundary_common import DOMAINS, MODELS, atomic_json, sha256_file


MODEL_LABEL = {"llama8b": "Llama-3.1-8B", "qwen4b": "Qwen3-4B", "mistral7b": "Mistral-7B-v0.3"}
DOMAIN_LABEL = {"c4": "C4", "wiki": "WikiText"}


def effect(row: dict) -> str:
    lo, hi = row["ci95"]
    return f"{row['estimate']:+.6f} [{lo:+.6f}, {hi:+.6f}]"


def save_figure(fig: plt.Figure, root: Path, stem: str) -> dict:
    png, svg = root / "figures" / f"{stem}.png", root / "figures" / f"{stem}.svg"
    png.parent.mkdir(exist_ok=True)
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"png": {"path": str(png), "sha256": sha256_file(png), "bytes": png.stat().st_size},
            "svg": {"path": str(svg), "sha256": sha256_file(svg), "bytes": svg.stat().st_size}}


def boundary_figure(root: Path, results: dict) -> dict:
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 12.6), sharey=False)
    colors = {"selected": "#1769aa", "rejected": "#d1495b"}
    markers = {"selected": "o", "rejected": "s"}
    for i, model in enumerate(MODELS):
        for j, domain in enumerate(DOMAINS):
            ax = axes[i, j]
            rows = results["models"][model][domain]["bands"]
            for partition in range(1, 5):
                part = [row for row in rows if row["partition"] == partition]
                ordered = sorted(part, key=lambda row: row["mean_kappa"])
                ax.plot([row["mean_kappa"] for row in ordered],
                        [row["delta_vs_four_over_six"]["estimate"] for row in ordered],
                        color="#7a7a7a", alpha=0.26, lw=0.9)
                for side in ("selected", "rejected"):
                    subset = [row for row in part if row["side"] == side]
                    x = np.asarray([row["mean_kappa"] for row in subset])
                    y = np.asarray([row["delta_vs_four_over_six"]["estimate"] for row in subset])
                    lo = np.asarray([row["delta_vs_four_over_six"]["ci95"][0] for row in subset])
                    hi = np.asarray([row["delta_vs_four_over_six"]["ci95"][1] for row in subset])
                    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt=markers[side], ms=4.2,
                                color=colors[side], ecolor=colors[side], alpha=0.78, capsize=1.8,
                                label=side.capitalize() if partition == 1 else None)
            ax.axhline(0, color="black", lw=0.8, alpha=0.55)
            ax.axvline(3, color="#555", lw=0.9, ls="--", alpha=0.7, label="k=3 boundary" if i == 0 and j == 0 else None)
            ax.set_title(f"{MODEL_LABEL[model]} — {DOMAIN_LABEL[domain]}")
            ax.set_xlabel("Mean critical-k (κ) in composition-matched band")
            ax.set_ylabel("ΔNLL vs FourOverSix (lower is better)")
            ax.grid(alpha=0.18)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Critical-k boundary response (four construction partitions)", fontsize=15, y=1.01)
    fig.tight_layout()
    return save_figure(fig, root, "critical_k_boundary_response")


def corruption_figure(root: Path, results: dict) -> dict:
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 12.6), sharex=True)
    pool_colors = ["#1769aa", "#2a9d8f", "#e9c46a", "#e76f51"]
    for i, model in enumerate(MODELS):
        for j, domain in enumerate(DOMAINS):
            ax = axes[i, j]
            panel = results["models"][model][domain]
            for pool in range(1, 5):
                rows = sorted((row for row in panel["levels"] if row["pool"] == pool), key=lambda row: row["p"])
                x = np.asarray([row["p"] for row in rows])
                y = np.asarray([row.get("delta_vs_p0", {"estimate": 0})["estimate"] for row in rows])
                lo = np.asarray([row.get("delta_vs_p0", {"ci95": [0, 0]})["ci95"][0] for row in rows])
                hi = np.asarray([row.get("delta_vs_p0", {"ci95": [0, 0]})["ci95"][1] for row in rows])
                ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", ms=4.2,
                            lw=1.35, elinewidth=0.75, capsize=1.7, color=pool_colors[pool - 1],
                            alpha=0.88, label=f"Near pool {pool}")
            random = [row["delta_vs_p0"]["estimate"] for row in panel["matched_random_p050"]]
            ax.scatter([0.5] * 4, random, marker="x", s=42, lw=1.5, color="#6f2dbd",
                       label="Matched-random p=.50")
            ax.axhline(0, color="black", lw=0.8, alpha=0.55)
            ax.set_title(f"{MODEL_LABEL[model]} — {DOMAIN_LABEL[domain]}")
            ax.set_xlabel("Imposed replacement fraction p")
            ax.set_ylabel("ΔNLL vs exact conjunction p=0")
            ax.grid(alpha=0.18)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Full-map corruption across four distinct score-balanced pools", fontsize=15, y=1.01)
    fig.tight_layout()
    return save_figure(fig, root, "full_map_corruption")


def write_reports(root: Path, boundary: dict, corruption: dict, coverage: dict, power: dict) -> None:
    deviations = json.loads((root / "PROTOCOL_DEVIATIONS.json").read_text())
    tables = ["# Primary results tables", "", "All effects are delta NLL (delta log perplexity); brackets are paired 95% cluster-bootstrap intervals.", "",
              "## Outcome-blind common support", "", "| Model | Selected tiles | Retained | Coverage | Tiles/global band |",
              "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        row = coverage["candidates"][str(coverage["selected_B"])][model]
        tables.append(f"| {MODEL_LABEL[model]} | {row['selected_tiles']:,} | {row['retained_selected_tiles']:,} | {row['selected_common_support_coverage']:.3f} | {row['tiles_per_global_band']:,} |")
    tables += ["", "## Boundary primary endpoints", "",
               "| Model | Corpus | beta_kappa | Selected−rejected | Weakest selected−nearest rejected | Power status (slope) |",
               "|---|---|---:|---:|---:|---|"]
    for model in MODELS:
        for domain in DOMAINS:
            primary = boundary["models"][model][domain]["primary"]
            tables.append(f"| {MODEL_LABEL[model]} | {DOMAIN_LABEL[domain]} | {effect(primary['beta_kappa'])} | {effect(primary['selected_minus_rejected'])} | {effect(primary['weakest_selected_minus_nearest_rejected'])} | {primary['beta_kappa']['power_status']} |")
    tables += ["", "## Full-map corruption primary endpoints", "",
               "| Model | Corpus | beta_p | p=1−p=0 | near p=.50−p=0 | near−random at p=.50 | Slope power |",
               "|---|---|---:|---:|---:|---:|---|"]
    for model in MODELS:
        for domain in DOMAINS:
            panel = corruption["models"][model][domain]
            primary = panel["primary"]
            tables.append(f"| {MODEL_LABEL[model]} | {DOMAIN_LABEL[domain]} | {effect(primary['beta_p'])} | {effect(primary['p100_minus_p0'])} | {effect(primary['p050_near_minus_p0'])} | {effect(panel['p050_near_minus_matched_random'])} | {primary['beta_p']['power_status']} |")
    tables += ["", "## Standardized pooled sensitivities", "",
               "| Experiment | All models | Leave Qwen3-4B out |", "|---|---:|---:|",
               f"| Boundary beta_kappa | {effect(boundary['pooled_standardized']['all_models'])} | {effect(boundary['pooled_standardized']['leave_qwen4b_out'])} |",
               f"| Corruption beta_p | {effect(corruption['pooled_standardized']['all_models'])} | {effect(corruption['pooled_standardized']['leave_qwen4b_out'])} |", ""]
    (root / "PRIMARY_RESULTS_TABLES.md").write_text("\n".join(tables))

    stats = ["# Statistical report", "", "## Frozen design", "",
             f"The outcome-blind coverage gate tested B=8, 6, then 4 and selected B={coverage['selected_B']} with {coverage['partitions']} construction partitions. All 32 boundary maps per model use group-only maps against the same FourOverSix baseline. Corruption uses four disjoint near-boundary pools and nested p={0,.10,.25,.50,.75,1.00} maps with exact module quotas.", "",
             "Natural inference units are C4 documents and WikiText articles. Every result uses 10,000 deterministic paired-cluster bootstrap draws. P-values use the finite plus-one rule and primary endpoints are Holm-adjusted in the four frozen 12/6/12/6 families. Correlation intervals are bounded to the valid parameter domain.", "",
             "## Power qualification", "",
             "Power was computed before new outcomes from centered, verified prior paired arrays. Existing effect locations were removed; only their paired covariance/noise was reused. Endpoint status did not select arms.", "",
             "| Model | Corpus | Boundary slope | Selected−rejected | Weak boundary | Corruption slope | p=.50 | p=1 |",
             "|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for domain in DOMAINS:
            e = power["models"][model][domain]["endpoints"]
            stats.append(f"| {MODEL_LABEL[model]} | {DOMAIN_LABEL[domain]} | {e['boundary_trend_slope_end_to_end']['status']} | {e['selected_minus_rejected_aggregate']['status']} | {e['weakest_selected_minus_nearest_rejected']['status']} | {e['corruption_slope_end_to_end']['status']} | {e['p050_minus_p0']['status']} | {e['p100_minus_p0']['status']} |")
    stats += ["", "### Numeric power and MDE", "",
              "| Model | Corpus | Endpoint | Power @ .0010 | Power @ .0025 | MDE80 | MDE90 | Status |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for model in MODELS:
        for domain in DOMAINS:
            for endpoint, row in power["models"][model][domain]["endpoints"].items():
                stats.append(
                    f"| {MODEL_LABEL[model]} | {DOMAIN_LABEL[domain]} | `{endpoint}` | "
                    f"{row['power']['delta_nll_0.0010']:.3f} | {row['power']['delta_nll_0.0025']:.3f} | "
                    f"{row['mde_80pct']:.6f} | {row['mde_90pct']:.6f} | {row['status']} |")
    stats += ["", "## Frozen multiplicity families", ""]
    for payload, prefix in ((boundary, "A"), (corruption, "B")):
        for name, rows in payload["holm_families"].items():
            stats.append(f"- `{name}`: {len(rows)} endpoints; {sum(bool(row.get('holm_reject_0p05')) for row in rows)} Holm rejections.")
    stats += ["", "## Interpretation", "",
              "Construction partitions and corruption pools are sensitivity/replicate factors, not independent model replications. Delta NLL is delta log PPL; relative PPL change is exp(delta NLL)-1. Mistral is a one-shot analysis validation, and Qwen3-4B remains the predeclared high-response case.", "",
              "The continuous boundary slopes and selected-versus-rejected aggregate contrasts favor the frozen ranking, but the weakest-selected versus nearest-rejected contrast is heterogeneous: it is positive for Llama on both corpora, negative for Qwen on both, negative for Mistral C4, and near zero for Mistral WikiText. The evidence therefore supports an aggregate ordering signal, not a sharp causal discontinuity at kappa=3.", "",
              "All six corruption slopes and p=1 contrasts are positive. At p=.50, score-near rejected replacements are less harmful than matched-random rejected replacements in all six point estimates, although two intervals include zero. This is ranking evidence under imposed stress, not an estimate of a real selector error rate.", ""]
    stats += ["## Operational protocol conformance", "",
              "The 24-hour target and hour-20 GPU-launch cutoff were not met. Required Llama and Qwen retries began after the frozen cutoff because their original runs were invalidated on foreign co-tenancy and workflow continuation occurred later. `PROTOCOL_DEVIATIONS.json` records exact timestamps and safeguards. No outcome-driven scientific definition changed, but the late launch remains a submission-risk limitation.", "",
              "Two CPU analysis attempts also failed before successful inference: attempt1 used invalid relative bind mounts, and attempt2 stopped before reading outcome values because Qwen's prior report carried extra screening-provenance metadata that changed an overbroad container manifest hash. `ANALYSIS_CORRECTIONS.json` freezes the outcome-blind correction: exact model, revision, token-window hashes, natural-cluster assignments, and token accounting must match directly. The original frozen script is preserved byte-identical, and no map, endpoint, seed, family, or classification rule changed.", ""]
    (root / "STATISTICAL_REPORT.md").write_text("\n".join(stats))

    bclass, cclass = boundary["classification"], corruption["classification"]
    verdict = ["# Boundary/corruption mechanism verdict", "",
               f"**Boundary classification:** `{bclass['classification']}` (frozen pattern gate passed: {str(bclass['frozen_gate_passed']).lower()}).",
               "", f"**Corruption classification:** `{cclass['classification']}` (frozen pattern gate passed: {str(cclass['frozen_gate_passed']).lower()}).", "",
               "The evidence addresses aggregate, composition-matched tile groups and imposed full-map stress. It does not validate individual-tile causal signs, magnitudes, or calibration.", "",
               "## Model-specific reading", ""]
    for model in MODELS:
        bsign = [boundary["models"][model][domain]["primary"]["beta_kappa"]["estimate"] for domain in DOMAINS]
        csign = [corruption["models"][model][domain]["primary"]["beta_p"]["estimate"] for domain in DOMAINS]
        weak = [boundary["models"][model][domain]["primary"]["weakest_selected_minus_nearest_rejected"]["estimate"] for domain in DOMAINS]
        verdict.append(f"- **{MODEL_LABEL[model]}:** boundary slopes {bsign[0]:+.6f} (C4), {bsign[1]:+.6f} (WikiText); weakest-selected minus nearest-rejected {weak[0]:+.6f}, {weak[1]:+.6f}; corruption slopes {csign[0]:+.6f}, {csign[1]:+.6f}. Power labels and confidence intervals in the primary table determine whether these directions are inferential or descriptive.")
    verdict += ["", "## Pooled sensitivity", "",
                f"Boundary standardized slope: all models {effect(boundary['pooled_standardized']['all_models'])}; leave-Qwen {effect(boundary['pooled_standardized']['leave_qwen4b_out'])}.", "",
                f"Corruption standardized slope: all models {effect(corruption['pooled_standardized']['all_models'])}; leave-Qwen {effect(corruption['pooled_standardized']['leave_qwen4b_out'])}.", "",
                "## Candid interpretation", "",
                "Both frozen pattern gates pass, but both classifications are power-limited. Continuous aggregate ranking and imposed corruption robustness are supported across models and corpora, including after leaving Qwen3-4B out. A clean local threshold discontinuity is not supported uniformly: the weakest-selected versus nearest-rejected contrast reverses sign for Llama and is heterogeneous across validation panels. This is compatible with useful group-level ranking plus noisy local/tile-level effects; it is not evidence that every kappa>3 tile is beneficial.", "",
                "## Operational limitation", "",
                "The campaign did not satisfy its hour-20 launch cutoff or 24-hour wall-clock target. Late Llama/Qwen runs were frozen required retries after co-tenancy invalidations, not outcome-selected additions. A separately frozen, outcome-blind analysis-plumbing correction was also required after two failed CPU attempts. These limitations must be disclosed with any use of the results; see `PROTOCOL_DEVIATIONS.json` and `ANALYSIS_CORRECTIONS.json`.", "",
                "## Claims that remain prohibited", "",
                "Reliable individual-tile causality or finite-effect calibration; a true wrong-tile percentage; every selected tile being helpful or every rejected tile harmful; k=3 as a simultaneous confidence guarantee; universal selector optimality; all-model generalization; native FP4/E0M3 execution; and latency, throughput, speedup, overhead, area, power, or Blackwell claims.", ""]
    (root / "BOUNDARY_CORRUPTION_VERDICT.md").write_text("\n".join(verdict))
    (root / "NEXT_STEP_RECOMMENDATION.md").write_text(
        "# Next-step recommendation\n\nDo not tune a replacement selector from these outcomes. Use the frozen group-level and corruption results only to decide whether a separately preregistered replication is warranted. Any future work should add genuinely new model families or direct tile-intervention validation, retain natural-cluster pairing, and preserve the current negative/heterogeneous results. Native-kernel performance claims require a separate hardware study.\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    root = Path(args.campaign_root)
    boundary = json.loads((root / "BOUNDARY_RESULTS.json").read_text())
    corruption = json.loads((root / "CORRUPTION_RESULTS.json").read_text())
    coverage = json.loads((root / "COVERAGE_GATE.json").read_text())
    power = json.loads((root / "POWER_ANALYSIS.json").read_text())
    figures = {"critical_k_boundary_response": boundary_figure(root, boundary),
               "full_map_corruption": corruption_figure(root, corruption)}
    write_reports(root, boundary, corruption, coverage, power)
    report = {"schema": "mixfp4-boundary-corruption-render-report/v1", "figures": figures,
              "reports": {name: {"sha256": sha256_file(root / name), "bytes": (root / name).stat().st_size}
                          for name in ("STATISTICAL_REPORT.md", "PRIMARY_RESULTS_TABLES.md",
                                       "BOUNDARY_CORRUPTION_VERDICT.md", "NEXT_STEP_RECOMMENDATION.md")}}
    atomic_json(root / "analysis/RENDER_REPORT.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

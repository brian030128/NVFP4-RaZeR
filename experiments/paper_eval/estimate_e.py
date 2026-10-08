"""Part E's cost estimate (lm-eval log-likelihood tasks, 6 models x 6 policies; nothing is run): from the FlipQuant
paper's lm-eval runs (paper step 04, run_lmeval_deploy.py, lm-eval 0.4.11 HFLM, batch 16, Qwen3.8-27B batch 8), which
timed MMLU 5-shot, ARC-Challenge, ARC-Easy, HellaSwag and PIQA per task for Llama-3.1-8B, Mistral-7B, Phi-4 and
Qwen3.8-27B (the last in the fallback env). -> results/paper_eval/e_estimate/{E_ESTIMATE.md, e_estimate.json}

- Models that the paper did not time take a timed model's seconds times a factor (assumed, not measured): Qwen3-8B =
  Llama-3.1-8B x 1.0, Nemotron-Nano-9B-v2 = x 1.15, Qwen3-1.7B = x 0.4. Qwen3.8-27B in n16k64-fast = the fallback
  run's seconds / s, s = 1.6 (range 1.0 - 2.2: the BF16 1x2048 prefill was 2.2x faster in n16k64-fast).
- The appendix tasks, from the timed tasks by their request and token volumes: MMLU-Pro (5-shot, 12,032 questions,
  one forward per question since lm-eval scores one-token continuations from one context) = MMLU x r, r = 1.7 (range
  1.2 - 2.5; its 5-shot prompts carry 10 options per question); WinoGrande = PIQA x 0.7; BoolQ = ARC-Easy x 1.0;
  LAMBADA = ARC-Easy x 0.8.
- Per process: model load and native install, 1.5 min (Qwen3.8-27B 3 min).

    python estimate_e.py
"""
import glob
import json
from pathlib import Path

PAPER = Path("/home/dev/n16k64_campaign/paper/lmeval")
OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval" / "e_estimate"
TIMED = ("mmlu", "arc_challenge", "arc_easy", "hellaswag", "piqa")
SOURCE = {"qwen3-1.7b": ("llama8b", 0.4), "qwen3-8b": ("llama8b", 1.0), "mistral-7b": ("mistral7b", 1.0),
          "nemotron-nano-9b-v2": ("llama8b", 1.15), "phi4-14b": ("phi4", 1.0), "qwen3.8-27b": ("qwen27b", None)}
QWEN27_SPEEDUP = {"low": 2.2, "mid": 1.6, "high": 1.0}           # fewer hours <- larger speedup
MMLU_PRO = {"low": 1.2, "mid": 1.7, "high": 2.5}
OVERHEAD_MIN = {"qwen3.8-27b": 3.0}


def measured():
    """{paper model: {"bf16" | "fp4": {task: mean seconds}}}"""
    out = {}
    for f in sorted(glob.glob(str(PAPER / "*" / "*" / "report.json"))):
        model, pol = Path(f).parts[-3:-1]
        ev = next(iter(json.load(open(f))["evaluations"].values()))
        cls = "bf16" if pol == "bf16" else "fp4"
        for t in TIMED:
            out.setdefault(model, {}).setdefault(cls, {}).setdefault(t, []).append(ev["tasks"][t]["seconds"])
    return {m: {c: {t: sum(v) / len(v) for t, v in d.items()} for c, d in mm.items()} for m, mm in out.items()}


def suite_seconds(s, r_pro, with_pro=True):
    extra = dict(winogrande=0.7 * s["piqa"], boolq=1.0 * s["arc_easy"], lambada_openai=0.8 * s["arc_easy"])
    if with_pro:
        extra["mmlu_pro_loglik"] = r_pro * s["mmlu"]
    return dict(s, **extra)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meas = measured()
    rec = dict(measured_seconds=meas, assumptions=dict(source=SOURCE, qwen27_fast_env_speedup=QWEN27_SPEEDUP,
                                                       mmlu_pro_over_mmlu=MMLU_PRO, overhead_min=OVERHEAD_MIN),
               hours={})
    for case in ("low", "mid", "high"):
        for with_pro in (True, False):
            key = f"{case}{'' if with_pro else '_no_mmlu_pro'}"
            per_model = {}
            for model, (src, k) in SOURCE.items():
                if k is None:
                    k = 1 / QWEN27_SPEEDUP[case]
                tot = 0.0
                for cls, n in (("bf16", 1), ("fp4", 5)):
                    s = {t: v * k for t, v in meas[src][cls].items()}
                    sec = sum(suite_seconds(s, MMLU_PRO[case], with_pro).values())
                    tot += n * (sec / 3600 + OVERHEAD_MIN.get(model, 1.5) / 60)
                per_model[model] = tot
            rec["hours"][key] = dict(per_model, total=sum(per_model.values()))
    md = ["# Part E: cost estimate (nothing run)", "",
          "lm-eval 0.4.11 log-likelihood suite (MMLU 5-shot, MMLU-Pro 5-shot by log-likelihood, ARC-C/E, HellaSwag, "
          "PIQA, WinoGrande, BoolQ, LAMBADA), 6 models x 6 policies (BF16, NVFP4, FourOverSix, FlipQuant 8x64 / 16x64 / "
          "256x64), from the FlipQuant paper's timed lm-eval runs; assumptions in `experiments/paper_eval/estimate_e.py`.",
          "", "| model | hours (low / mid / high) | without MMLU-Pro (mid) |", "|---|---:|---:|"]
    for model in list(SOURCE) + ["total"]:
        h = [rec["hours"][c][model] for c in ("low", "mid", "high")]
        md.append(f"| {model} | {h[0]:.1f} / {h[1]:.1f} / {h[2]:.1f} | {rec['hours']['mid_no_mmlu_pro'][model]:.1f} |")
    md += ["", "Measured basis (paper step 04, seconds per task, mean over its FP4 policies; BF16 in brackets):", "",
           "| paper model | " + " | ".join(TIMED) + " |", "|---|" + "---:|" * len(TIMED)]
    for m, d in meas.items():
        md.append(f"| {m} | " + " | ".join(f"{d['fp4'][t]:.0f} ({d['bf16'][t]:.0f})" for t in TIMED) + " |")
    (OUT / "e_estimate.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "E_ESTIMATE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

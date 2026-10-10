"""Part N reports (PROTOCOL.md part N): tab:ptq's GSM8K column with lm-eval's gsm8k_llama, thinking off.

    python report_gsm8k_lmeval.py pilot    # N0: results/paper_eval/ptq_gsm8k_lmeval/pilot/{PILOT.md, pilot.json, ...}
    python report_gsm8k_lmeval.py report   # N1: results/paper_eval/ptq_gsm8k_lmeval/{GSM8K_LMEVAL.md, gsm8k_lmeval.json,
                                           #     samples/}, checks N1-N5

Accuracy is lm-eval's exact_match per filter (strict_match, flexible_extract), ± 2 SE binomial (sqrt(p(1-p)/n)).
Paired comparisons are per problem and filter: A - B ± 2 SE of the per-problem difference, the discordant counts and
McNemar's exact two-sided p. A generation "used the budget" when it has no end-of-sequence token within its 1024
tokens (evaluation.downstream --gen-lengths). A filter "cannot extract" when lm-eval's regex filter returns
'[invalid]'.
"""
import gzip
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_gsm8k as RM  # noqa: E402
import run_gsm8k as M  # noqa: E402
import run_gsm8k_lmeval as NL  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval" / "ptq_gsm8k_lmeval"
TASK = NL.TASK
FILTERS = ("strict_match", "flexible_extract")
THINK = ("<think>", "</think>")
N_DOCS = 1319
#: check exceptions the coordinator accepted (config, doc) -> note
ACCEPTED = {("qwen3.8-27b/hadamard_fo6", "262"): "accepted by the coordinator; thinking was off, the tag carries no "
                                                  "reasoning, the accuracy is unaffected"}


def load(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return r if r.get("status") == "complete" else None


def per_filter(ex, flt):
    """{doc: correct} for one filter."""
    return {d: bool(e[flt]["exact_match"]) for d, e in ex.items() if flt in e}


def summary(rep):
    t = rep["tasks"][TASK]
    ex = t["examples"]
    out = dict(n=len(ex), batch=rep["batch_size"], seconds=t["seconds"], peak_gpu_allocated_gib=t["peak_gpu_allocated_gib"],
               generate_seconds=t["model_timing"].get("generate_until_seconds"))
    for f in FILTERS:
        c = per_filter(ex, f)
        p = sum(c.values()) / len(c)
        out[f] = dict(accuracy=p, two_se=2 * math.sqrt(p * (1 - p) / len(c)), correct=sum(c.values()), n=len(c),
                      lm_eval=t["metrics"].get(f"exact_match,{f}"), lm_eval_stderr=t["metrics"].get(f"exact_match_stderr,{f}"),
                      cannot_extract=sum(1 for e in ex.values() if e.get(f, {}).get("response") == "[invalid]"))
    gen = [e.get("generated") or {} for e in ex.values()]
    toks = [g["tokens"] for g in gen if "tokens" in g]
    out["generated"] = dict(recorded=len(toks), mean_tokens=statistics.mean(toks) if toks else None,
                            max_tokens=max(toks) if toks else None,
                            used_budget=sum(1 for g in gen if g and not g["eos"] and g["tokens"] >= g["budget"]),
                            budget=max((g["budget"] for g in gen if g), default=None))
    out["think_markers"] = sum(1 for e in ex.values() if any(m in (e.get("raw") or "") for m in THINK))
    return out


def h_record(model, method, fmt):
    return RM.h_record(model, method, fmt)


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    rec = dict(task=TASK, configs={}, paired={}, checks=dict(N1={}, N2={}, N3={}, N4={}, N5={}))
    ex_all = {}
    for model in M.MODELS:
        for method in M.METHODS:
            for fmt in M.FMTS:
                path = NL.ROOT / model / f"{method}_{fmt}.json"
                rep = load(path)
                if rep is None:
                    continue
                key = f"{model}/{method}_{fmt}"
                s = summary(rep)
                rec["configs"][key] = s
                ex = rep["tasks"][TASK]["examples"]
                ex_all[(model, method, fmt)] = ex
                rec["checks"]["N1"][key] = s["think_markers"] == 0
                rec["checks"]["N4"][key] = len(ex) == N_DOCS and all(all(f in e for f in FILTERS) for e in ex.values())
                cov = rep.get("coverage") or {}
                rec["checks"]["N5"][key] = (cov.get("native_called") == cov.get("native") and not cov.get("remaining_bf16_linears_called")
                                            and bool(rep["tasks"][TASK].get("native_gemm", {}).get("every_linear_every_forward")))
                h = h_record(model, method, fmt)
                pol = rep.get("policy") or {}
                if method == "gptq":
                    a, b = (pol.get("ptq") or {}).get("codes_sha256"), (h.get("ptq") or {}).get("codes_sha256")
                    rec["checks"]["N2"][key] = dict(gsm8k=a, ppl=b, equal=a is not None and a == b)
                if fmt == "fq-16x64":
                    mine, theirs = pol.get("map") or {}, h.get("map") or {}
                    ok = all(mine.get(k) == theirs.get(k) and mine.get(k) is not None for k in ("path", "modules", "e0m3_tiles"))
                    rec["checks"]["N3"][key] = dict(path=mine.get("path"), e0m3_tiles=mine.get("e0m3_tiles"), equal_to_h=ok)
                src = path.with_suffix("").with_name(path.stem + f".{TASK}.jsonl.gz")
                if src.exists():
                    shutil.copyfile(src, OUT / "samples" / f"{model}__{method}_{fmt}.{TASK}.jsonl.gz")
    for model in M.MODELS:
        pairs = [(m, f, m, "fo6") for m in M.METHODS for f in ("nvfp4", "fq-16x64")]
        pairs += [(m, "fq-16x64", "rtn", "fq-16x64") for m in ("gptq", "hadamard")]
        for ma, fa, mb, fb in pairs:
            a, b = ex_all.get((model, ma, fa)), ex_all.get((model, mb, fb))
            if a and b:
                rec["paired"][f"{model}: {ma} {fa} vs {mb} {fb}"] = {
                    f: RM.paired({d: dict(correct=v) for d, v in per_filter(a, f).items()},
                                 {d: dict(correct=v) for d, v in per_filter(b, f).items()}) for f in FILTERS}
    want = {"N1": 18, "N4": 18, "N5": 18, "N2": 6, "N3": 6}
    ok = {k: sum(bool(x) if k in ("N1", "N4", "N5") else bool(x["equal"] if k == "N2" else x["equal_to_h"])
                 for x in v.values()) for k, v in rec["checks"].items()}
    rec["checks_summary"] = {k: dict(passed=ok[k], of=want[k]) for k in want}
    md = ["# tab:ptq, GSM8K with lm-eval's gsm8k_llama, thinking off (part N)", "",
          "lm-eval 0.4.11 `gsm8k_llama` as shipped: 8-shot CoT (first_n), apply_chat_template + fewshot_as_multiturn, greedy, "
          "max_gen_toks 1024, filters strict_match / flexible_extract; the full test set (1,319). Thinking off: Qwen3.8-27B "
          "`enable_thinking=False` (HFLM chat_template_args), Nemotron-Nano-9B-v2 system instruction `/no_think`. lm-eval's "
          "HFLM on the natively quantized model (flipquant `evaluation.downstream`, native sm120, build_V, `--kernel-set "
          "auto`, n16k64-fast); part H's artifacts and activations. Accuracy ± 2 SE (binomial).", "",
          "| model | method | format | batch | strict-match (%) | flexible-extract (%) | cannot extract (strict / flexible) | "
          "used the 1024 budget | mean / max tokens |", "|---|---|---|---:|---:|---:|---|---:|---|"]
    for key, s in rec["configs"].items():
        model, mf = key.split("/")
        method, fmt = mf.split("_", 1)
        g = s["generated"]
        md.append(f"| {RM.TITLES[model]} | {RM.METHOD_NAMES[method]} | {RM.FMT_NAMES[fmt]} | {s['batch']} | "
                  f"{100 * s['strict_match']['accuracy']:.2f} ± {100 * s['strict_match']['two_se']:.2f} | "
                  f"{100 * s['flexible_extract']['accuracy']:.2f} ± {100 * s['flexible_extract']['two_se']:.2f} | "
                  f"{s['strict_match']['cannot_extract']} / {s['flexible_extract']['cannot_extract']} | {g['used_budget']} | "
                  f"{g['mean_tokens']:.0f} / {g['max_tokens']} |")
    md += ["", "Paired per problem (A − B, percentage points ± 2 SE; A-only / B-only correct; McNemar exact p; * beyond 2 SE):",
           "", "| comparison | filter | Δ (pp) | A only | B only | McNemar p |", "|---|---|---:|---:|---:|---:|"]
    for k, d in rec["paired"].items():
        for f, p in d.items():
            md.append(f"| {k} | {f} | {100 * p['delta']:+.2f} ± {100 * p['two_se']:.2f}{' *' if p['significant'] else ''} | "
                      f"{p['a_only']} | {p['b_only']} | {p['mcnemar_p']:.3g} |")
    md += ["", "Checks: N1 no think content {N1}; N2 GPTQ codes_sha256 equal to part H's {N2}; N3 FlipQuant maps equal to part "
           "H's {N3}; N4 1,319 documents with both filters {N4}; N5 native coverage {N5}.".format(
               **{k: f"{ok[k]} of {want[k]}" for k in want})]
    # every output with a think marker, in full context (the N1 exceptions)
    exc = []
    for (model, method, fmt), ex in ex_all.items():
        for d, e in ex.items():
            raw = e.get("raw") or ""
            if any(m in raw for m in THINK):
                exc.append(dict(config=f"{model}/{method}_{fmt}", doc=d, generated=e.get("generated"),
                                strict=e["strict_match"], flexible=e["flexible_extract"], raw=raw))
    for x in exc:
        x["acceptance"] = ACCEPTED.get((x["config"], x["doc"]))
    rec["n1_exceptions"] = exc
    if exc and all(x["acceptance"] for x in exc):
        md += [f"N1's exception ({len(exc)} output): {exc[0]['acceptance'] if len(exc) == 1 else 'see below'}."]
    md.append("")
    if exc:
        md += ["## Check N1 exceptions (outputs containing a think marker)", ""]
        for x in exc:
            md += [f"- `{x['config']}`, doc {x['doc']} ({x['generated']['tokens']} tokens, EOS {x['generated']['eos']}): "
                   f"strict {x['strict']['response']!r} ({'correct' if x['strict']['exact_match'] else 'wrong'}), "
                   f"flexible {x['flexible']['response']!r} ({'correct' if x['flexible']['exact_match'] else 'wrong'})."
                   + (f" **{x['acceptance'][0].upper() + x['acceptance'][1:]}.**" if x["acceptance"] else "") + " Output:",
                   "", "```", x["raw"], "```", ""]
    (OUT / "gsm8k_lmeval.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "GSM8K_LMEVAL.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def context_of(sample):
    """The context string of a logged generate_until sample (lm-eval's 'arguments', in either layout)."""
    a = sample.get("arguments")
    if isinstance(a, dict):
        first = next(iter(a.values()))
        return first.get("arg_0") if isinstance(first, dict) else first[0]
    return a[0][0] if isinstance(a[0], (list, tuple)) else a[0]


SWITCH = {"qwen3.8-27b": lambda pr: pr.endswith("<think>\n\n</think>\n\n") and "Reasoning effort" not in pr,
          "nemotron-nano-9b-v2": lambda pr: pr.endswith("<think></think>") and "/no_think" not in pr}


def pilot():
    """N0: the smoke (the rendered prompt of one request per model, the off switch, no think content) and the pilot
    (memory and time per batch at the pilot's batch size, the ETA of the 18 configurations)."""
    d = OUT / "pilot"
    d.mkdir(parents=True, exist_ok=True)
    rec, md = {}, ["# Part N0: lm-eval gsm8k_llama smoke and pilot (not a result)", ""]
    total_h = 0.0
    for model in M.MODELS:
        p = NL.ROOT / "pilot" / model
        (d / model).mkdir(exist_ok=True)
        r = dict(runs={})
        for f in ("doc_ids.json", "doc_ids_record.json"):
            if (p / f).exists():
                shutil.copyfile(p / f, d / model / f)
        for name in sorted(p.glob("*_rtn_fo6*.json")):
            rep = load(name)
            if rep is None:
                continue
            s = summary(rep)
            s.update(chat_options=rep.get("chat_options"), chat_template_args=rep.get("chat_template_args"),
                     hflm_max_length=rep.get("hflm_max_length"), load_seconds=rep.get("load_seconds"))
            r["runs"][name.stem] = s
            samples = name.with_name(name.stem + f".{TASK}.jsonl.gz")
            if samples.exists():
                shutil.copyfile(samples, d / model / samples.name)
                if name.stem.startswith("smoke"):
                    first = json.loads(gzip.open(samples, "rt").readline())
                    prompt = context_of(first)
                    (d / model / "rendered_prompt.txt").write_text(prompt)
                    r["rendered_prompt"] = dict(doc_id=first["doc_id"], chars=len(prompt), off_switch=SWITCH[model](prompt),
                                                tail=prompt[-80:])
        rec[model] = r
        md += [f"## {RM.TITLES[model]}", ""]
        if "rendered_prompt" in r:
            rp = r["rendered_prompt"]
            md += [f"Rendered prompt of doc {rp['doc_id']} ({rp['chars']} characters; `rendered_prompt.txt`), off switch "
                   f"present: {rp['off_switch']}; it ends with `{rp['tail'][-40:]!r}`.", ""]
        md += ["| run | batch | problems | strict / flexible (%) | think markers | used budget | mean / max tokens | "
               "seconds (generate) | peak GiB |", "|---|---:|---:|---|---:|---:|---|---|---:|"]
        for k, s in r["runs"].items():
            g = s["generated"]
            md.append(f"| {k} | {s['batch']} | {s['n']} | {100 * s['strict_match']['accuracy']:.1f} / "
                      f"{100 * s['flexible_extract']['accuracy']:.1f} | {s['think_markers']} | {g['used_budget']} | "
                      f"{g['mean_tokens']:.0f} / {g['max_tokens']} | {s['seconds']:.0f} ({s['generate_seconds']:.0f}) | "
                      f"{s['peak_gpu_allocated_gib']:.1f} |")
        pil = [s for k, s in r["runs"].items() if k.startswith("pilot")]
        if pil:
            s = pil[-1]
            batches = math.ceil(s["n"] / s["batch"])
            per_batch = s["generate_seconds"] / batches
            one = (r["runs"][next(k for k in r["runs"] if k.startswith("pilot"))].get("load_seconds") or 60) + \
                math.ceil(N_DOCS / s["batch"]) * per_batch
            nine = 9 * one + 3 * 120 + 3 * 0.2 * (one - 60)       # GPTQ loads +2 min, Hadamard +20 % (assumed)
            r["eta"] = dict(seconds_per_batch=per_batch, hours_per_config=one / 3600, hours_9_configs=nine / 3600)
            total_h += nine / 3600
            md += ["", f"ETA: {per_batch:.0f} s per batch of {s['batch']}, {one / 3600:.2f} h per configuration, "
                       f"{nine / 3600:.1f} h for the 9 (GPTQ loads +2 min, Hadamard +20 % assumed)."]
        md.append("")
    rec["eta_total_hours"] = total_h
    md += [f"Total ETA for the 18 configurations: {total_h:.1f} h.", ""]
    (d / "pilot.json").write_text(json.dumps(rec, indent=1) + "\n")
    (d / "PILOT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    {"pilot": pilot, "report": report}[sys.argv[1]]()

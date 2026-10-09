"""Part M reports (PROTOCOL.md part M): tab:ptq's GSM8K column.

    python report_gsm8k.py pilot    # M0: results/paper_eval/ptq_gsm8k/pilot/{PILOT.md, pilot.json} (lengths, timing, batch identity)
    python report_gsm8k.py report   # M1: results/paper_eval/ptq_gsm8k/{GSM8K.md, gsm8k.json, records/}, checks M1-M4

Accuracy ± 2 SE is binomial (sqrt(p(1-p)/n)). Paired comparisons are per problem: the accuracy difference A - B ± 2 SE of
the per-problem difference, the discordant counts (A right / B wrong, A wrong / B right) and McNemar's exact two-sided p.
"""
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gsm8k as M  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval" / "ptq_gsm8k"
TITLES = {"nemotron-nano-9b-v2": "Nemotron-Nano-9B-v2", "qwen3.8-27b": "Qwen3.8-27B"}
FMT_NAMES = {"nvfp4": "NVFP4", "fo6": "FourOverSix", "fq-16x64": "FlipQuant 16x64"}
METHOD_NAMES = {"rtn": "RTN", "gptq": "GPTQ", "hadamard": "Hadamard"}
N_PROBLEMS = 1319
THINK = ("<think>", "</think>")


def records(jsonl):
    """The harness's per-problem records, the last one per (id, sample) (a resumed run appends)."""
    out = {}
    for line in Path(jsonl).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["id"], r["sample"])] = r
    return out


def ordered(jsonl):
    """The records in the order they were written (batch order), first occurrence."""
    seen, out = set(), []
    for line in Path(jsonl).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if (r["id"], r["sample"]) not in seen:
                seen.add((r["id"], r["sample"]))
                out.append(r)
    return out


def timeline(log):
    """(t_cmd, [(t, generated)]) from a time-stamped job log."""
    t0, marks = None, []
    for line in Path(log).read_text().splitlines():
        t, _, msg = line.partition(" ")
        if msg.startswith("CMD ") and t0 is None:
            t0 = float(t)
        elif msg.startswith("gsm8k: ") and "generated" in msg:
            marks.append((float(t), int(msg.split()[1].split("/")[0])))
    return t0, marks


def summary(recs):
    v = list(recs.values())
    n = len(v)
    acc = sum(r["correct"] for r in v) / n
    toks = [r["tokens"] for r in v]
    return dict(n=n, accuracy=acc, two_se=2 * math.sqrt(acc * (1 - acc) / n), correct=sum(r["correct"] for r in v),
                truncated=sum(r["truncated"] for r in v), truncated_fraction=sum(r["truncated"] for r in v) / n,
                unparseable=sum(r["pred"] is None for r in v), mean_tokens=statistics.mean(toks), max_tokens=max(toks),
                median_tokens=statistics.median(toks), think_markers=sum(any(m in r["completion"] for m in THINK) for r in v))


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a, b):
    """A vs B per problem (the problems both have)."""
    ids = sorted(set(a) & set(b))
    d = [int(a[i]["correct"]) - int(b[i]["correct"]) for i in ids]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
    bb = sum(x == 1 for x in d)
    cc = sum(x == -1 for x in d)
    return dict(n=n, delta=m, two_se=2 * se, a_only=bb, b_only=cc, mcnemar_p=mcnemar_p(bb, cc),
                significant=abs(m) > 2 * se)


def eta(lengths, step_s, batch, load_s=45.0, n=N_PROBLEMS, draws=20000, seed=0):
    """Hours for one configuration: load + the expected decode steps of n problems in static batches of ``batch``
    (a batch runs until its longest completion ends), by resampling the pilot's completion lengths."""
    import random
    rng = random.Random(seed)
    nb = math.ceil(n / batch)
    if batch == 1:
        steps = n * statistics.mean(lengths)
    else:
        steps = nb * statistics.mean(max(rng.choice(lengths) for _ in range(batch)) for _ in range(draws // batch + 200))
    return (load_s + steps * step_s) / 3600


# ------------------------------------------------------------------------------------------------------------- pilot
def pilot():
    d = OUT / "pilot"
    d.mkdir(parents=True, exist_ok=True)
    rec, md = {}, ["# Part M0: GSM8K pilot (not a result)", "",
                   "RTN FourOverSix, greedy, thinking off, 2048 new tokens at most; the first 64 problems at batch 16, the "
                   "first 64 (Nemotron-Nano-9B-v2) / 32 (Qwen3.8-27B) at batch 1. Batch identity: each problem's completion "
                   "text at batch 16 against batch 1's.", ""]
    for model in M.MODELS:
        p = M.ROOT / "pilot" / model
        prm = json.loads((p / "prompt.json").read_text())
        runs = {}
        for b in (16, 1):
            js = p / f"rtn_fo6_b{b}.jsonl"
            if not js.exists() or not (p / f"rtn_fo6_b{b}.json").exists():
                continue
            rs, order = records(js), ordered(js)
            t0, marks = timeline(p / f"rtn_fo6_b{b}.log")
            rep = json.loads((p / f"rtn_fo6_b{b}.json").read_text())
            first = marks[0][0] if marks else None
            # per batch: duration (between progress marks; the first from the previous mark is unknown) and its steps
            batches = []
            for k in range(1, len(marks)):
                lo, hi = marks[k - 1][1], marks[k][1]
                chunk = order[lo:hi]
                batches.append(dict(seconds=marks[k][0] - marks[k - 1][0], problems=len(chunk),
                                    steps=max(r["tokens"] for r in chunk), tokens=sum(r["tokens"] for r in chunk)))
            steps = sum(x["steps"] for x in batches)
            secs = sum(x["seconds"] for x in batches)
            runs[b] = dict(summary=summary(rs), seconds_total=rep["resources"]["seconds"],
                           seconds_before_first_batch_end=(first - t0) if first and t0 else None,
                           batches_timed=len(batches), timed_seconds=secs, timed_steps=steps,
                           seconds_per_step=secs / steps if steps else None,
                           seconds_per_problem=secs / sum(x["problems"] for x in batches) if batches else None,
                           gpu_peak_gib=rep["resources"]["gpu_peak_gib"], records=rs)
        same = None
        if 16 in runs and 1 in runs:
            a, b1 = runs[16]["records"], runs[1]["records"]
            ids = sorted(set(a) & set(b1), key=lambda k: int(k[0].split("/")[1]))
            diff = []
            for i in ids:
                x, y = a[i]["completion"], b1[i]["completion"]
                if x != y:
                    j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
                    diff.append(dict(id=i[0], first_diff_char=j, len_b16=len(x), len_b1=len(y),
                                     pred_b16=a[i]["pred"], pred_b1=b1[i]["pred"], correct_b16=a[i]["correct"],
                                     correct_b1=b1[i]["correct"]))
            same = dict(problems=len(ids), identical=len(ids) - len(diff), differ=len(diff),
                        answer_differs=sum(x["pred_b16"] != x["pred_b1"] for x in diff),
                        correctness_differs=sum(x["correct_b16"] != x["correct_b1"] for x in diff), differences=diff)
        lengths = [r["tokens"] for rr in runs.values() for r in rr["records"].values()]
        etas = {}
        if 16 in runs:
            s16 = runs[16]["seconds_per_step"]
            s1 = runs[1]["seconds_per_step"] if 1 in runs else s16
            for bb, st in ((1, s1), (16, s16), (32, s16 * 1.02), (64, s16 * 1.10)):
                one = eta(lengths, st, bb)
                # 9 configurations: 3 GPTQ loads (+2 min each), 3 Hadamard runs with a 20 % slower decode step (estimates)
                total = 9 * one + 3 * 2 / 60 + 3 * 0.2 * (one - 45 / 3600)
                etas[bb] = dict(hours_per_config=one, hours_9_configs=total, step_s=st,
                                piloted=bb in runs)
        rec[model] = dict(prompt_tail_off=prm["prompt_off"][-60:], pad_token=prm["pad_token"],
                          runs={b: {k: v for k, v in r.items() if k != "records"} for b, r in runs.items()},
                          batch_identity=same, lengths_pooled=len(lengths), eta=etas)
        md += [f"## {TITLES[model]}", "",
               f"Thinking-off prompt (end): `{prm['prompt_off'][-48:]!r}`; registry switch {prm['thinking_registry']}.", ""]
        md += ["| batch | problems | accuracy | mean / median / max tokens | truncated | unparseable | think markers | "
               "s per step | s per problem | total s (load included) | GPU peak GiB |", "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for b, r in runs.items():
            s = r["summary"]
            md.append(f"| {b} | {s['n']} | {100 * s['accuracy']:.1f} % | {s['mean_tokens']:.0f} / {s['median_tokens']:.0f} / "
                      f"{s['max_tokens']} | {s['truncated']} | {s['unparseable']} | {s['think_markers']} | "
                      f"{r['seconds_per_step'] or float('nan'):.4f} | {r['seconds_per_problem'] or float('nan'):.2f} | "
                      f"{r['seconds_total']:.0f} | {r['gpu_peak_gib'][0]:.1f} |")
        if same:
            md += ["", f"Batch identity (batch 16 vs batch 1, {same['problems']} problems): {same['identical']} identical, "
                       f"{same['differ']} differ ({same['answer_differs']} with a different extracted answer, "
                       f"{same['correctness_differs']} with a different correctness)."]
            for x in same["differences"][:20]:
                md.append(f"- {x['id']}: first difference at character {x['first_diff_char']} (lengths {x['len_b16']} / "
                          f"{x['len_b1']}); answers {x['pred_b16']} / {x['pred_b1']}; correct {x['correct_b16']} / "
                          f"{x['correct_b1']}")
        pc = d / "padcheck.json"
        if pc.exists():
            q = json.loads(pc.read_text()).get(model)
            if q:
                md += ["", f"Padding (gsm8k_padcheck.py): rows with no left padding in their batch-16 batch: "
                           f"{q['no_pad']['identical']} of {q['no_pad']['n']} identical to batch 1; padded rows: "
                           f"{q['padded']['identical']} of {q['padded']['n']}. The dependence is not only padding."]
        if etas:
            md += ["", "Estimated hours (resampling the pilot's completion lengths; batches 32 / 64 not piloted, their step "
                       "time assumed 2 % / 10 % above batch 16's; GPTQ +2 min per load and Hadamard +20 % per step assumed):",
                   "", "| batch | per configuration | 9 configurations |", "|---:|---:|---:|"]
            md += [f"| {bb} | {e['hours_per_config']:.2f} | {e['hours_9_configs']:.1f} |" for bb, e in etas.items()]
        md.append("")
    for model in M.MODELS:                         # the rendered prompts and the per-problem records, gzipped
        src, dst = M.ROOT / "pilot" / model, d / model
        dst.mkdir(exist_ok=True)
        (dst / "prompt.json").write_text((src / "prompt.json").read_text())
        for js in sorted(src.glob("*.jsonl")):
            with gzip.open(dst / (js.name + ".gz"), "wt") as f:
                f.write(js.read_text())
    (d / "pilot.json").write_text(json.dumps(rec, indent=1) + "\n")
    (d / "PILOT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------------------------------------------------ report
def h_record(model, method, fmt):
    if method == "rtn":
        return json.loads((M.G.RUN / "pplfast" / model / f"{fmt}.json").read_text())
    return json.loads((M.G.RUN / "ptq" / model / f"{method}_{fmt}.json").read_text())


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records").mkdir(exist_ok=True)
    rec = dict(configs={}, paired={}, checks=dict(M1={}, M2={}, M3={}, M4={}))
    recs = {}
    for model in M.MODELS:
        for method in M.METHODS:
            for fmt in M.FMTS:
                base = M.ROOT / model / f"{method}_{fmt}"
                js, rj = base.with_suffix(".jsonl"), base.with_suffix(".json")
                if not js.exists() or not rj.exists() or "resources" not in json.loads(rj.read_text()):
                    continue
                rep = json.loads(rj.read_text())
                rs = {k[0]: v for k, v in records(js).items()}
                recs[(model, method, fmt)] = rs
                s = summary({(k, 0): v for k, v in rs.items()})
                key = f"{model}/{method}_{fmt}"
                rec["configs"][key] = dict(s, batch=int(rep["args"]["batch"]), seconds=rep["resources"]["seconds"],
                                           gpu_peak_gib=rep["resources"]["gpu_peak_gib"], thinking=rep["thinking"],
                                           decoding=rep["decoding"], max_new_tokens=rep["results"]["gsm8k"]["max_new_tokens"])
                rec["checks"]["M1"][key] = s["think_markers"] == 0
                rec["checks"]["M4"][key] = s["n"] == N_PROBLEMS == rep["results"]["gsm8k"]["problems"]
                h = h_record(model, method, fmt)
                if method == "gptq":
                    a, b = (rep.get("ptq") or {}).get("codes_sha256"), (h.get("ptq") or {}).get("codes_sha256")
                    rec["checks"]["M2"][key] = dict(gsm8k=a, ppl=b, equal=a is not None and a == b)
                if fmt == "fq-16x64":
                    mine, theirs = rep.get("map") or {}, h.get("map") or {}
                    ok = all(mine.get(k) == theirs.get(k) and mine.get(k) is not None for k in ("path", "modules", "e0m3_tiles"))
                    rec["checks"]["M3"][key] = dict(path=mine.get("path"), modules=mine.get("modules"),
                                                    e0m3_tiles=mine.get("e0m3_tiles"), equal_to_h=ok)
                with gzip.open(OUT / "records" / f"{model}__{method}_{fmt}.jsonl.gz", "wt") as f:
                    for k in sorted(rs, key=lambda x: int(x.split("/")[1])):
                        f.write(json.dumps(rs[k]) + "\n")
    for model in M.MODELS:
        for method in M.METHODS:
            for fmt in ("nvfp4", "fq-16x64"):
                a, b = recs.get((model, method, fmt)), recs.get((model, method, "fo6"))
                if a and b:
                    rec["paired"][f"{model}/{method}: {fmt} vs fo6"] = paired(a, b)
        for method in ("gptq", "hadamard"):
            a, b = recs.get((model, method, "fq-16x64")), recs.get((model, "rtn", "fq-16x64"))
            if a and b:
                rec["paired"][f"{model}/fq-16x64: {method} vs rtn"] = paired(a, b)
    md = ["# tab:ptq, GSM8K (part M)", "",
          "Greedy, thinking off (Qwen3.8-27B: `enable_thinking=False`; Nemotron-Nano-9B-v2: `/no_think`), EOS or 2048 new "
          "tokens, the full GSM8K test set (1,319 problems), flipquant `evaluation.accuracy` (its prompt and scorer), "
          "native 16x64, n16k64-fast, build_V, `--kernel-set auto`; the artifacts and activations of part H's PPL runs. "
          "Accuracy ± 2 SE (binomial). Truncated: 2048 tokens generated without EOS. Unparseable: no `\\boxed{}` and no "
          "number in the completion.", "",
          "| model | method | format | batch | accuracy (%) | truncated | unparseable | mean / max tokens |",
          "|---|---|---|---:|---:|---:|---:|---|"]
    for key, s in rec["configs"].items():
        model, mf = key.split("/")
        method, fmt = mf.split("_", 1)
        md.append(f"| {TITLES[model]} | {METHOD_NAMES[method]} | {FMT_NAMES[fmt]} | {s['batch']} | "
                  f"{100 * s['accuracy']:.2f} ± {100 * s['two_se']:.2f} | {s['truncated']} ({100 * s['truncated_fraction']:.1f} %) | "
                  f"{s['unparseable']} | {s['mean_tokens']:.0f} / {s['max_tokens']} |")
    md += ["", "Paired per problem (A − B, percentage points ± 2 SE; A-only / B-only correct; McNemar exact p; * beyond 2 SE):",
           "", "| comparison | Δ accuracy (pp) | A only | B only | McNemar p |", "|---|---:|---:|---:|---:|"]
    for k, p in rec["paired"].items():
        md.append(f"| {k} | {100 * p['delta']:+.2f} ± {100 * p['two_se']:.2f}{' *' if p['significant'] else ''} | "
                  f"{p['a_only']} | {p['b_only']} | {p['mcnemar_p']:.3g} |")
    md += ["", "Checks: " + json.dumps({k: (all(v.values()) if k in ("M1", "M4") else
                                           all(x["equal"] if k == "M2" else x["equal_to_h"] for x in v.values()))
                                       for k, v in rec["checks"].items()}), ""]
    (OUT / "gsm8k.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "GSM8K.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    {"pilot": pilot, "report": report}[sys.argv[1]]()

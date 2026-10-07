"""The D pilot report (amendment 4): results/paper_eval/d_pilot/{DPILOT.md, dpilot.json}.

- Lengths: per (model, task) the generated-token distribution of the BF16 and FlipQuant 16x64 samples at the 32768
  cap, and the share of samples that a cap of 4k / 8k / 16k / 32k would truncate (tokens >= cap; at 32k the harness's
  own truncation flag).
- Step time: a batch's duration is the gap between two logged batch ends (the log's wall-clock prefix); divided by the
  batch's longest sample it is the average decode step at that batch size and context. From the length runs (batch 16,
  long contexts) and from the batch-scaling runs (512 tokens, batch 16 / 32 / 64).
- Full-run estimate per (model, policy, task, batch size): batches x E[longest sample in a batch] x step time, with
  E[longest] bootstrapped from the pilot lengths (FlipQuant 16x64's for the other FP4 policies) and the step time of the
  length runs at batch 16, scaled by each policy's batch-scaling ratio step(b) / step(16).

    python report_dpilot.py
"""
import json
import random
import re
import statistics
from pathlib import Path

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval" / "d_pilot"
MODELS = [("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2"), ("qwen3.8-27b", "Qwen3.8-27B")]
POLS = ["bf16", "nvfp4", "fo6", "fq-8x64", "fq-16x64", "fq-256x64"]
TASKS = {"gsm8k": 1319, "math500": 500, "ifeval": 541, "aime24": 240, "aime25": 240}   # full sizes (AIME: 30 x 8)
CAPS = (4096, 8192, 16384, 32768)
GEN = re.compile(r"^(?P<ts>\d+\.\d+) (?P<task>\w+): (?P<n>\d+)/(?P<N>\d+) generated")


def samples(model, pol, tag):
    f = RUN / "d_pilot" / model / pol / f"{tag}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def batch_times(model, pol, tag, recs):
    """[(task, batch size, seconds, longest tokens)] for every batch after a task's first."""
    log = RUN / "logs" / f"dpilot_{model}_{pol}_{tag}.log"
    if not log.exists():
        return []
    ends = [(m["task"], float(m["ts"]), int(m["n"])) for m in (GEN.match(l) for l in log.read_text().splitlines()) if m]
    by_task = {}
    for r in recs:
        by_task.setdefault(r["task"], []).append(r)
    out = []
    for task in dict.fromkeys(t for t, _, _ in ends):
        ev = [(ts, n) for t, ts, n in ends if t == task]
        rows = by_task.get(task, [])
        for (t0, n0), (t1, n1) in zip(ev, ev[1:]):
            chunk = rows[n0:n1]                 # the .jsonl lists samples in generation order
            if chunk:
                out.append((task, n1 - n0, t1 - t0, max(r["tokens"] for r in chunk)))
    return out


def dist(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return dict(n=len(v), mean=statistics.mean(v), p25=q(0.25), median=statistics.median(v), p75=q(0.75), p90=q(0.9),
                max=v[-1], truncated_at={c: sum(x >= c for x in v) / len(v) for c in CAPS})


def expected_max(lengths, b, draws=2000, seed=0):
    rnd = random.Random(seed)
    return statistics.mean(max(rnd.choice(lengths) for _ in range(b)) for _ in range(draws))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(lengths={}, step_ms={}, scaling={}, estimate_h={}, oom={})
    L = ["# D pilot: generation lengths and run-time estimate (amendment 4; not a result)", ""]
    for model, title in MODELS:
        lengths = {}
        for pol in ("bf16", "fq-16x64"):
            for tag in ("main", "aime"):
                for r in samples(model, pol, tag):
                    lengths.setdefault(pol, {}).setdefault(r["task"], []).append(r["tokens"])
        rec["lengths"][model] = {p: {t: dist(v) for t, v in d.items()} for p, d in lengths.items()}
        # step times
        long_step = {}
        for pol in ("bf16", "fq-16x64"):
            bt = [x for tag in ("main", "aime") for x in batch_times(model, pol, tag, samples(model, pol, tag))]
            if bt:
                long_step[pol] = sum(s for _, _, s, _ in bt) / sum(m for _, _, _, m in bt)
        scal = {}
        for pol in POLS:
            for b in (16, 32, 64):
                tag = f"batch{b}"
                bt = batch_times(model, pol, tag, samples(model, pol, tag))
                if bt:
                    scal.setdefault(pol, {})[b] = sum(s for _, _, s, _ in bt) / sum(m for _, _, _, m in bt)
                elif (RUN / "logs" / f"dpilot_{model}_{pol}_{tag}.log").exists():
                    rec["oom"].setdefault(model, []).append(f"{pol} batch {b}")
        rec["step_ms"][model] = dict(long_context_batch16={p: 1e3 * v for p, v in long_step.items()},
                                     short_context={p: {b: 1e3 * v for b, v in d.items()} for p, d in scal.items()})
        # estimates
        est = {}
        for pol in POLS:
            lsrc = lengths.get("bf16" if pol == "bf16" else "fq-16x64", {})
            base = long_step.get("bf16" if pol == "bf16" else "fq-16x64")
            if not lsrc or base is None or pol not in scal or 16 not in scal[pol]:
                continue
            ref16 = scal["bf16" if pol == "bf16" else "fq-16x64"].get(16) if ("bf16" if pol == "bf16" else "fq-16x64") in scal else None
            for b in (16, 32, 64):
                if b not in scal[pol] or not ref16:
                    continue
                step = base * scal[pol][b] / ref16          # the long-context step, scaled to this policy and batch
                tot = {}
                for task, n in TASKS.items():
                    v = lsrc.get(task)
                    if not v:
                        continue
                    batches = -(-n // b)
                    tot[task] = batches * expected_max(v, b) * step / 3600
                est.setdefault(pol, {})[b] = dict(tasks=tot, total=sum(tot.values()))
        rec["estimate_h"][model] = est
        # the markdown
        L += [f"## {title}", "", "Generated tokens per sample at the 32768 cap (BF16 / FlipQuant 16x64):", "",
              "| task | samples | mean | median | p90 | max | truncated at 4k | 8k | 16k | 32k |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for task in TASKS:
            cells = []
            for pol in ("bf16", "fq-16x64"):
                d = rec["lengths"][model].get(pol, {}).get(task)
                cells.append(d)
            if not any(cells):
                continue
            def f(key, fmt="{:.0f}"):
                return " / ".join(fmt.format(d[key]) if d else "—" for d in cells)
            def tr(c):
                return " / ".join(f"{100 * d['truncated_at'][c]:.0f} %" if d else "—" for d in cells)
            L.append(f"| {task} | {f('n')} | {f('mean')} | {f('median')} | {f('p90')} | {f('max')} | {tr(4096)} | {tr(8192)} | "
                     f"{tr(16384)} | {tr(32768)} |")
        L += ["", "Decode step (ms): long context at batch 16 (the length runs) and 512-token batches at 16 / 32 / 64:", "",
              "| policy | long ctx b16 | b16 | b32 | b64 |", "|---|---:|---:|---:|---:|"]
        for pol in POLS:
            lc = rec["step_ms"][model]["long_context_batch16"].get(pol)
            sc = rec["step_ms"][model]["short_context"].get(pol, {})
            L.append(f"| {pol} | {lc:.1f} |" if lc else f"| {pol} | — |")
            L[-1] += " " + " | ".join(f"{sc[b]:.1f}" if b in sc else "—" for b in (16, 32, 64)) + " |"
        L += ["", "Estimated full-run hours per policy (GSM8K 1319, MATH-500 500, IFEval 541, AIME 24+25 60 × 8 at the "
              "32768 cap):", "", "| policy | batch 16 | batch 32 | batch 64 |", "|---|---:|---:|---:|"]
        for pol in POLS:
            e = est.get(pol, {})
            L.append(f"| {pol} | " + " | ".join(f"{e[b]['total']:.1f} h" if b in e else "—" for b in (16, 32, 64)) + " |")
        L.append("")
        if rec["oom"].get(model):
            L += [f"Failed (OOM or other) batch-scaling runs: {', '.join(rec['oom'][model])}.", ""]
    (OUT / "dpilot.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "DPILOT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

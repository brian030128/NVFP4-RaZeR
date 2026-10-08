"""The D pilot report (amendments 4, 7 and 8): results/paper_eval/d_pilot/{DPILOT.md, dpilot.json}. Not a result: it
sizes the full D run (generation lengths, decode-step times, memory) so that the budgets and batch sizes can be chosen.

- Lengths: per (model, task) the generated-token distribution of the BF16 and FlipQuant 16x64 samples at the 32768
  cap, and the share of samples that a cap of 4k / 8k / 16k / 32k would truncate (tokens >= cap).
- Timed batches: a batch's duration is the gap between two consecutive "generated" lines of one process (the log's
  wall-clock prefix); the first batch of every process has no logged start (it follows the model load) and is not timed.
- Decode-step model: HF generate runs a batch until its longest sample ends, so a batch whose longest sample has M
  tokens takes T = a M + c M^2 / 2 (a step costs a + c x context; c is the attention layers' KV reads, which HF's SDPA
  path inflates with repeat_kv copies under a padding mask). Per model, at batch 16: a per policy from its own
  512-token batches; c per weight class (BF16; FP4) from that class's long batches (BF16's; FlipQuant 16x64's), least
  squares in seconds, so the longest batches, which dominate the cost, set it (the per-step cost grows faster than
  linearly with the context, so mid-length batches are over-predicted rather than the longest under-predicted). At
  batch b in {32, 64}: c_b = c b / 16 (the KV reads scale with the batch) and a_b from that policy's batch-b runs; at
  batch 8 (not measured): a_8 = a and c_8 = c / 2.
- Full-run estimate per (model, policy, task, cap, batch, AIME samples per problem): batches x E[a M + c M^2 / 2],
  M = a batch's longest sample at that cap, bootstrapped from the pilot lengths (FlipQuant 16x64's stand in for the
  other FP4 policies, and for Qwen3.8-27B BF16 AIME, whose length job was cancelled: amendment 8); for AIME whole
  problems are drawn (the harness batches a problem's samples together; with 4 samples per problem, 4 of its 8 pilot
  samples), elsewhere single samples.
- Memory at long contexts (HF generate, SDPA): weights + b x context x (KV cache + SDPA's repeat_kv copies, per token)
  + 2.5 GiB, which matches the Qwen3.8-27B BF16 OOM (79.7 GiB allocated at batch 16 and ~22.7k context, with 11.6 GiB
  more reserved but unallocated). A (cap, batch) fits when the prediction at cap + 512 prompt tokens is <= 82 GiB with
  the default allocator, or <= 92 GiB with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; the estimates run every
  task at the largest batch that fits (expandable segments) when the requested one does not.

    python report_dpilot.py
"""
import json
import re
import statistics
from pathlib import Path

import numpy as np

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval" / "d_pilot"
MODELS = [("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2"), ("qwen3.8-27b", "Qwen3.8-27B")]
POLS = ["bf16", "nvfp4", "fo6", "fq-8x64", "fq-16x64", "fq-256x64"]
FP4 = POLS[1:]
LONG = ("bf16", "fq-16x64")                   # the policies with length runs
TASKS = {"gsm8k": 1319, "math500": 500, "ifeval": 541, "aime24": 30, "aime25": 30}   # problems
AIME = ("aime24", "aime25")
CAPS = (4096, 8192, 16384, 32768)
BATCHES = (8, 16, 32, 64)
GEN = re.compile(r"^(?P<ts>\d+\.\d+) (?P<task>\w+): (?P<n>\d+)/(?P<N>\d+) generated")
CMD_BATCH = re.compile(r"--batch (\d+)")
DRAWS = 4000

# Per token and sequence (KiB), from the configs: Qwen3.8-27B 16 full-attention layers x 4 KV heads x 256 (24 heads
# after repeat_kv), Nemotron-H 4 attention layers x 8 KV heads x 128 (40 heads); weights (GiB) from the smoke records.
MEM = {"nemotron-nano-9b-v2": dict(kv_kib=4 * 8 * 128 * 2 * 2 / 1024, rep_kib=40 * 128 * 2 * 2 / 1024,
                                   weights={"bf16": 16.56, "fp4": 6.25}),
       "qwen3.8-27b": dict(kv_kib=16 * 4 * 256 * 2 * 2 / 1024, rep_kib=24 * 256 * 2 * 2 / 1024,
                           weights={"bf16": 50.96, "fp4": 18.39})}
OTHER_GIB = 2.5
PROMPT = 512
LIMIT = {"default": 82.0, "expandable": 92.0}


def cls(pol):
    return "bf16" if pol == "bf16" else "fp4"


def samples(model, pol, tag):
    f = RUN / "d_pilot" / model / pol / f"{tag}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def logs(model, pol, tag):
    d, name = RUN / "logs", f"dpilot_{model}_{pol}_{tag}"
    passes = sorted(d.glob(f"{name}.pass*.log"), key=lambda p: int(p.name[len(name) + 5:-4]))
    return passes + ([d / f"{name}.log"] if (d / f"{name}.log").exists() else [])


def timed(model, pol, tag):
    """Every timed batch: dict(task, b = the process's --batch, n, seconds, M = its longest sample's tokens)."""
    by_task = {}
    for r in samples(model, pol, tag):
        by_task.setdefault(r["task"], []).append(r)          # the .jsonl lists samples in generation order
    out, prev = [], {}
    for log in logs(model, pol, tag):                        # passes in order; a resumed process skips done samples
        lines = log.read_text(errors="replace").splitlines()
        m = CMD_BATCH.search(lines[0]) if lines else None
        b = int(m.group(1)) if m else None
        ends = [(g["task"], float(g["ts"]), int(g["n"])) for g in (GEN.match(l) for l in lines) if g]
        for i, (task, ts, n) in enumerate(ends):
            n0 = prev.get(task, 0)
            chunk = by_task.get(task, [])[n0:n]
            if i > 0 and chunk:
                out.append(dict(task=task, b=b, n=n - n0, seconds=ts - ends[i - 1][1],
                                M=max(r["tokens"] for r in chunk), log=log.name))
            prev[task] = n
    return out


def dist(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return dict(n=len(v), mean=statistics.mean(v), p25=q(0.25), median=statistics.median(v), p75=q(0.75), p90=q(0.9),
                max=v[-1], truncated_at={c: sum(x >= c for x in v) / len(v) for c in CAPS})


def fit16(model):
    """The batch-16 step model: a per policy from its own 512-token batches; c per weight class (BF16, FP4) from that
    class's long batches (BF16's, FlipQuant 16x64's), least squares in seconds so that the longest batches, which
    dominate the cost, set it (mid-length batches are over-predicted rather than the longest under-predicted)."""
    short = {pol: [(x["M"], x["seconds"]) for x in timed(model, pol, "batch16") if x["b"] == 16 and x["n"] == 16]
             for pol in POLS}
    short = {p: v for p, v in short.items() if v}
    long = {src: [(x["M"], x["seconds"], tag, x["task"]) for tag in ("main", "aime") for x in timed(model, src, tag)
                  if x["b"] == 16 and x["n"] == 16 and x["M"] > 1024] for src in LONG}
    if not short or not all(long.values()):
        return {}, {}, []
    c = {"bf16": 0.0, "fp4": 0.0}
    for _ in range(20):
        a = {p: statistics.mean(T / M - c[cls(p)] * M / 2 for M, T in v) for p, v in short.items()}
        for k, src in (("bf16", "bf16"), ("fp4", "fq-16x64")):
            if src in a:
                pts = long[src]
                c[k] = sum((T - a[src] * M) * M * M / 2 for M, T, *_ in pts) / sum((M * M / 2) ** 2 for M, *_ in pts)
    resid = [dict(pol=p, tag="batch16", task="gsm8k", M=M, seconds=round(T, 2),
                  predicted=round(a[p] * M + c[cls(p)] * M * M / 2, 2)) for p, v in short.items() for M, T in v]
    resid += [dict(pol=src, tag=tag, task=task, M=M, seconds=round(T, 2),
                   predicted=round(a[src] * M + c[cls(src)] * M * M / 2, 2))
              for src, pts in long.items() if src in a for M, T, tag, task in pts]
    return a, c, resid


def step_params(model, a16, c16):
    """{pol: {b: (a_b, c_b)}}; c16 per weight class."""
    out = {}
    for pol, a in a16.items():
        c = c16[cls(pol)]
        out[pol] = {16: (a, c), 8: (a, c / 2)}
        for b in (32, 64):
            cb = c * b / 16
            xs = [x for x in timed(model, pol, f"batch{b}") if x["b"] == b and x["n"] == b]
            if xs:
                out[pol][b] = (statistics.mean((x["seconds"] - cb * x["M"] ** 2 / 2) / x["M"] for x in xs), cb)
    return out


def lengths(model, pol):
    """{task: [tokens]} and {task: {problem: [tokens]}} of a length-run policy."""
    flat, by_problem = {}, {}
    for tag in ("main", "aime"):
        for r in samples(model, pol, tag):
            flat.setdefault(r["task"], []).append(r["tokens"])
            by_problem.setdefault(r["task"], {}).setdefault(r["id"], []).append(r["tokens"])
    return flat, by_problem


def peak_gib(model, kind, b, L):
    m = MEM[model]
    return m["weights"][kind] + b * L * (m["kv_kib"] + m["rep_kib"]) / 2 ** 20 + OTHER_GIB


def max_batch(model, kind, cap, alloc="expandable"):
    ok = [b for b in BATCHES if peak_gib(model, kind, b, cap + PROMPT) <= LIMIT[alloc]]
    return max(ok) if ok else None


class Estimator:
    """Hours for (policy, task, cap, batch, AIME samples per problem) of one model's full run."""

    def __init__(self, model, L, par, rng):
        self.model, self.L, self.par, self.rng, self.cache = model, L, par, rng, {}

    def source(self, pol, task):
        """The length-run policy whose lengths stand in for `pol` on `task` (None if no lengths)."""
        for s in (("bf16", "fq-16x64") if pol == "bf16" else ("fq-16x64",)):
            if task in self.L[s][0]:
                return s
        return None

    def draws(self, src, task, k, s):
        key = (src, task, k, s)
        if key not in self.cache:
            flat, byp = self.L[src]
            if task in AIME:                       # the pilot's AIME problems of both years, pooled
                probs = np.array([v[:8] for t in AIME for v in byp.get(t, {}).values()])
                per = max(1, k // s)
                idx = self.rng.integers(len(probs), size=(DRAWS, per))
                if s >= probs.shape[1]:
                    out = probs.max(axis=1)[idx].max(axis=1)
                else:                              # s of a problem's 8 pilot samples, without replacement
                    pick = np.argsort(self.rng.random((DRAWS, per, probs.shape[1])), axis=2)[:, :, :s]
                    out = probs[idx[:, :, None], pick].max(axis=(1, 2))
            else:
                out = self.rng.choice(np.array(flat[task]), size=(DRAWS, k)).max(axis=1)
            self.cache[key] = out
        return self.cache[key]

    def hours(self, pol, task, cap, b, s=8):
        src = self.source(pol, task)
        if src is None or pol not in self.par or b not in self.par[pol]:
            return None
        a, c = self.par[pol][b]
        n = TASKS[task] * (s if task in AIME else 1)
        nfull, rem = divmod(n, b)

        def mean_t(d):
            M = np.minimum(d, cap).astype(float)
            return float(np.mean(a * M + c * M * M / 2))
        h = nfull * mean_t(self.draws(src, task, b, s)) + (mean_t(self.draws(src, task, rem, s)) if rem else 0.0)
        return h / 3600

    def fitted_batch(self, pol, cap, b):
        mb = max_batch(self.model, cls(pol), cap)
        return min(b, mb) if mb else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rec = dict(lengths={}, fit={}, step_ms={}, estimate_h={}, aime_h={}, memory_gib={}, max_batch={}, length_source={},
               aime_problems={}, slot_utilisation_b16={}, options={}, failed=[])
    md = ["# D pilot: generation lengths, decode steps, memory and the full-run estimate", "",
          "Amendments 4, 7 and 8; not a result. flipquant `evaluation.accuracy`, natively in n16k64-fast, "
          "`--recommended-decoding` (thinking on), seed 0, every task at a 32768-token cap. Lengths: BF16 and FlipQuant "
          "16x64 on 16 problems of GSM8K, MATH-500 and IFEval, and 2 problems x 8 samples of AIME 2024 and 2025, batch "
          "16 (the Qwen3.8-27B BF16 AIME job ran out of memory at batch 16 and its batch-8 rerun was cancelled: "
          "amendments 7 and 8). Batch scaling: all six policies on GSM8K at 512 tokens, batch 16 / 32 / 64. The other "
          "FP4 policies' lengths are taken to be FlipQuant 16x64's.", ""]
    est = {}
    for model, title in MODELS:
        L = {pol: lengths(model, pol) for pol in LONG}
        rec["lengths"][model] = {pol: {t: dist(v) for t, v in L[pol][0].items()} for pol in LONG}
        a16, c16, resid = fit16(model)
        rec["fit"][model] = dict(a_ms={p: 1e3 * v for p, v in a16.items()},
                                 c_us_per_token={k: 1e6 * v for k, v in c16.items()}, points=resid)
        par = step_params(model, a16, c16) if c16 else {}
        rec["step_ms"][model] = {p: {b: dict(a=1e3 * ab, c_us=1e6 * cb, at_8k=1e3 * (ab + cb * 8192),
                                             at_32k=1e3 * (ab + cb * 32768)) for b, (ab, cb) in d.items()}
                                 for p, d in par.items()}
        E = est[model] = Estimator(model, L, par, rng)
        rec["estimate_h"][model] = {pol: {task: {cap: {b: E.hours(pol, task, cap, b) for b in BATCHES} for cap in CAPS}
                                          for task in TASKS} for pol in POLS}
        aime_h = {}
        for pol in POLS:
            for s in (8, 4):
                for cap in CAPS:
                    for b in BATCHES:
                        hs = [E.hours(pol, t, cap, b, s) for t in AIME]
                        aime_h.setdefault(pol, {}).setdefault(s, {}).setdefault(cap, {})[b] = \
                            None if None in hs else sum(hs)
        rec["aime_h"][model] = aime_h
        rec["length_source"][model] = {pol: {t: E.source(pol, t) for t in TASKS} for pol in POLS}
        rec["memory_gib"][model] = {k: {b: {cap: round(peak_gib(model, k, b, cap + PROMPT), 1) for cap in CAPS}
                                        for b in BATCHES} for k in ("bf16", "fp4")}
        rec["max_batch"][model] = {k: {alloc: {cap: max_batch(model, k, cap, alloc) for cap in CAPS} for alloc in LIMIT}
                                   for k in ("bf16", "fp4")}
        for pol in POLS:
            for tag in ("main", "aime", "batch16", "batch32", "batch64"):
                if logs(model, pol, tag) and not (RUN / "d_pilot" / model / pol / f"{tag}.json").exists():
                    rec["failed"].append(f"{model} {pol} {tag}")
        # ---- markdown: lengths
        md += [f"## {title}", "", "Generated tokens per sample at the 32768 cap (BF16 / FlipQuant 16x64):", "",
               "| task | samples | mean | median | p90 | max | ≥ 4k | ≥ 8k | ≥ 16k | ≥ 32k |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for task in TASKS:
            cells = [rec["lengths"][model].get(pol, {}).get(task) for pol in LONG]
            if not any(cells):
                continue
            f = lambda k: " / ".join(f"{d[k]:.0f}" if d else "—" for d in cells)
            tr = lambda c: " / ".join(f"{100 * d['truncated_at'][c]:.0f} %" if d else "—" for d in cells)
            md.append(f"| {task} | {f('n')} | {f('mean')} | {f('median')} | {f('p90')} | {f('max')} | {tr(4096)} | "
                      f"{tr(8192)} | {tr(16384)} | {tr(32768)} |")
        aime = {}
        for pol in LONG:
            for r in samples(model, pol, "aime"):
                aime.setdefault(r["id"], {}).setdefault(pol, []).append(r)
        if aime:
            md += ["", "AIME per problem (8 samples each; BF16 / FlipQuant 16x64): generated tokens min–max, samples "
                   "truncated at 32k, samples correct:", "", "| problem | tokens | truncated | correct |",
                   "|---|---:|---:|---:|"]
            for pid in sorted(aime):
                d = aime[pid]
                cell = lambda fn: " / ".join(fn(d[p]) if p in d else "—" for p in LONG)
                md.append(f"| {pid} | " + cell(lambda v: f"{min(x['tokens'] for x in v)}–{max(x['tokens'] for x in v)}") +
                          " | " + cell(lambda v: f"{sum(x['truncated'] for x in v)}/{len(v)}") + " | " +
                          cell(lambda v: f"{sum(bool(x.get('correct')) for x in v)}/{len(v)}") + " |")
            rec["aime_problems"][model] = {pid: {p: [dict(tokens=x["tokens"], truncated=x["truncated"],
                                                          correct=x.get("correct")) for x in v] for p, v in d.items()}
                                           for pid, d in aime.items()}
        util = {}
        for pol in LONG:
            for task in TASKS:
                if E.source(pol, task) == pol:
                    util.setdefault(pol, {})[task] = float(np.mean(L[pol][0][task]) /
                                                           np.mean(E.draws(pol, task, 16, 8)))
        rec["slot_utilisation_b16"][model] = util
        if util:
            md += ["", "Static batches: a batch runs until its longest sample ends, so at batch 16 the share of decode "
                   "slots that produce tokens (mean length / expected longest in the batch) is " + "; ".join(
                       f"{t} " + " / ".join(f"{100 * util[p][t]:.0f} %" if t in util.get(p, {}) else "—" for p in LONG)
                       for t in TASKS) + " (BF16 / FlipQuant 16x64)."]
        # ---- step model
        if c16:
            md += ["", f"Decode step (ms) = a + c x context; c = {1e6 * c16['bf16']:.2f} (BF16) / {1e6 * c16['fp4']:.2f} (FP4) "
                   "µs per context token at batch 16 (scaled with the batch). a per policy and batch, and the step at 8k / "
                   "32k context:", "",
                   "| policy | a, b16 | b32 | b64 | step at 8k, b16 | 32k, b16 | 32k, b64 |",
                   "|---|---:|---:|---:|---:|---:|---:|"]
            for pol in POLS:
                s = rec["step_ms"][model].get(pol)
                if not s:
                    continue
                g = lambda b, k: f"{s[b][k]:.1f}" if b in s else "—"
                md.append(f"| {pol} | {g(16, 'a')} | {g(32, 'a')} | {g(64, 'a')} | {g(16, 'at_8k')} | {g(16, 'at_32k')} | "
                          f"{g(64, 'at_32k')} |")
            md += ["", "Fit (predicted / measured seconds) of the long batches: " + ", ".join(
                f"{r['pol']} {r['task']} M={r['M']}: {r['predicted']:.0f} / {r['seconds']:.0f}" for r in resid
                if r["M"] > 1024) + "; the 512-token batches within " + f"{100 * max(abs(r['predicted'] / r['seconds'] - 1) for r in resid if r['M'] <= 1024):.0f} %."]
        # ---- per task
        md += ["", "Estimated hours per policy and task for the full run (BF16 / mean of the five FP4 policies; AIME per "
               "year, 30 problems x 8 samples); ✗ = does not fit in memory at that cap:", "",
               "| task | cap | batch 8 | batch 16 | batch 32 | batch 64 |", "|---|---:|---:|---:|---:|---:|"]
        for task in TASKS:
            for cap in CAPS:
                row = []
                for b in BATCHES:
                    cell = []
                    for kind, pols in (("bf16", ["bf16"]), ("fp4", FP4)):
                        hs = [h for h in (rec["estimate_h"][model][p][task][cap][b] for p in pols) if h is not None]
                        mb = rec["max_batch"][model][kind]["expandable"][cap]
                        cell.append("✗" if mb is None or b > mb else (f"{statistics.mean(hs):.1f}" if hs else "—"))
                    row.append(" / ".join(cell))
                md.append(f"| {task} | {cap // 1024}k | " + " | ".join(row) + " |")
        # ---- AIME per policy
        md += ["", "Full AIME (2024 + 2025, 60 problems x 8 samples) per policy, hours, at the 16k and 32k caps; ✗ = does "
               "not fit in memory (expandable segments):", "",
               "| policy | 16k, b8 | b16 | b32 | b64 | 32k, b8 | b16 | b32 | b64 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for pol in POLS:
            cells = []
            for cap in (16384, 32768):
                for b in BATCHES:
                    h = aime_h[pol][8][cap][b]
                    mb = rec["max_batch"][model][cls(pol)]["expandable"][cap]
                    cells.append("✗" if mb is None or b > mb else ("—" if h is None else f"{h:.1f}"))
            note = " †" if pol == "bf16" and any(rec["length_source"][model][pol][t] != "bf16" for t in AIME) else ""
            md.append(f"| {pol}{note} | " + " | ".join(cells) + " |")
        if any(rec["length_source"][model]["bf16"][t] != "bf16" for t in AIME):
            md += ["", "† Estimate: BF16's AIME lengths are FlipQuant 16x64's (the BF16 length job was cancelled, "
                   "amendment 8), with BF16's step model (batch 8, not measured: a and c / 2)."]
        md += ["", "Predicted peak memory (GiB) at cap + 512 prompt tokens, BF16 / FP4 (fits: ≤ 82 default allocator, "
               "≤ 92 with expandable segments):", "", "| batch | 4k | 8k | 16k | 32k |", "|---:|---:|---:|---:|---:|"]
        for b in BATCHES:
            md.append(f"| {b} | " + " | ".join(
                " / ".join(f"{peak_gib(model, k, b, cap + PROMPT):.0f}" for k in ("bf16", "fp4")) for cap in CAPS) + " |")
        md.append("")
    # ---- options for the full run
    caps_i = {}
    for model, _ in MODELS:
        caps_i[model] = {}
        for t in TASKS:
            mx = max([d[t]["max"] for d in rec["lengths"][model].values() if t in d] or [0])
            caps_i[model][t] = next((c for c in CAPS if c > mx), CAPS[-1])
    three = ["bf16", "fo6", "fq-16x64"]
    opts = {"(i) caps above every pilot sample, AIME avg@8, 6 policies": dict(aime_cap=None, s=8, aime_pols=POLS),
            "(ii) = (i) with AIME avg@4": dict(aime_cap=None, s=4, aime_pols=POLS),
            "(iii) = (i) with AIME at a 16k cap": dict(aime_cap=16384, s=8, aime_pols=POLS),
            "(iv) = (i) with AIME for BF16, FO6, FQ 16x64 only": dict(aime_cap=None, s=8, aime_pols=three),
            "(ii) + (iv)": dict(aime_cap=None, s=4, aime_pols=three),
            "(iii) + (iv)": dict(aime_cap=16384, s=8, aime_pols=three)}
    md += ["## Full D run under the candidate options", "",
           "(i) caps per model and task: the smallest of 4k / 8k / 16k / 32k above every pilot sample (" + "; ".join(
               f"{title}: " + ", ".join(f"{t} {c // 1024}k" for t, c in caps_i[m].items()) for m, title in MODELS) +
           "). With 16 samples per task (32 for AIME) a < 1 % truncation rate cannot be verified; Qwen3.8-27B's AIME "
           "exceeds every measured cap (it truncates at 32k: see above), so (i) keeps 32k there. Every task runs at the "
           "requested batch or, when that does not fit at its cap, at the largest batch that does (marked *). Hours, "
           "all policies (AIME hours in brackets):", "", "| option | batch | Nemotron-Nano-9B-v2 | Qwen3.8-27B | total |",
           "|---|---:|---:|---:|---:|"]
    for name, o in opts.items():
        for b in (16, 32, 64):
            tot, aime_part, star = {}, {}, False
            for model, _ in MODELS:
                E, h, ha = est[model], 0.0, 0.0
                for pol in POLS:
                    for task in TASKS:
                        is_aime = task in AIME
                        if is_aime and pol not in o["aime_pols"]:
                            continue
                        cap = (o["aime_cap"] or caps_i[model][task]) if is_aime else caps_i[model][task]
                        bb = E.fitted_batch(pol, cap, b)
                        v = E.hours(pol, task, cap, bb, o["s"] if is_aime else 8) if bb else None
                        if v is None:
                            h = None
                            break
                        star |= bb != b
                        h += v
                        ha += v if is_aime else 0.0
                    if h is None:
                        break
                tot[model], aime_part[model] = h, (ha if h is not None else None)
            total = None if None in tot.values() else sum(tot.values())
            rec["options"].setdefault(name, {})[b] = dict(hours=tot, aime_hours=aime_part, total=total, fallback=star)
            g = lambda m: "—" if tot[m] is None else f"{tot[m]:.0f} ({aime_part[m]:.0f})"
            md.append(f"| {name} | {b}{'*' if star else ''} | {g('nemotron-nano-9b-v2')} | {g('qwen3.8-27b')} | "
                      f"{'—' if total is None else f'{total:.0f}'} |")
    rec["caps_i"] = caps_i
    md += ["", "Limits of the estimate: 16 problems per task (2 per AIME year) set the length distributions, and "
           "sampling makes a rerun's lengths differ; the step model is fitted on 512-token batches and a few long "
           "ones; the FP4 policies other than FlipQuant 16x64 are assumed to generate FlipQuant 16x64's lengths; "
           "batch 8 is not measured.", ""]
    if rec["failed"]:
        md += ["Pilot jobs without a summary: " + ", ".join(rec["failed"]) + ".", ""]
    (OUT / "dpilot.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "DPILOT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Part AC (PROTOCOL.md): the reports of the paper's 1,024-window reruns, results/paper_eval/paper1k/<item>/.

    python report_paper1k.py item2       # LATENCY.md, MEMORY.md, item2.json (tab:latency SM120 rows, the memory paragraph)

The reports hold no local paths or host names, so that the same files can be copied to flipquant paper-1k-reruns.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_paper1k as P  # noqa: E402

G = P.G
RUN = G.RUN / "paper1k"
OUT = G.RZ / "results" / "paper_eval" / "paper1k"
GIB = 2 ** 30
MODELS = [("llama3.2-1b-instruct", "Llama-3.2-1B-Instruct"), ("llama3.1-8b-instruct", "Llama-3.1-8B-Instruct"),
          ("mistral-7b", "Mistral-7B-Instruct-v0.3"), ("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2"),
          ("phi4-14b", "Phi-4"), ("qwen3.8-27b", "Qwen3.8-27B")]
TITLE = dict(MODELS)
POLS = ["bf16", "nvfp4", "fo6", "fo6-wB", "fq-8x64", "fq-16x64", "fq-256x64"]
FQS = ["fq-8x64", "fq-16x64", "fq-256x64"]
NAMES = {"bf16": "BF16", "nvfp4": "NVFP4", "fo6": "FourOverSix", "fo6-wB": "FourOverSix (wB)", "fq-8x64": "FlipQuant 8x64",
         "fq-16x64": "FlipQuant 16x64", "fq-256x64": "FlipQuant 256x64"}
COMPARE = [("fq-16x64", "fo6"), ("fq-8x64", "fo6"), ("fq-256x64", "fo6"), ("fq-8x64", "fo6-wB"), ("nvfp4", "fo6"),
           ("fo6-wB", "fo6"), ("bf16", "fo6")]
SHAPES = G.SHAPES.split(",")
SHAPE = "1x2048"
TABLE_ROWS = ["bf16", "nvfp4", "fo6", "fo6-wB", "fq-16x64", "fq-8x64", "fq-256x64"]
#: the published 256-window rows: part A (58fe644) for four models, part W (277f59d) for the two Llamas
W_COMMIT = "277f59d"
PUBLISHED = {"llama3.2-1b-instruct": "W", "llama3.1-8b-instruct": "W", "mistral-7b": "A", "nemotron-nano-9b-v2": "A",
             "phi4-14b": "A", "qwen3.8-27b": "A"}


def git_json(commit, path):
    return json.loads(subprocess.run(["git", "-C", str(G.RZ), "show", f"{commit}:{path}"], capture_output=True, text=True,
                                     check=True).stdout)


def pct(a, b):
    return 100 * (a / b - 1)


# ---------------------------------------------------------------------------------------------- item 2: latency
def load_latency(root, models):
    data = {}
    for m in models:
        for p in POLS:
            d = RUN / root / m / p
            for f in sorted(d.glob("round*.json")) if d.exists() else []:
                r = json.loads(f.read_text())
                if r.get("status") == "complete":
                    data.setdefault(m, {}).setdefault(p, {})[int(f.stem[5:])] = r
    return data


def latency_tables(data):
    """A's numbers (report_latency.main): per (model, policy, shape) the median over rounds of each process's graph
    median, the eager one, and the paired ratios (median [min, max] over rounds)."""
    rec = dict(ms={}, eager_ms={}, ratios={}, rounds={}, capture={}, weights_gib={})
    for m, rr_all in data.items():
        for p, rr in rr_all.items():
            mode = "graph" if all("ms" in r["graph"].get(s, {}) for r in rr.values() for s in SHAPES) else "eager"
            rec["capture"].setdefault(m, {})[p] = mode
            rec["ms"].setdefault(m, {})[p] = {s: statistics.median(r[mode][s]["ms"] for r in rr.values()) for s in SHAPES}
            rec["eager_ms"].setdefault(m, {})[p] = {s: statistics.median(r["eager"][s]["ms"] for r in rr.values())
                                                    for s in SHAPES}
            rec["rounds"].setdefault(m, {})[p] = sorted(rr)
            rec["weights_gib"].setdefault(m, {})[p] = statistics.median(r["weights_gib"] for r in rr.values())
        for a, b in COMPARE:
            if a in rr_all and b in rr_all:
                common = sorted(set(rr_all[a]) & set(rr_all[b]))
                mode = "graph" if rec["capture"][m][a] == rec["capture"][m][b] == "graph" else "eager"
                rows = {}
                for s in SHAPES:
                    v = [pct(rr_all[a][r][mode][s]["ms"], rr_all[b][r][mode][s]["ms"]) for r in common]
                    rows[s] = dict(median=statistics.median(v), min=min(v), max=max(v), rounds=len(v))
                rec["ratios"].setdefault(m, {})[f"{a} vs {b}"] = dict(
                    mode=mode, shapes=rows, median_over_shapes=statistics.median(x["median"] for x in rows.values()))
    return rec


def record_checks(data, libs):
    """A's registered checks, re-applied to every record (the driver stops on the first failure)."""
    out = dict(records=0, failures=[])
    for m, rr_all in data.items():
        for p, rr in rr_all.items():
            for r in rr:
                out["records"] += 1
                bad = G.check_fq_record(RUN / "latency" / m / p / f"round{r}.json", p, libs)
                if bad:
                    out["failures"].append(dict(model=m, policy=p, round=r, bad=bad))
    return out


def gpu_summary():
    rows = [json.loads(x) for x in (RUN / "gpu_state.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["job"].startswith("p1k_latency") or r["job"].startswith("p1k_memory")]
    num = lambda k: [float(r[k]) for r in rows if r.get(k) not in (None, "", "[N/A]")]  # noqa: E731
    return dict(jobs=len(rows), names=sorted({r["name"] for r in rows}), drivers=sorted({r["driver_version"] for r in rows}),
                power_limits_w=sorted({r["power.limit"] for r in rows}), max_sm_mhz=sorted({r["clocks.max.sm"] for r in rows}),
                boots=len({r["boot_id"] for r in rows}), first_utc=rows[0]["utc"], last_utc=rows[-1]["utc"],
                pre_job_power_w=[min(num("power.draw")), max(num("power.draw"))],
                pre_job_temp_c=[min(num("temperature.gpu")), max(num("temperature.gpu"))],
                pre_job_sm_mhz=[min(num("clocks.sm")), max(num("clocks.sm"))],
                pre_job_pstates=sorted({r["pstate"] for r in rows}))


def record_gpu(data):
    """The prefill records' own start / end snapshots (evaluation.latency)."""
    t0, t1, lim, drv = [], [], set(), set()
    for rr_all in data.values():
        for rr in rr_all.values():
            for r in rr.values():
                t0.append(float(r["gpu"]["temperature.gpu"]))
                t1.append(float(r["gpu_end"]["temperature.gpu"]))
                lim.add(r["gpu"]["power.limit"])
                drv.add(r["gpu"]["driver_version"])
    if not t0:
        return dict(start_temp_c=[None, None], end_temp_c=[None, None], power_limits=[], drivers=[])
    return dict(start_temp_c=[min(t0), max(t0)], end_temp_c=[min(t1), max(t1)], power_limits=sorted(lim),
                drivers=sorted(drv))


def temperature_section(data, out):
    """Per model, the prefill records' own start / end GPU temperatures (degC): this session against the published
    session's records (part A's for four models, part W's for the Llamas)."""
    def ranges(rr_all):
        t0 = [float(r["gpu"]["temperature.gpu"]) for rr in rr_all.values() for r in rr.values()]
        t1 = [float(r["gpu_end"]["temperature.gpu"]) for rr in rr_all.values() for r in rr.values()]
        return dict(start=[min(t0), max(t0)], end=[min(t1), max(t1)], start_median=statistics.median(t0),
                    end_median=statistics.median(t1), jobs=len(t0))
    rows, L = {}, ["## GPU temperature per job (the prefill records' start / end snapshots)", "",
                   "| model | jobs | start °C, now (median) | end °C, now (median) | published session | start °C (median) | "
                   "end °C (median) |", "|---|---:|---|---|---|---|---|"]
    for m, t in MODELS:
        if m not in data:
            continue
        root = "latency" if PUBLISHED[m] == "A" else "latency_llama"
        old = {}
        for p in POLS:
            d = G.RUN / root / m / p
            for f in sorted(d.glob("round*.json")) if d.exists() else []:
                r = json.loads(f.read_text())
                if r.get("status") == "complete":
                    old.setdefault(p, {})[f.stem] = r
        now, pub = ranges(data[m]), ranges(old) if old else None
        rows[m] = dict(now=now, published=pub, published_session=PUBLISHED[m])
        L.append(f"| {t} | {now['jobs']} | {now['start'][0]:.0f}–{now['start'][1]:.0f} ({now['start_median']:.0f}) | "
                 f"{now['end'][0]:.0f}–{now['end'][1]:.0f} ({now['end_median']:.0f}) | {PUBLISHED[m]} | "
                 + (f"{pub['start'][0]:.0f}–{pub['start'][1]:.0f} ({pub['start_median']:.0f}) | "
                    f"{pub['end'][0]:.0f}–{pub['end'][1]:.0f} ({pub['end_median']:.0f}) |" if pub else "— | — |"))
    out["temperatures"] = rows
    return L + [""]


def model_section(rec, m):
    L = [f"## {TITLE[m]}", "", "| shape | " + " | ".join(NAMES[p] for p in POLS if p in rec["ms"][m]) + " |",
         "|---" + "|---:" * len([p for p in POLS if p in rec["ms"][m]]) + "|"]
    for s in SHAPES:
        L.append(f"| {s} | " + " | ".join(f"{rec['ms'][m][p][s]:.2f}" for p in POLS if p in rec["ms"][m]) + " |")
    keys = [f"{a} vs {b}" for a, b in COMPARE if f"{a} vs {b}" in rec["ratios"].get(m, {})]
    L += ["", "| shape | " + " | ".join(f"{NAMES[k.split(' vs ')[0]]} vs {NAMES[k.split(' vs ')[1]]}" for k in keys) + " |",
          "|---" + "|---:" * len(keys) + "|"]
    for s in SHAPES:
        L.append(f"| {s} | " + " | ".join(
            f"{rec['ratios'][m][k]['shapes'][s]['median']:+.1f} % [{rec['ratios'][m][k]['shapes'][s]['min']:+.1f}, "
            f"{rec['ratios'][m][k]['shapes'][s]['max']:+.1f}]" for k in keys) + " |")
    modes = [p for p, v in rec["capture"][m].items() if v != "graph"]
    if modes:
        L += ["", f"Eager timings (no CUDA graph) for: {', '.join(modes)}."]
    return L + [""]


def anchor_section(out):
    c = json.loads((RUN / "anchor_check.json").read_text())
    L = ["## Step 0: the session-drift anchor (Mistral-7B-Instruct-v0.3, the release 16x64 map, against part A 58fe644)", "",
         "Median over 5 rounds of each process's graph median (ms), this session against A, before the panel. Registered "
         f"pass: every cell within ±{c['cell_tol_pct']:.0f} % of A's, and the FlipQuant 16x64 / FourOverSix ratio's median "
         f"over shapes within ±{c['ratio_tol_pp']:.1f} pp of A's.", "",
         "| shape | FourOverSix, now | A | Δ | FlipQuant 16x64 (release map), now | A | Δ | FQ16 vs FO6, now |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SHAPES:
        a, b = c["cells"][f"fo6 {s}"], c["cells"][f"fq-16x64 {s}"]
        L.append(f"| {s} | {a['new']:.2f} | {a['a']:.2f} | {a['pct']:+.2f} % | {b['new']:.2f} | {b['a']:.2f} | "
                 f"{b['pct']:+.2f} % | {c['ratio_per_shape'][s]:+.2f} % |")
    allp = [x["pct"] for x in c["cells"].values()]
    L += ["", f"Cells against A: median {statistics.median(allp):+.2f} %, range [{min(allp):+.2f}, {max(allp):+.2f}] % "
          f"(worst |Δ| {c['worst_cell_pct']:.2f} %). FQ16 vs FO6, median over shapes: {c['ratio_median_over_shapes']:+.2f} % "
          f"now, {c['a_ratio_median_over_shapes']:+.2f} % in A. **{'PASS' if c['passed'] else 'FAIL'}.**", ""]
    out["anchor"] = c
    return L


def combined_section(rec, out):
    raw = {m: {p: rec["ms"][m][p][SHAPE] for p in TABLE_ROWS} for m, _ in MODELS}
    out["combined_1x2048"] = dict(models=[m for m, _ in MODELS], ms=raw)
    L = [f"## The combined {SHAPE} table: the six paper models, one session, the 1k maps", "",
         "Raw medians over rounds (graph ms):", "", "| policy | " + " | ".join(t for _, t in MODELS) + " |",
         "|---" + "|---:" * len(MODELS) + "|"]
    for p in TABLE_ROWS:
        L.append(f"| {NAMES[p]} | " + " | ".join(f"{raw[m][p]:.4f}" for m, _ in MODELS) + " |")
    L += ["", "LaTeX, each cell rounded once to one decimal from the raw value above:", "", "```latex"]
    for p in TABLE_ROWS:
        L.append(f"{NAMES[p]} & " + " & ".join(f"{raw[m][p]:.1f}" for m, _ in MODELS) + r" \\")
    L += ["```", ""]
    return L, raw


def summaries_section(rec, raw, out):
    summ, L = {}, ["## Six-model summaries (part W's)", ""]
    for key in ("fq-16x64 vs fo6", "fq-8x64 vs fo6", "fq-256x64 vs fo6", "fq-8x64 vs fo6-wB"):
        per = {m: [rec["ratios"][m][key]["shapes"][s]["median"] for s in SHAPES] for m, _ in MODELS}
        allv = [v for vv in per.values() for v in vv]
        med = {m: rec["ratios"][m][key]["median_over_shapes"] for m, _ in MODELS}
        at = {m: rec["ratios"][m][key]["shapes"][SHAPE]["median"] for m, _ in MODELS}
        a_p, b_p = key.split(" vs ")
        rom = {m: pct(raw[m][a_p], raw[m][b_p]) for m, _ in MODELS}
        summ[key] = dict(range_all=[min(allv), max(allv)], per_model_median=med, at_1x2048_paired=at,
                         at_1x2048_ratio_of_medians=rom)
        L += [f"**{NAMES[a_p]} vs {NAMES[b_p]}** (paired within rounds; the per-shape value is the median over rounds):",
              f"- Over all 8 shapes and six models: [{min(allv):+.2f}, {max(allv):+.2f}] %.",
              "- Per-model median over the 8 shapes: " + "; ".join(f"{t} {med[m]:+.2f} %" for m, t in MODELS) + ".",
              f"- At {SHAPE}, the paired ratio: [{min(at.values()):+.2f}, {max(at.values()):+.2f}] % ("
              + "; ".join(f"{t} {at[m]:+.2f}" for m, t in MODELS) + ").",
              f"- At {SHAPE}, the ratio of the medians: [{min(rom.values()):+.2f}, {max(rom.values()):+.2f}] % ("
              + "; ".join(f"{t} {rom[m]:+.2f}" for m, t in MODELS) + ").", ""]
    sp = {m: raw[m]["bf16"] / raw[m]["fq-16x64"] for m, _ in MODELS}
    summ["bf16_over_fq16_1x2048"] = sp
    L += [f"**BF16 / FlipQuant 16x64 at {SHAPE}** (ratio of the medians): [{min(sp.values()):.2f}, {max(sp.values()):.2f}] x ("
          + "; ".join(f"{t} {sp[m]:.2f}x" for m, t in MODELS) + ").", ""]
    out["six_model"] = summ
    return L


def published_section(rec, out):
    """This session's panel against the published 256-window rows (A, W). The BF16 / NVFP4 / FourOverSix rows do not
    depend on a map, so their offsets are this session's drift on each model; the FlipQuant rows add the map change."""
    pub = dict(A=git_json(P.A_COMMIT, "results/paper_eval/latency/latency.json"),
               W=git_json(W_COMMIT, "results/paper_eval/latency_llama/latency.json"))
    cmp = {}
    L = ["## Against the published 256-window rows", "",
         "Published: part A (58fe644) for Mistral, Nemotron, Phi-4 and Qwen3.8-27B; part W (277f59d) for the two Llamas. "
         "BF16, NVFP4 and FourOverSix do not use a map, so their offsets measure this session against the published one "
         "on each model. The FlipQuant rows change the map as well (1,024 instead of 256 calibration windows).", "",
         f"**Raw {SHAPE} medians, now / published, and Δ:**", "",
         "| model | " + " | ".join(NAMES[p] for p in TABLE_ROWS) + " |", "|---" + "|---:" * len(TABLE_ROWS) + "|"]
    for m, t in MODELS:
        src = pub[PUBLISHED[m]]
        cells = []
        for p in TABLE_ROWS:
            if p not in src["ms"].get(m, {}):
                cells.append("—")
                continue
            new, old = rec["ms"][m][p][SHAPE], src["ms"][m][p][SHAPE]
            per_shape = [pct(rec["ms"][m][p][s], src["ms"][m][p][s]) for s in SHAPES]
            cmp.setdefault(m, {})[p] = dict(now_1x2048=new, published_1x2048=old, pct_1x2048=pct(new, old),
                                            median_pct_over_shapes=statistics.median(per_shape),
                                            range_pct_over_shapes=[min(per_shape), max(per_shape)])
            cells.append(f"{new:.2f} / {old:.2f} ({pct(new, old):+.2f} %)")
        L.append(f"| {t} ({PUBLISHED[m]}) | " + " | ".join(cells) + " |")
    L += ["", "**Median over the 8 shapes of Δ (now vs published), per policy:**", "",
          "| model | " + " | ".join(NAMES[p] for p in TABLE_ROWS) + " |", "|---" + "|---:" * len(TABLE_ROWS) + "|"]
    for m, t in MODELS:
        L.append(f"| {t} | " + " | ".join(f"{cmp[m][p]['median_pct_over_shapes']:+.2f} %" if p in cmp[m] else "—"
                                          for p in TABLE_ROWS) + " |")
    fixed = [cmp[m][p]["median_pct_over_shapes"] for m, _ in MODELS for p in ("bf16", "nvfp4", "fo6") if p in cmp[m]]
    L += ["", f"Map-free rows (BF16, NVFP4, FourOverSix; {len(fixed)} model-policy medians): "
          f"[{min(fixed):+.2f}, {max(fixed):+.2f}] %, median {statistics.median(fixed):+.2f} %.", "",
          "**The paired FlipQuant vs FourOverSix ratio, median over shapes, now (1k maps) and published (256 maps):**", "",
          "| model | " + " | ".join(f"{NAMES[k.split(' vs ')[0]]} vs {NAMES[k.split(' vs ')[1]]}"
                                   for k in ("fq-16x64 vs fo6", "fq-8x64 vs fo6", "fq-256x64 vs fo6")) + " |",
          "|---|---:|---:|---:|"]
    ratio = {}
    for m, t in MODELS:
        src, cells = pub[PUBLISHED[m]], []
        for k in ("fq-16x64 vs fo6", "fq-8x64 vs fo6", "fq-256x64 vs fo6"):
            now, old = rec["ratios"][m][k]["median_over_shapes"], src["ratios"][m][k]["median_over_shapes"]
            ratio.setdefault(m, {})[k] = dict(now=now, published=old, diff_pp=now - old)
            cells.append(f"{now:+.2f} % / {old:+.2f} % ({now - old:+.2f} pp)")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    out["vs_published"] = dict(sources=dict(A=P.A_COMMIT, W=W_COMMIT), cells=cmp, fq_vs_fo6=ratio,
                               map_free_median_range=[min(fixed), max(fixed)], map_free_median=statistics.median(fixed))
    return L + [""]


def e0m3_shares():
    out = {}
    for m, _ in MODELS:
        for u, tag in P.UNITS.items():
            j = json.loads((G.RUN / "calib_size" / m / f"{tag}_1024.job.json").read_text())
            out.setdefault(m, {})[u] = dict(e0m3_share=j["e0m3_share"], map_sha256=j["map_file_sha256"])
    return out


# ---------------------------------------------------------------------------------------------- item 2: memory
def memory_section(out):
    L = ["# Part AC item 2: memory on the RTX PRO 6000 at 1k (the paper's memory paragraph)", "",
         "Part B's probe (mem_probe.py, flipquant's loader and install, build_V) on flipquant calib-size d940fbf, with "
         "the 1k maps; fo6-wB added to B's six policies. PROTOCOL.md part AC item 2.", "",
         "- **Weights:** every parameter and buffer on the GPU after the load and install.",
         "- **After load:** torch.cuda.memory_allocated.",
         "- **Peak:** torch.cuda.max_memory_allocated during one 1x2048 eager prefill.", ""]
    a = json.loads((RUN / "memory_anchor_check.json").read_text())
    L += ["## Step 0: the memory anchor (Mistral, FlipQuant 16x64 with the release map, against part B's record)", "",
          f"- Installed quantized linears equal: {a['installed_equal']}; whole model equal: {a['model_equal']}.",
          f"- After load Δ {a['after_load_diff_gib']:+.4f} GiB; 1x2048 peak Δ {a['peak_diff_gib']:+.4f} GiB "
          f"(tolerance ±{a['tol_gib']} GiB). **{'PASS' if a['passed'] else 'FAIL'}.**", "",
          "## The panel", "",
          "| model | policy | weights | quantized linears / BF16 | whole model / BF16 | after load | 1x2048 peak |",
          "|---|---|---:|---:|---:|---:|---:|"]
    mem, claims = {}, {}
    for m, t in MODELS:
        for p in POLS:
            f = RUN / "memory" / m / f"{p}.json"
            if f.exists():
                r = json.loads(f.read_text())
                q = r["quantized_linears"]
                mem.setdefault(m, {})[p] = dict(model_bytes=r["model_bytes"], installed_bytes=q["installed_bytes"],
                                                bf16_bytes=q["bf16_bytes"], installed_over_bf16=q["installed_over_bf16"],
                                                allocated_after_load=r["allocated_after_load"],
                                                peak_prefill_1x2048=r["peak_prefill_1x2048"], modules=q["modules"])
        mm = mem.get(m, {})
        if "bf16" not in mm:
            continue
        for p in POLS:
            if p in mm:
                r = mm[p]
                L.append(f"| {t} | {p} | {r['model_bytes'] / GIB:.2f} GiB | {r['installed_over_bf16']:.4f} | "
                         f"{r['model_bytes'] / mm['bf16']['model_bytes']:.3f} | {r['allocated_after_load'] / GIB:.2f} GiB | "
                         f"{r['peak_prefill_1x2048'] / GIB:.2f} GiB |")
        if all(p in mm for p in ("nvfp4", "fo6", "fo6-wB", *FQS)):
            c = dict(installed_equal={}, model_equal={}, after_load_max_diff_gib=0.0, peak_max_diff_gib=0.0)
            for fq in FQS:
                for ref in ("nvfp4", "fo6", "fo6-wB"):
                    x, y = mm[fq], mm[ref]
                    c["installed_equal"][f"{fq} vs {ref}"] = x["installed_bytes"] == y["installed_bytes"]
                    c["model_equal"][f"{fq} vs {ref}"] = x["model_bytes"] == y["model_bytes"]
                    c["after_load_max_diff_gib"] = max(c["after_load_max_diff_gib"],
                                                      abs(x["allocated_after_load"] - y["allocated_after_load"]) / GIB)
                    c["peak_max_diff_gib"] = max(c["peak_max_diff_gib"],
                                                 abs(x["peak_prefill_1x2048"] - y["peak_prefill_1x2048"]) / GIB)
            c["byte_equal"] = all(c["installed_equal"].values()) and all(c["model_equal"].values())
            c["within_tolerance"] = c["after_load_max_diff_gib"] <= P.MEM_TOL_GIB and c["peak_max_diff_gib"] <= P.MEM_TOL_GIB
            c["installed_bytes"] = mm["fq-16x64"]["installed_bytes"]
            c["bf16_over_installed"] = mm["bf16"]["installed_bytes"] / mm["fq-16x64"]["installed_bytes"]
            c["bf16_over_model"] = mm["bf16"]["model_bytes"] / mm["fq-16x64"]["model_bytes"]
            # against the published 256-window record (part B; part Z for the Llamas)
            old = {}
            for p in POLS:
                f = G.RUN / "memory" / m / f"{p}.json"
                if f.exists() and p in mm:
                    o = json.loads(f.read_text())
                    old[p] = dict(installed_equal=o["quantized_linears"]["installed_bytes"] == mm[p]["installed_bytes"],
                                  model_equal=o["model_bytes"] == mm[p]["model_bytes"],
                                  after_load_diff_gib=(mm[p]["allocated_after_load"] - o["allocated_after_load"]) / GIB,
                                  peak_diff_gib=(mm[p]["peak_prefill_1x2048"] - o["peak_prefill_1x2048"]) / GIB)
            c["vs_published"] = old
            claims[m] = c
    L += ["", "## The paper's memory claims, per model", ""]
    for m, t in MODELS:
        c = claims.get(m)
        if c is None:
            L.append(f"- {t}: incomplete")
            continue
        vp = c["vs_published"]
        L += [f"- **{t}:**",
              f"  - FlipQuant 8x64 / 16x64 / 256x64 install byte-identical quantized linears and whole models to NVFP4, "
              f"FourOverSix and FourOverSix (wB): **{'yes' if c['byte_equal'] else 'NO'}** "
              f"({c['installed_bytes']:,} B of quantized linears; BF16 / FlipQuant = {c['bf16_over_installed']:.2f}x for "
              f"the linears, {c['bf16_over_model']:.2f}x for the whole model).",
              f"  - After load and at the 1x2048 peak, FlipQuant differs from them by at most "
              f"{c['after_load_max_diff_gib']:.4f} / {c['peak_max_diff_gib']:.4f} GiB. Within 0.02 GiB: "
              f"**{'yes' if c['within_tolerance'] else 'NO'}**.",
              f"  - Against the published 256 record (B / Z; {len(vp)} policies): bytes equal on "
              f"{sum(v['installed_equal'] and v['model_equal'] for v in vp.values())} of {len(vp)}; after load "
              f"[{min(v['after_load_diff_gib'] for v in vp.values()):+.4f}, {max(v['after_load_diff_gib'] for v in vp.values()):+.4f}] GiB; "
              f"peak [{min(v['peak_diff_gib'] for v in vp.values()):+.4f}, {max(v['peak_diff_gib'] for v in vp.values()):+.4f}] GiB."]
    allc = list(claims.values())
    if len(allc) == len(MODELS):
        L += ["", f"**All six models:** byte-identical installs {'yes' if all(c['byte_equal'] for c in allc) else 'NO'}; "
              f"max after-load difference {max(c['after_load_max_diff_gib'] for c in allc):.4f} GiB, max peak difference "
              f"{max(c['peak_max_diff_gib'] for c in allc):.4f} GiB; BF16 / FlipQuant whole model "
              f"{min(c['bf16_over_model'] for c in allc):.2f}–{max(c['bf16_over_model'] for c in allc):.2f}x."]
    out["memory"], out["memory_claims"], out["memory_anchor"] = mem, claims, a
    return L


def item2():
    out_dir = OUT / "item2"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_latency("latency", [m for m, _ in MODELS])
    rec = latency_tables(data)
    libs = G.manifests()
    checks = record_checks(data, libs)
    out = dict(item=2, code=dict(flipquant="calib-size d940fbf", kernels="build_V"), latency=rec, checks=checks,
               gpu_state=gpu_summary(), record_gpu=record_gpu(data), maps=e0m3_shares())
    n = sum(len(rr) for x in data.values() for rr in x.values())
    gs = out["gpu_state"]
    L = ["# Part AC item 2: tab:latency SM120 rows at 1k (CUDA-graph prefill on the RTX PRO 6000)", "",
         "PROTOCOL.md part AC item 2: part A / W's protocol unchanged (flipquant `evaluation.latency prefill`; shapes "
         "1x128 … 1x8192 and 4x2048; 7 repetitions; eager and CUDA graph per process; 5 rounds; one process per model / "
         "policy / round, in the order `random.Random(20260928 + r).shuffle`; build_V). Code: flipquant calib-size d940fbf "
         "(its timed path, tile table and install byte-identical to part W's 6f5ad27). The FlipQuant rows use the 1k maps "
         "of part AA / amendment 6 (sha256-checked against their job records).", "",
         f"- **Records:** {n} of {len(MODELS) * len(POLS) * 5}; A's checks re-applied: "
         f"{checks['records'] - len(checks['failures'])} pass, {len(checks['failures'])} fail.",
         f"- **Session:** {gs['jobs']} jobs from {gs['first_utc']} to {gs['last_utc']}, {gs['boots']} boot; driver "
         f"{', '.join(gs['drivers'])}; power limit {', '.join(gs['power_limits_w'])} W; max SM clock "
         f"{', '.join(gs['max_sm_mhz'])} MHz; nothing locked or profiled.",
         f"- **GPU before each job** (read-only nvidia-smi): {gs['pre_job_power_w'][0]:.1f}–{gs['pre_job_power_w'][1]:.1f} W, "
         f"{gs['pre_job_temp_c'][0]:.0f}–{gs['pre_job_temp_c'][1]:.0f} °C, SM {gs['pre_job_sm_mhz'][0]:.0f}–"
         f"{gs['pre_job_sm_mhz'][1]:.0f} MHz, pstates {', '.join(gs['pre_job_pstates'])}. In the prefill records: start "
         f"{out['record_gpu']['start_temp_c'][0]:.0f}–{out['record_gpu']['start_temp_c'][1]:.0f} °C, end "
         f"{out['record_gpu']['end_temp_c'][0]:.0f}–{out['record_gpu']['end_temp_c'][1]:.0f} °C.",
         "- **E0M3 tile shares of the 1k maps** (8x64 / 16x64 / 256x64): " + "; ".join(
             f"{t} {100 * out['maps'][m]['8x64']['e0m3_share']:.1f} / {100 * out['maps'][m]['16x64']['e0m3_share']:.1f} / "
             f"{100 * out['maps'][m]['256x64']['e0m3_share']:.1f} %" for m, t in MODELS) + ".", "",
         "Medians over rounds of each process's median (ms); ratios paired within rounds, median [min, max] over rounds.", ""]
    L += anchor_section(out)
    wf = RUN / "warmup" / "mistral-7b_fo6_4x2048.json"
    if wf.exists():
        w = json.loads(wf.read_text())
        out["warmup"] = dict(reps=w.get("reps"),
                             start_temp_c=float(w["gpu"]["temperature.gpu"]), end_temp_c=float(w["gpu_end"]["temperature.gpu"]),
                             eager_4x2048_median_ms=w["eager"]["4x2048"]["ms"])
        L += [f"**Warm-up (amendment 1, unrecorded):** one Mistral fo6 4x2048 eager prefill process right before the panel "
              f"({w.get('reps')} repetitions, median {w['eager']['4x2048']['ms']:.2f} ms); the GPU went from "
              f"{out['warmup']['start_temp_c']:.0f} to {out['warmup']['end_temp_c']:.0f} °C. Its timings are not used.", ""]
    L += temperature_section(data, out)
    if len(rec["ms"]) == len(MODELS) and all(len(rec["ms"][m]) == len(POLS) for m, _ in MODELS):
        tab, raw = combined_section(rec, out)
        L += tab + summaries_section(rec, raw, out) + published_section(rec, out)
    for m, _ in MODELS:
        if m in rec["ms"]:
            L += model_section(rec, m)
    L += ["## Summary: the median over the 8 shapes of the paired ratio", "",
          "| model | " + " | ".join(f"{NAMES[a]} vs {NAMES[b]}" for a, b in COMPARE) + " |", "|---" + "|---:" * len(COMPARE) + "|"]
    for m, t in MODELS:
        if m in rec["ratios"]:
            L.append(f"| {t} | " + " | ".join(
                f"{rec['ratios'][m][f'{a} vs {b}']['median_over_shapes']:+.1f} %" if f"{a} vs {b}" in rec["ratios"][m]
                else "—" for a, b in COMPARE) + " |")
    (out_dir / "LATENCY.md").write_text("\n".join(L) + "\n")
    if (RUN / "memory_anchor_check.json").exists():
        (out_dir / "MEMORY.md").write_text("\n".join(memory_section(out)) + "\n")
    (out_dir / "item2.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(L[:12]))


# ---------------------------------------------------------------------------------------------- item 3: tab:ablation
G_COMMIT = "7c06fb3"                                    # G's published 256-window tab:ablation
ABL_ROWS = [("random", "Random"), ("act", "Activation-weighted"), ("oneshot", "One-shot gradient"), ("flipquant", "FlipQuant")]
CORPORA = (("wiki", "WikiText-2"), ("c4", "C4"))
ABL_MODELS = [("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2"), ("qwen3.8-27b", "Qwen3.8-27B")]
UNIT_LIST = ("8x64", "16x64", "256x64")


def load_done(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return r if "resources" in r else None


def nll1k(model, unit, row):
    """{corpus: [per-window NLL]} at 1k (Random: the per-window mean of the three seeds); FlipQuant = part AA's / amendment
    6's 1k map record."""
    if row == "flipquant":
        recs = [load_done(G.RUN / "calib_size" / model / "eval" / "ppl" / f"{P.UNITS[unit]}_1024.json")]
    elif row == "random":
        recs = [load_done(RUN / "ablation" / model / unit / f"ppl_random{s}.json") for s in (0, 1, 2)]
    else:
        recs = [load_done(RUN / "ablation" / model / unit / f"ppl_{row}.json")]
    if any(r is None for r in recs):
        return None
    out = {}
    for c, _ in CORPORA:
        lists = [r["results"][c]["nll"] for r in recs]
        assert len({len(v) for v in lists}) == 1
        out[c] = [sum(x) / len(x) for x in zip(*lists)]
    return out


def paired(x, ref):
    import math
    assert len(x) == len(ref)
    d = [a - b for a, b in zip(x, ref)]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n)
    return dict(delta_milli=1e3 * m, two_se_milli=2e3 * se, n=n, significant=abs(m) > 2 * se)


def cell(p):
    return f"{p['delta_milli']:+.2f} ± {p['two_se_milli']:.2f}{'*' if p['significant'] else ''}"


def item3():
    import math
    out_dir = OUT / "item3"
    out_dir.mkdir(parents=True, exist_ok=True)
    pub = git_json(G_COMMIT, "results/paper_eval/ablation/ablation.json")
    rec = dict(cells={}, ppl={}, maps={}, fq_vs={}, published_256=dict(commit=G_COMMIT), vs_256={})
    step0 = json.loads((RUN / "item3_step0.json").read_text()) if (RUN / "item3_step0.json").exists() else None
    rec["step0"] = step0
    md = ["# Part AC item 3: tab:ablation at the 1k calibration (Nemotron-Nano-9B-v2, Qwen3.8-27B)", "",
          "G's protocol (PROTOCOL.md G; part AC item 3) on flipquant paper-1k-reruns 2739458 (calib-size d940fbf + "
          "tmopt_ext's opt-in --fit-windows / --teacher-host). Native (build_V, --kernel-set auto, paper convention, "
          "n16k64-fast). Every baseline flips exactly the 1k map's k_l tiles per layer. The activation-weighted and "
          "one-shot statistics use the 1k windows (the record's 128 + part AA's 896); Qwen3.8-27B's one-shot keeps the "
          "top-1000 teacher on the host. Cells: paired ΔNLL vs FourOverSix, x 1e-3 nats/token, ± 2 SE (* = beyond 2 "
          "SE); Random = the per-window mean of seeds 0 / 1 / 2. FourOverSix: part F's record (part AA's re-measurement "
          "is equal per window); FlipQuant: the 1k maps' records (part AA / amendment 6).", ""]
    if step0:
        sm, rb = step0.get("smoke", {}), step0.get("rebuild", {})
        md += ["## Step 0", "",
               "- **Smoke at 1k** (random-init 3-layer models, 2 optimizer steps, `--fit-windows 1024`): " + "; ".join(
                   f"{m}: {'pass' if r['passed'] else 'FAIL'} (extension {r['extension_windows']} windows, token hashes "
                   f"equal part AA's: {r['extension_equal_part_aa']}; teacher on {', '.join(r['teacher']['devices'])})"
                   for m, r in sm.items()) + ".",
               f"- **The 256 rebuild** (G's Nemotron 16x64 one-shot map on this code): tiles bit for bit "
               f"{'equal' if rb.get('tiles_equal') else 'DIFFERENT'} ({rb.get('modules')} modules, "
               f"{rb.get('e0m3_a')} E0M3 tiles); the summed-gradient file {'equal' if rb.get('grads_file_equal') else 'differs'}.",
               "- **Default command** (CPU): with the new options at their defaults, the trainer command equals calib-size "
               "d940fbf's for oneshot / train / train --rotate / train --candidates on five models (20 of 20).", ""]
    tex = {}
    for c, cname in CORPORA:
        md += [f"## {cname}", "", "| model | map | 8x64 | 16x64 | 256x64 |", "|---|---|---:|---:|---:|"]
        lines = [f"% tab:ablation at 1k ({cname}): paired dNLL vs FourOverSix, x1e-3 nats/token, +-2SE; native sm120 "
                 "(build_V, kernel-set auto), per-token FourOverSix activations; 1,024 calibration windows"]
        for model, title in ABL_MODELS:
            fo6 = load_done(G.RUN / "pplfast" / model / "fo6.json")
            lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{title}}}}} \\\\")
            for row, label in ABL_ROWS:
                cells, tcells = [], []
                for u in UNIT_LIST:
                    x = nll1k(model, u, row)
                    if x is None or fo6 is None:
                        cells.append("TBD")
                        tcells.append("TBD")
                        continue
                    p = paired(x[c], fo6["results"][c]["nll"])
                    rec["cells"].setdefault(model, {}).setdefault(row, {}).setdefault(u, {})[c] = p
                    rec["ppl"].setdefault(model, {}).setdefault(row, {}).setdefault(u, {})[c] = math.exp(sum(x[c]) / len(x[c]))
                    old = pub["cells"].get(model, {}).get(row, {}).get(u, {}).get(c)
                    if old:
                        rec["vs_256"].setdefault(model, {}).setdefault(row, {}).setdefault(u, {})[c] = dict(
                            now=p["delta_milli"], published=old["delta_milli"], diff=p["delta_milli"] - old["delta_milli"])
                    cells.append(cell(p))
                    tcells.append(f"${p['delta_milli']:.1f}$\\,{{\\scriptsize$\\pm${p['two_se_milli']:.1f}}}")
                md.append(f"| {title} | {label} | " + " | ".join(cells) + " |")
                lines.append(f"{label} & " + " & ".join(tcells) + " \\\\")
        md.append("")
        tex[c] = "\n".join(lines) + "\n"
        (out_dir / f"table_ablation_{c}.tex").write_text(tex[c])
    md += ["## Against the published 256-window table (G, 7c06fb3)", "",
           "Each cell's ΔNLL vs FourOverSix (x 1e-3) at 1k, then at 256, WikiText-2 / C4. The FourOverSix reference is the "
           "same record in both.", "", "| model | map | 8x64 | 16x64 | 256x64 |", "|---|---|---:|---:|---:|"]
    for model, title in ABL_MODELS:
        for row, label in ABL_ROWS:
            v = rec["vs_256"].get(model, {}).get(row, {})
            md.append(f"| {title} | {label} | " + " | ".join(
                " / ".join(f"{v[u][c]['now']:+.1f} ({v[u][c]['published']:+.1f})" for c, _ in CORPORA) if u in v else "TBD"
                for u in UNIT_LIST) + " |")
    md += ["", "## PPL per cell (WikiText-2 / C4)", "", "| model | map | 8x64 | 16x64 | 256x64 |", "|---|---|---:|---:|---:|"]
    for model, title in ABL_MODELS:
        fo6 = load_done(G.RUN / "pplfast" / model / "fo6.json")
        if fo6:
            md.append(f"| {title} | FourOverSix | " + " | ".join(
                " / ".join(f"{math.exp(sum(fo6['results'][c]['nll']) / len(fo6['results'][c]['nll'])):.4f}"
                           for c, _ in CORPORA) for _ in UNIT_LIST) + " |")
        for row, label in ABL_ROWS:
            pp = rec["ppl"].get(model, {}).get(row, {})
            md.append(f"| {title} | {label} | " + " | ".join(
                " / ".join(f"{pp[u][c]:.4f}" for c, _ in CORPORA) if u in pp else "TBD" for u in UNIT_LIST) + " |")
    md += ["", "## FlipQuant vs each baseline (paired ΔNLL FlipQuant − baseline, x 1e-3, ± 2 SE; negative = FlipQuant "
           "better; WikiText-2; C4)", "", "| model | unit | vs Random | vs Activation-weighted | vs One-shot |",
           "|---|---|---:|---:|---:|"]
    for model, title in ABL_MODELS:
        for u in UNIT_LIST:
            fq = nll1k(model, u, "flipquant")
            if fq is None:
                continue
            cells = []
            for row in ("random", "act", "oneshot"):
                b = nll1k(model, u, row)
                if b is None:
                    cells.append("TBD")
                    continue
                ps = [paired(fq[c], b[c]) for c, _ in CORPORA]
                rec["fq_vs"].setdefault(model, {}).setdefault(u, {})[row] = dict(zip(("wiki", "c4"), ps))
                cells.append("; ".join(cell(q) for q in ps))
            md.append(f"| {title} | {u} | " + " | ".join(cells) + " |")
    for model, _ in ABL_MODELS:
        for u in UNIT_LIST:
            d = RUN / "ablation" / model / u
            for m in ("act", "oneshot"):
                f = d / f"{m}.json"
                if not f.exists():
                    continue
                r = json.loads(f.read_text())
                info = dict(status=r.get("status"))
                if m == "act":
                    info.update(windows=(r.get("data") or {}).get("fit_windows"),
                                layers_short_of_positive=r.get("layers_short_of_positive"))
                else:
                    meta = r.get("meta", {})
                    info.update(steps=(meta.get("oneshot") or {}).get("steps"), fit_windows=meta.get("fit_windows"),
                                layers_short_of_negative=r.get("layers_short_of_negative"),
                                non_release=meta.get("non_release"),
                                settings={k: (meta.get("settings") or {}).get(k) for k in ("fit_windows", "teacher_topk",
                                                                                          "epochs", "lr", "init_logit")})
                rec["maps"].setdefault(model, {}).setdefault(u, {})[m] = info
    md += ["", "## The maps' records", "", "| model | unit | act windows | act layers short of positives | one-shot steps | "
           "one-shot fit windows | one-shot layers short of negatives |", "|---|---|---:|---:|---:|---:|---:|"]
    for model, title in ABL_MODELS:
        for u in UNIT_LIST:
            mm = rec["maps"].get(model, {}).get(u, {})
            a, o = mm.get("act", {}), mm.get("oneshot", {})
            md.append(f"| {title} | {u} | {a.get('windows')} | {a.get('layers_short_of_positive')} | {o.get('steps')} | "
                      f"{o.get('fit_windows')} | {o.get('layers_short_of_negative')} |")
    (out_dir / "item3.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out_dir / "ABLATION.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:30]))


# ---------------------------------------------------------------------------------------------- item 4: tab:ptq
PTQ_MODELS = ABL_MODELS


def ptq_rows():
    """(key, method label, format label, PPL record, GSM8K record, the 256 counterpart (PPL, GSM8K)) per model."""
    R, GRUN = RUN / "ptq", G.RUN
    NL = GRUN / "ptq_gsm8k_lmeval"
    out = {}
    for m, _ in PTQ_MODELS:
        out[m] = [
            ("bf16", "BF16", "--", GRUN / "pplfast" / m / "bf16.json", GRUN / "ptq_round3" / "bf16" / m / "bf16.json", None),
            ("rtn_nvfp4", "RTN", "NVFP4", GRUN / "pplfast" / m / "nvfp4.json", NL / m / "rtn_nvfp4.json", None),
            ("rtn_fo6", "RTN", "FourOverSix", GRUN / "pplfast" / m / "fo6.json", NL / m / "rtn_fo6.json", None),
            ("rtn_fq-16x64", "RTN", "FlipQuant (16x64)", GRUN / "calib_size" / m / "eval" / "ppl" / "fq16_1024.json",
             GRUN / "calib_size" / m / "eval" / "gsm8k" / "fq16_1024.json",
             (GRUN / "pplfast" / m / "fq-16x64.json", NL / m / "rtn_fq-16x64.json")),
            ("gptq_nvfp4", "GPTQ (act.)", "NVFP4", R / m / "gptq_nvfp4.json", R / "gsm8k" / m / "gptq_nvfp4.json",
             (GRUN / "ptq_round3" / m / "gptq_nvfp4.json", GRUN / "ptq_round3" / "gsm8k" / m / "gptq_nvfp4.json")),
            ("gptq_fo6", "GPTQ (act.)", "FourOverSix", R / m / "gptq_fo6.json", R / "gsm8k" / m / "gptq_fo6.json",
             (GRUN / "ptq_round3" / m / "gptq_fo6.json", GRUN / "ptq_round3" / "gsm8k" / m / "gptq_fo6.json")),
            ("gptq_fq-16x64", "GPTQ (act.)", "FlipQuant (16x64)", R / m / "gptq_fq-16x64.json",
             R / "gsm8k" / m / "gptq_fq-16x64.json",
             (GRUN / "ptq_round3" / m / "gptq_fq-16x64.json", GRUN / "ptq_round3" / "gsm8k" / m / "gptq_fq-16x64.json")),
            ("gptqcand_fq-16x64", "GPTQ cand. + map", "FlipQuant (16x64)", R / m / "gptqcand_fq-16x64.json",
             R / "gsm8k" / m / "gptqcand_fq-16x64.json",
             (GRUN / "ptq_round3" / m / "gptqcand_fq-16x64.json",
              GRUN / "ptq_round3" / "gsm8k" / m / "gptqcand_fq-16x64.json")),
            ("hadamard_nvfp4", "Hadamard", "NVFP4", GRUN / "ptq" / m / "hadamard_nvfp4.json", NL / m / "hadamard_nvfp4.json",
             None),
            ("hadamard_fo6", "Hadamard", "FourOverSix", GRUN / "ptq" / m / "hadamard_fo6.json", NL / m / "hadamard_fo6.json",
             None),
            ("hadamard_fq-16x64", "Hadamard", "FlipQuant (16x64)", R / m / "hadamard_fq-16x64.json",
             R / "gsm8k" / m / "hadamard_fq-16x64.json",
             (GRUN / "ptq" / m / "hadamard_fq-16x64.json", NL / m / "hadamard_fq-16x64.json")),
        ]
    return out


def item4():
    import math
    import report_gsm8k as RM
    import report_gsm8k_lmeval as RN
    out_dir = OUT / "item4"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = dict(rows={}, paired={}, gsm8k_paired={}, recheck={}, checks={}, step0=None)
    s0 = RUN / "ptq" / "step0" / "step0.json"
    rec["step0"] = json.loads(s0.read_text()) if s0.exists() else None

    def ppl_rec(path):
        r = load_done(path)
        return r

    def ppl_of(r, c):
        v = r["results"][c]["nll"]
        return math.exp(sum(v) / len(v))

    def gsm(path):
        rep = RN.load(path) if path else None
        return (rep, rep["tasks"][RN.TASK]["examples"]) if rep else (None, None)

    def pair_gsm(a, b):
        return {f: RM.paired({d: dict(correct=v) for d, v in RN.per_filter(a, f).items()},
                             {d: dict(correct=v) for d, v in RN.per_filter(b, f).items()}) for f in RN.FILTERS}

    def fmt_pp(p):
        return (f"{100 * p['delta']:+.2f} ± {100 * p['two_se']:.2f}{'*' if p['significant'] else ''} "
                f"({p['a_only']} / {p['b_only']}, p {p['mcnemar_p']:.3g})")

    md = ["# Part AC item 4: tab:ptq at the 1k calibration (Nemotron-Nano-9B-v2, Qwen3.8-27B; 16x64)", "",
          "Round 3's protocol (PROTOCOL.md part O) and part H's Hadamard rows with the 1k windows (part AC item 4): native "
          "sm120 (build_V, --kernel-set auto, paper convention), n16k64-fast; GPTQ with activation quantization on during "
          "calibration on the 1,024 windows (the release record's 128 + part AA's 896); the GPTQ-candidate map and the "
          "Hadamard map trained at 1k (Qwen3.8-27B with the teacher on the host). flipquant paper-1k-reruns 2739458. "
          "ΔNLL: paired per window vs RTN FourOverSix, x 1e-3 ± 2 SE (* beyond 2 SE). GSM8K: part N's protocol (lm-eval "
          "0.4.11 gsm8k_llama, chat template, greedy, 1,024 tokens, thinking off, 1,319 problems) at batch 64; strict-match "
          "± 2 SE; paired per problem, A − B in pp ± 2 SE (A-only / B-only, McNemar exact p). Reused rows: BF16 (part F, "
          "round 3's O0), RTN NVFP4 / FourOverSix (part F, part N), RTN FlipQuant at 1k (part AA), Hadamard NVFP4 / "
          "FourOverSix (part H, part N).", ""]
    if rec["step0"]:
        s = rec["step0"]
        md += ["## Step 0", "", f"Round 3's Nemotron GPTQ NVFP4 job (256 windows) on this code: codes_sha256 "
               f"{'equal to' if s['codes_equal'] else 'DIFFERENT from'} round 3's (`{s['codes_sha256'][:16]}`); data hash "
               f"equal: {s['data_equal']}; per-window NLL equal: {s['nll_equal']}.", ""]
    md += ["## The table", "", "| model | method | format | WikiText-2 | C4 | ΔNLL wiki vs RTN FO6 | ΔNLL C4 vs RTN FO6 | "
           "GSM8K strict (%) | 256: WikiText-2 / C4 / GSM8K |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    tex = ["% tab:ptq at the 1k calibration (part AC item 4): WikiText-2 / C4 PPL (native sm120, 16x64) and GSM8K "
           "strict-match (%), lm-eval gsm8k_llama, thinking off; GPTQ with activation quantization on during calibration; "
           "1,024 calibration windows"]
    ex_all = {}
    for m, title in PTQ_MODELS:
        ref = ppl_rec(G.RUN / "pplfast" / m / "fo6.json")
        tex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for key, method, fmt, pf, gf, old in ptq_rows()[m]:
            r = ppl_rec(pf)
            rep, ex = gsm(gf)
            if ex is not None:
                ex_all[(m, key)] = ex
            row = dict(method=method, format=fmt)
            cells = []
            if r is not None:
                row["ppl"] = {c: ppl_of(r, c) for c, _ in CORPORA}
                row["vs_rtn_fo6"] = {c: paired(r["results"][c]["nll"], ref["results"][c]["nll"]) for c, _ in CORPORA} \
                    if ref and key != "rtn_fo6" else None
                pt = r.get("ptq") or {}
                row["ptq"] = {k: pt.get(k) for k in ("method", "weight", "act", "windows", "hessian_input", "propagate",
                                                     "calib", "data_sha256", "codes_sha256", "map_sha256", "rotate")}
            if rep is not None:
                sm = RN.summary(rep)
                row["gsm8k"] = sm
                pol = (rep.get("policy") or {}).get("ptq") or {}
                row["gsm8k_codes_sha256"] = pol.get("codes_sha256")
            if old:
                o_ppl, o_gsm = ppl_rec(old[0]), gsm(old[1])[0]
                row["published_256"] = dict(ppl={c: ppl_of(o_ppl, c) for c, _ in CORPORA} if o_ppl else None,
                                            gsm8k=RN.summary(o_gsm)["strict_match"]["accuracy"] if o_gsm else None)
            rec["rows"].setdefault(m, {})[key] = row
            pp = row.get("ppl")
            vs = row.get("vs_rtn_fo6") or {}
            g = row.get("gsm8k")
            o = row.get("published_256") or {}
            md.append(f"| {title} | {method} | {fmt} | " + (f"{pp['wiki']:.4f} | {pp['c4']:.4f}" if pp else "TBD | TBD")
                      + " | " + (" | ".join(cell(vs[c]) for c, _ in CORPORA) if vs else "— | —")
                      + " | " + (f"{100 * g['strict_match']['accuracy']:.2f} ± {100 * g['strict_match']['two_se']:.2f}"
                                 if g else ("—" if key == "recheck" else "TBD"))
                      + " | " + ((" / ".join(f"{o['ppl'][c]:.4f}" for c, _ in CORPORA) if o.get("ppl") else "—")
                                 + (f" / {100 * o['gsm8k']:.2f}" if o.get("gsm8k") is not None else "") if old else "unchanged")
                      + " |")
            fmt_tex = fmt.replace("16x64", "16$\\times$64")
            tex.append(f"{method} & {fmt_tex} & "
                       + (f"{pp['wiki']:.2f} & {pp['c4']:.2f}" if pp else "TBD & TBD") + " & "
                       + (f"{100 * g['strict_match']['accuracy']:.1f}" if g else "TBD") + " \\\\")
        # the re-check: the 1k RTN map composed with the 1k GPTQ candidates, against the retrained map
        rc = ppl_rec(RUN / "ptq" / m / "recheck_rtnmap_gptqcand.json")
        rt = ppl_rec(RUN / "ptq" / m / "gptqcand_fq-16x64.json")
        if rc is not None:
            rec["recheck"][m] = dict(ppl={c: ppl_of(rc, c) for c, _ in CORPORA},
                                     vs_rtn_fo6={c: paired(rc["results"][c]["nll"], ref["results"][c]["nll"])
                                                 for c, _ in CORPORA} if ref else None,
                                     retrained_minus_recheck={c: paired(rt["results"][c]["nll"], rc["results"][c]["nll"])
                                                              for c, _ in CORPORA} if rt else None)
    md += ["", "## The re-check: the 1k RTN map on the 1k GPTQ candidates (no retraining)", "",
           "| model | WikiText-2 | C4 | ΔNLL vs RTN FO6 (wiki; C4) | retrained map − re-check (wiki; C4) |",
           "|---|---:|---:|---:|---:|"]
    for m, title in PTQ_MODELS:
        x = rec["recheck"].get(m)
        if not x:
            md.append(f"| {title} | TBD | TBD | | |")
            continue
        md.append(f"| {title} | {x['ppl']['wiki']:.4f} | {x['ppl']['c4']:.4f} | "
                  + ("; ".join(cell(x['vs_rtn_fo6'][c]) for c, _ in CORPORA) if x["vs_rtn_fo6"] else "—") + " | "
                  + ("; ".join(cell(x['retrained_minus_recheck'][c]) for c, _ in CORPORA) if x["retrained_minus_recheck"]
                     else "—") + " |")
    md += ["", "## GSM8K, paired per problem", "", "| model | A vs B | strict-match Δ (pp) | flexible-extract Δ (pp) |",
           "|---|---|---:|---:|"]
    pairs = [("gptq_nvfp4", "gptq_fo6"), ("gptq_fq-16x64", "gptq_fo6"), ("gptqcand_fq-16x64", "gptq_fo6"),
             ("gptqcand_fq-16x64", "gptq_fq-16x64"), ("gptq_fq-16x64", "rtn_fq-16x64"), ("gptqcand_fq-16x64", "rtn_fq-16x64"),
             ("hadamard_fq-16x64", "hadamard_fo6"), ("hadamard_fq-16x64", "rtn_fq-16x64"), ("rtn_fq-16x64", "rtn_fo6")]
    for m, title in PTQ_MODELS:
        for a, b in pairs:
            if (m, a) in ex_all and (m, b) in ex_all:
                pg = pair_gsm(ex_all[(m, a)], ex_all[(m, b)])
                rec["gsm8k_paired"].setdefault(m, {})[f"{a} vs {b}"] = pg
                md.append(f"| {title} | {a} vs {b} | {fmt_pp(pg['strict_match'])} | {fmt_pp(pg['flexible_extract'])} |")
    # checks
    for m, _ in PTQ_MODELS:
        rows = rec["rows"].get(m, {})
        g = [rows[k]["ptq"] for k in ("gptq_nvfp4", "gptq_fo6", "gptq_fq-16x64") if rows.get(k, {}).get("ptq")]
        if g:
            rec["checks"][f"{m}: GPTQ quantized Hessian and propagation, 1,024 windows, one data hash"] = (
                all(x["hessian_input"] == "quantized" and x["windows"] == 1024 and x["calib"] == "release" for x in g)
                and len({x["data_sha256"] for x in g}) == 1)
        for k in ("gptq_nvfp4", "gptq_fo6", "gptq_fq-16x64"):
            row = rows.get(k, {})
            if row.get("gsm8k") and row.get("ptq"):
                rec["checks"][f"{m}: {k} GSM8K codes = PPL codes"] = row["gsm8k_codes_sha256"] == row["ptq"]["codes_sha256"]
    md += ["", "## Checks", "", "| check | passed |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in rec["checks"].items()]
    (out_dir / "item4.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (out_dir / "REPORT.md").write_text("\n".join(md) + "\n")
    (out_dir / "table_ptq.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    {"item2": item2, "item3": item3, "item4": item4}[sys.argv[1]]()

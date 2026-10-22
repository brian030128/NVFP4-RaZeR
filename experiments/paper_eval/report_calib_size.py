"""Part AA (PROTOCOL.md): the calibration-size report. results/paper_eval/calib_size/: REPORT.md, calib_size.json,
WINDOWS.md, STEP0.md, PROBES.md. Reads RUN/calib_size/ and the part X / Y / final-table records; writes nothing else.

    python report_calib_size.py
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_gsm8k6 as RY  # noqa: E402
import report_zeroshot6 as RX  # noqa: E402
import run_calib_size as CS  # noqa: E402
import run_gpu as G  # noqa: E402
import run_gsm8k6 as Y  # noqa: E402
import run_zeroshot6 as Z  # noqa: E402
import tables_cpu as T  # noqa: E402
import tables_final as TF  # noqa: E402

OUT = G.RZ / "results" / "paper_eval" / "calib_size"
TITLES = {"llama3.2-1b-instruct": "Llama-3.2-1B-Instruct", "phi4-14b": "Phi-4", "qwen3.8-27b": "Qwen3.8-27B",
          "mistral-7b": "Mistral-7B-Instruct-v0.3", "llama3.1-8b-instruct": "Llama-3.1-8B-Instruct",
          "nemotron-nano-9b-v2": "Nemotron-Nano-9B-v2", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B"}
SHORT = {"llama3.2-1b-instruct": "Llama-1B", "phi4-14b": "Phi-4", "qwen3.8-27b": "Qwen27B", "mistral-7b": "Mistral",
         "llama3.1-8b-instruct": "Llama-8B", "nemotron-nano-9b-v2": "Nemotron", "qwen3-1.7b": "Qwen3-1.7B",
         "qwen3-8b": "Qwen3-8B"}
SIZES = (1024, 2048)
GIB = 2 ** 30
#: the pilot's rows: (label, source); a part X / Y row name, or a calib-size eval name
BASE_ROWS = [("BF16", "bf16"), ("NVFP4", "nvfp4"), ("FourOverSix", "fo6"), ("IF4†", "if4"), ("MixFP4 (Zou)†", "zou"),
             ("GPTQ‡", "gptq"), ("FlipQuant 8x64", "fq-8x64"), ("FlipQuant 256x64", "fq-256x64")]
NEW_ROWS = [("FOCUS, 256", "focus256"), ("FOCUS, 1,024", "focus_1024"), ("FOCUS, 2,048", "focus_2048"),
            ("FlipQuant 16x64, 256", "fq16_256"), ("FlipQuant 16x64, 1,024", "fq16_1024"),
            ("FlipQuant 16x64, 2,048", "fq16_2048")]
#: amendment 5: GPTQ‡ recalibrated on the 1,024 / 2,048 windows; in the tables right after GPTQ‡ (256)
GPTQ_ROWS = [("GPTQ‡, 1,024", "gptq_1024"), ("GPTQ‡, 2,048", "gptq_2048")]
NEW = ("focus_1024", "fq16_1024", "focus_2048", "fq16_2048", "focus256_rerun", "focus256_local", "gptq_1024", "gptq_2048")
#: amendment 3: Qwen3.8-27B's FOCUS-256 retrained locally as its 1,024 / 2,048 runs (offload, micro 1), the
#: environment-matched reference; the cluster state (part Q) stays, marked as a different environment
CONTROL = ("FOCUS, 256 (local control, amendment 3)", "focus256_local")
CONTROL_COMPARISONS = [("FOCUS-256 local vs FOCUS-256 cluster", "focus256_local", "focus256"),
                       ("FQ16-256 vs FOCUS-256 local (FOCUS in this environment)", "fq16_256", "focus256_local"),
                       ("FOCUS-1k vs FOCUS-256 local (same environment)", "focus_1024", "focus256_local"),
                       ("FOCUS-2k vs FOCUS-256 local (same environment)", "focus_2048", "focus256_local"),
                       ("FQ16-1k vs FOCUS-256 local", "fq16_1024", "focus256_local")]
RERUN = ("FOCUS, 256 (retrained, step 0 d)", "focus256_rerun")


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def ok_eval(r):
    return r is not None and (r.get("status") == "complete" or ("resources" in r and "results" in r))


# ---------------------------------------------------------------------------------------------- the rows' records

def zs_record(model, name):
    """Zero-shot: part X's record for a baseline row; the 256 rows are part X's cells (their identity is step 0 (c));
    the new rows are calib-size's."""
    x = {"focus256": "focus", "fq16_256": "fq-16x64"}.get(name, name)
    if name in NEW:
        return load(CS.ROOT / model / "eval" / "zs" / f"{name}.json")
    return load(Z.ZS / "panel" / model / f"{x}.json")


def gsm_record(model, name):
    x = {"focus256": "focus", "fq16_256": "fq-16x64"}.get(name, name)
    if name in NEW:
        r = load(CS.ROOT / model / "eval" / "gsm8k" / f"{name}.json")
    elif model in CS.QWEN3:                 # amendment 1: Qwen3's native GSM8K baselines are this part's runs
        r = load(CS.ROOT / model / "eval" / "gsm8k" / f"{x}.json")
    else:
        r = load(Y.ROOT / model / f"{x}.json")
    return r if r is not None and r.get("status") == "complete" else None


def ppl_path(model, name):
    """PPL: the step-0 (c2) calib-size runs for FO6 / FQ16-256 / FOCUS-256 (one harness with the new rows), the new
    rows', and the final table's records for the other baselines."""
    if name in ("fo6", "fq16_256", "focus256") + NEW:
        return CS.ROOT / model / "eval" / "ppl" / f"{name}.json"
    mp = T.git_json(T.MAIN_PPL_COMMIT, "results/main_ppl/main_ppl.json")["models"]
    mkey = next(k for key, k, _ in TF.FINAL_MODELS if key == model)
    src = TF.sources(mp, model, mkey)
    key = {"fq-8x64": "fq_8x64", "fq-256x64": "fq_256x64"}.get(name, name)
    return Path(src[key]) if key in src else None


def ppl_nll(model, name):
    p = ppl_path(model, name)
    if p is None or not p.exists():
        return None
    r = load(p)
    if "results" in r and "resources" not in r:
        return None
    return T.nll_of(p)


# ---------------------------------------------------------------------------------------------- statistics

def zs_avg(r):
    return sum(RX.acc(r, t) for t, _ in RX.TASK_NAMES) / len(RX.TASK_NAMES)


def zs_task_paired(a, b, t):
    ka, kb = a["tasks"][t]["examples"], b["tasks"][t]["examples"]
    assert set(ka) == set(kb)
    c = sum(ka[k] > kb[k] for k in ka)
    d = sum(kb[k] > ka[k] for k in ka)
    return dict(diff=100 * (c - d) / len(ka), row_only=c, ref_only=d, p=RX.mcnemar(d, c), n=len(ka))


def fmt_pair(p):
    if p is None:
        return "TBD"
    return f"{p['diff']:+.2f}{'*' if p['p'] < 0.05 else ''} ({p['row_only']}/{p['ref_only']}, p={p['p']:.2g})"


def fmt_nll(p):
    if p is None:
        return "TBD"
    return f"{p['delta']:+.4f} ± {p['two_se']:.4f}{'*' if p['significant'] else ''}"


def compare(model, a, b):
    """a vs b: zero-shot pooled and per task, GSM8K strict, ΔNLL per corpus."""
    out = {}
    za, zb = zs_record(model, a), zs_record(model, b)
    if za is not None and zb is not None:
        out["zeroshot"] = RX.paired(za, zb)
        out["zeroshot_tasks"] = {t: zs_task_paired(za, zb, t) for t, _ in RX.TASK_NAMES}
    ga, gb = gsm_record(model, a), gsm_record(model, b)
    if ga is not None and gb is not None:
        out["gsm8k"] = RY.paired(ga, gb)
        out["gsm8k_flex"] = RY.paired(ga, gb, "flexible_extract")
    na, nb = ppl_nll(model, a), ppl_nll(model, b)
    if na is not None and nb is not None:
        out["nll"] = {c: T.paired(na[c], nb[c]) for c, _ in T.CORPORA}
    return out


K = {1024: "1k", 2048: "2k"}
COMPARISONS = ([("FQ16-256 vs FOCUS-256", "fq16_256", "focus256")]
               + [c for n in SIZES for c in (
                   (f"FQ16-{K[n]} vs FourOverSix", f"fq16_{n}", "fo6"), (f"FQ16-{K[n]} vs FQ16-256", f"fq16_{n}", "fq16_256"),
                   (f"FOCUS-{K[n]} vs FourOverSix", f"focus_{n}", "fo6"),
                   (f"FOCUS-{K[n]} vs FOCUS-256", f"focus_{n}", "focus256"),
                   (f"FQ16-{K[n]} vs FOCUS-{K[n]}", f"fq16_{n}", f"focus_{n}"),
                   (f"FQ16-{K[n]} vs IF4†", f"fq16_{n}", "if4"), (f"FQ16-{K[n]} vs MixFP4 (Zou)†", f"fq16_{n}", "zou"))]
               + [("FQ16-2k vs FQ16-1k", "fq16_2048", "fq16_1024"), ("FOCUS-2k vs FOCUS-1k", "focus_2048", "focus_1024"),
                  # the motivating losses against the fixed 256-window FOCUS reference (flipquant README, 2026-10-22)
                  ("FQ16-1k vs FOCUS-256", "fq16_1024", "focus256"), ("FQ16-2k vs FOCUS-256", "fq16_2048", "focus256"),
                  ("FQ16-256 vs GPTQ‡-256", "fq16_256", "gptq"), ("FQ16-1k vs GPTQ‡-256", "fq16_1024", "gptq"),
                  ("FQ16-2k vs GPTQ‡-256", "fq16_2048", "gptq"),
                  ("FQ16-256 vs IF4†", "fq16_256", "if4"), ("FQ16-256 vs MixFP4 (Zou)†", "fq16_256", "zou")]
               # amendment 5: GPTQ‡ against its 256, and FlipQuant against GPTQ‡ at the same size
               + [("GPTQ‡-1k vs GPTQ‡-256", "gptq_1024", "gptq"), ("GPTQ‡-2k vs GPTQ‡-256", "gptq_2048", "gptq"),
                  ("GPTQ‡-2k vs GPTQ‡-1k", "gptq_2048", "gptq_1024"), ("FQ16-1k vs GPTQ‡-1k", "fq16_1024", "gptq_1024"),
                  ("FQ16-2k vs GPTQ‡-2k", "fq16_2048", "gptq_2048")])


def table_rows():
    """The per-model table's rows: BASE_ROWS with amendment 5's GPTQ‡ rows after GPTQ‡ (256), then NEW_ROWS."""
    out = []
    for row in BASE_ROWS:
        out.append(row)
        if row[1] == "gptq":
            out += GPTQ_ROWS
    return out + NEW_ROWS


def gsm8k_stops(model, rows):
    """Amendment 3's GSM8K diagnostics per row: strict-match no-answer ([invalid]) count, generations ending at "The
    final [answer [is]]" (a cut before the number), mean / max generated tokens and budget hits."""
    import gzip
    import re
    out = {}
    for label, name in rows:
        r = gsm_record(model, name)
        if r is None:
            continue
        x = {"focus256": "focus", "fq16_256": "fq-16x64"}.get(name, name)
        if name in NEW or model in CS.QWEN3:
            path = CS.ROOT / model / "eval" / "gsm8k" / f"{x}.{Y.TASK}.jsonl.gz"
        else:
            path = Y.ROOT / model / f"{x}.{Y.TASK}.jsonl.gz"
        cut = 0
        with gzip.open(path, "rt") as f:
            for line in f:
                smp = json.loads(line)
                if smp["filter"] == "strict_match" and re.search(r"The final( answer( is)?)?$", smp["resps"][0][0].rstrip()):
                    cut += 1
        out[name] = dict(label=label, strict_no_answer=RY.no_answer(r), cut_at_final=cut, **RY.lengths(r))
    return out


def training_curves(model):
    """Amendment 4: per FlipQuant 16x64 map and size, the trainer's per-epoch training KL, E0M3 tile count and
    within-epoch flips (report.json epochs), and the final E0M3 share; for a size with a theta rerun, the theta record."""
    out = {}
    rel = (G.RUN / "llama_instruct" / model / "records" / "16x64" / "trainer_report.json" if model in Z.LLAMA
           else G.REL / model / "records" / "16x64" / "trainer_report.json")
    for n, path in [(256, rel)] + [(n, CS.ROOT / model / f"fq16_{n}" / "run" / "report.json") for n in SIZES]:
        r = load(path)
        if not r or not r.get("epochs"):
            continue
        tiles = r.get("tiles")
        out[n] = dict(epochs=[dict(epoch=e["epoch"], train_kl=e["train_kl"], e0m3=e["e0m3_units"], flips=e["hard_flips"])
                              for e in r["epochs"]], tiles=tiles,
                      final_e0m3=r.get("final_e0m3_units", r["epochs"][-1]["e0m3_units"]))
        th = load(CS.ROOT / model / f"theta_{n}.json")
        if th:
            out[n]["theta"] = th
    return out


BASELINES = ("bf16", "gptq", "if4", "zou")


def bf16_check(model, name="bf16"):
    """A baseline row (BF16, GPTQ‡, IF4†, MixFP4†; the user's requests, 2026-10-22): each value against its source
    record's own metric, and its protocol against the rows it sits next to (the FQ16-1k row): zero-shot sample digests
    and batch; GSM8K digest and batch (Qwen3.8-27B BF16 ran at 48, marked ◊); PPL window counts per corpus. GSM8K's
    marks from the job's own log: § the chat-template date pin (part Y amendment 3), ¤ expandable segments (amendment
    4). IF4 / MixFP4 GSM8K for the Qwen3 models was not run (not approved): not_run."""
    out = {}
    z, zn = zs_record(model, name), zs_record(model, "fq16_1024")
    if z and zn:
        metric = sum(z["tasks"][t]["metrics"][Z.PRIMARY[t] + ",none"] for t in Z.SIZES) / len(Z.SIZES) * 100
        out["zeroshot"] = dict(value=zs_avg(z), record_metric=metric, equal=abs(zs_avg(z) - metric) < 1e-9,
                               batch=z["batch_size"], batch_next=zn["batch_size"],
                               digests_equal=all(z["tasks"][t]["sample_digest"] == zn["tasks"][t]["sample_digest"]
                                                 for t in Z.SIZES))
        out["zeroshot"]["ok"] = (out["zeroshot"]["equal"] and out["zeroshot"]["digests_equal"]
                                 and out["zeroshot"]["batch"] == out["zeroshot"]["batch_next"])
    g, gn = gsm_record(model, name), gsm_record(model, "fq16_1024")
    if g is None and name in ("if4", "zou") and model in CS.QWEN3:
        out["gsm8k"] = dict(not_run=True, ok=True)
    if g and gn:
        metric = 100 * g["tasks"][Y.TASK]["metrics"]["exact_match,strict_match"]
        out["gsm8k"] = dict(value=RY.acc(g)[0], record_metric=metric, equal=abs(RY.acc(g)[0] - metric) < 1e-9,
                            batch=g["batch_size"], batch_next=gn["batch_size"],
                            digest_equal=g["tasks"][Y.TASK]["sample_digest"] == gn["tasks"][Y.TASK]["sample_digest"])
        out["gsm8k"]["ok"] = out["gsm8k"]["equal"] and out["gsm8k"]["digest_equal"]
        out["gsm8k"]["batch_mark"] = out["gsm8k"]["batch"] != out["gsm8k"]["batch_next"]      # ◊, as part Y
        log = ((CS.ROOT / model / "eval" / "gsm8k" if model in CS.QWEN3 else Y.ROOT / model) / f"{name}.log")
        text = log.read_text(errors="replace") if log.exists() else ""
        out["gsm8k"].update(log=log.name, date_pin="gsm8k6_date.py" in text,                    # §
                            expandable="ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments" in text)  # ¤
    nb, nn, pp = ppl_nll(model, name), ppl_nll(model, "fq16_1024"), ppl_path(model, name)
    if nb and nn:
        r = load(pp)
        if "results" in r:
            reported = {c: r["results"][c]["ppl"] for c, _ in T.CORPORA}
        else:
            ev = next(iter(r["evaluations"].values()))["evaluation"]
            reported = {c: ev[c]["ppl"] for c, _ in T.CORPORA}
        value = {c: math.exp(T.mean(nb[c])) for c, _ in T.CORPORA}
        rel = {c: abs(value[c] - reported[c]) / reported[c] for c in value}
        # main-ppl's records (run_ppl_deploy) store their PPL in float32; flipquant's in float64. The value here is
        # exp(mean) of the record's own per-window NLL, so it must agree to float32 precision
        out["ppl"] = dict(value=value, record_ppl=reported, relative_difference=rel,
                          equal=all(x <= 1e-6 for x in rel.values()),
                          windows={c: len(nb[c]) for c in nb}, windows_next={c: len(nn[c]) for c in nn})
        out["ppl"]["ok"] = out["ppl"]["equal"] and out["ppl"]["windows"] == out["ppl"]["windows_next"]
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


# ---------------------------------------------------------------------------------------------- cost

def cost_rows(model):
    """Calibration cost per method and size: the pilot's job records, part P's / the release's (FQ16-256) and part
    K's / P's FOCUS report (FOCUS-256)."""
    rows = []
    if model in Z.LLAMA:                     # part P's two Llama-Instruct models
        r = load(G.RUN / "llama_instruct" / model / "records" / "16x64" / "run.json")
    else:
        r = load(G.REL / model / "records" / "16x64" / "run.json")
    if r:
        t = r["trainer"]
        rows.append(dict(method="FlipQuant 16x64", windows=256, source="release" if model not in Z.LLAMA else "part P",
                         steps=5 * 256 // 8, e0m3_share=r["e0m3_share"], job_seconds=r.get("seconds"),
                         training_seconds=t.get("training_seconds"), setup_seconds=t.get("setup_seconds"),
                         gpu_peak_allocated_gib=(t.get("gpu_peak_allocated_gib") or [None])[0],
                         gpu_peak_reserved_gib=(t.get("gpu_peak_reserved_gib") or [None])[0],
                         device_peak_mib=r.get("device_peak_mib"), cpu_peak_rss_gib=t.get("cpu_peak_rss_gib"),
                         teacher_bytes=(r.get("teacher_storage") or {}).get("bytes")))
    for n in SIZES:
        j = load(CS.ROOT / model / f"fq16_{n}.job.json")
        if not j:
            continue
        t = j["trainer"]
        rows.append(dict(method="FlipQuant 16x64", windows=n, source="this part", steps=5 * n // 8,
                         e0m3_share=j["e0m3_share"], job_seconds=j["job_seconds"],
                         training_seconds=t.get("training_seconds"), setup_seconds=t.get("setup_seconds"),
                         gpu_peak_allocated_gib=(t.get("gpu_peak_allocated_gib") or [None])[0],
                         gpu_peak_reserved_gib=(t.get("gpu_peak_reserved_gib") or [None])[0],
                         device_peak_mib=j["device_peak_mib"], cpu_peak_rss_gib=t.get("cpu_peak_rss_gib"),
                         teacher_bytes=(j.get("teacher_storage") or {}).get("bytes"),
                         teacher_devices=(j.get("teacher_storage") or {}).get("devices")))
    f = load(G.RUN / "focus_qwen27b" / "report.json" if model == "qwen3.8-27b"      # part Q's cluster run (2 x H200)
             else Path(Z.focus_state(model)).parent / "report.json")
    if f:
        rows.append(dict(method="FOCUS", windows=256, source="part P" if model in Z.LLAMA else
                         "part Q (cluster)" if model == "qwen3.8-27b" else "part K",
                         steps=f.get("steps"), micro=f["args"].get("micro"), job_seconds=None,
                         training_seconds=f.get("training_seconds"), calibration_seconds=f.get("calibration_seconds"),
                         gpu_peak_allocated_gib=((f.get("resources") or {}).get("gpu_peak_gib") or [None])[0],
                         state_bytes=Path(Z.focus_state(model)).stat().st_size))
    g = load(Z.gptq_record(model))
    if g:                                    # the main table's GPTQ‡ job (part J; part P for the Llama-Instruct models)
        jobs = CS.queue_seconds("llama_gptq_" if model in Z.LLAMA else "gptq_")
        rows.append(dict(method="GPTQ‡", windows=256, source="part P" if model in Z.LLAMA else "part J",
                         job_seconds=jobs.get(f"llama_gptq_{model}" if model in Z.LLAMA else f"gptq_{model}_nvfp4-fo6"),
                         training_seconds=g["ptq"]["seconds"],
                         gpu_peak_allocated_gib=((g.get("resources") or {}).get("gpu_peak_gib") or [None])[0],
                         state_bytes=Path(Z.codes_file(model)).stat().st_size))
    for n in SIZES:                          # amendment 5
        j = load(CS.ROOT / model / f"gptq_{n}.job.json")
        if j:
            rows.append(dict(method="GPTQ‡", windows=n, source="this part (amendment 5)", job_seconds=j["job_seconds"],
                             training_seconds=j["gptq_seconds"], gpu_peak_allocated_gib=(j.get("gpu_peak_gib") or [None])[0],
                             device_peak_mib=j["device_peak_mib"], state_bytes=j["cache_bytes"]))
    for n, sub in [(n, f"focus_{n}") for n in SIZES] + [(256, "focus_256_local")]:
        j = load(CS.ROOT / model / sub / "state.job.json")
        if not j:
            continue
        rows.append(dict(method="FOCUS" + (" (local control)" if sub == "focus_256_local" else ""), windows=n,
                         source="this part", steps=j["steps"], micro=j["micro"],
                         job_seconds=j["job_seconds"], training_seconds=j["training_seconds"],
                         calibration_seconds=j["calibration_seconds"],
                         gpu_peak_allocated_gib=(j.get("gpu_peak_gib") or [None])[0], device_peak_mib=j["device_peak_mib"],
                         state_bytes=j["state_bytes"]))
    return rows


def eval_seconds(model):
    """The pilot rows' evaluation wall times (queue.log START / END)."""
    times = {}
    for line in (G.RUN / "queue.log").read_text().splitlines():
        p = line.split()
        if len(p) >= 4 and p[1] == "END" and p[2].startswith(("cs_zs_", "cs_gsm8k_", "cs_ppl_")) and model in p[2]:
            times[p[2]] = float(p[4].rstrip("s")) if len(p) > 4 and p[4].endswith("s") else None
    return times


# ---------------------------------------------------------------------------------------------- the documents

def f2(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


#: the coordinator's request (2026-10-20): FQ16 vs FOCUS pooled also without Qwen3.8-27B, whose FOCUS-1k stops early
#: on GSM8K (amendment 3); the 8-model figure stays, marked
WITHOUT_QWEN27B = ("FQ16-256 vs FOCUS-256", "FQ16-1k vs FOCUS-1k", "FQ16-2k vs FOCUS-2k")


def pooled(rec, exclude=()):
    """Per comparison, the discordant counts summed over the models that have both rows, with the exact McNemar p
    (part Y's pooled-vs-FourOverSix formula), for zero-shot (all questions of the six tasks) and GSM8K."""
    out = {}
    for label, _, _ in COMPARISONS:
        for ev in ("zeroshot", "gsm8k"):
            got = [(m, v["paired"][label][ev]) for m, v in rec["models"].items()
                   if m not in exclude and ev in v["paired"].get(label, {})]
            if not got:
                continue
            a, b, n = (sum(x[k] for _, x in got) for k in ("row_only", "ref_only", "n"))
            out.setdefault(label, {})[ev] = dict(models=[m for m, _ in got], row_only=a, ref_only=b, n=n,
                                                 diff=100 * (a - b) / n, p=RX.mcnemar(b, a))
    return out


def summary_md(rec):
    models = list(rec["models"])
    L = ["## Summary", "",
         "Per model: zero-shot Avg. (%), GSM8K strict-match (%) and WikiText-2 PPL of each method and calibration size "
         "(TBD: not run yet; —: not in the scope).", ""]
    for what, key, fmt in (("Zero-shot Avg. (%)", "zeroshot", lambda v: f2(v["avg"])),
                           ("GSM8K strict-match (%)", "gsm8k", lambda v: f2(v[0])),
                           ("WikiText-2 PPL", "ppl", lambda v: f2(v["wiki"], 3))):
        L += [f"**{what}:**", "", "| method | " + " | ".join(SHORT[m] for m in models) + " |",
              "|---|" + "---:|" * len(models)]
        for label, name in [("FourOverSix", "fo6"), ("GPTQ‡, 256", "gptq")] + GPTQ_ROWS + NEW_ROWS:
            cells = []
            for m in models:
                v = rec["models"][m]["values"].get(name, {}).get(key)
                cells.append(fmt(v) if v is not None else "TBD")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")
    P = rec.get("pooled") or {}
    if P:
        L += ["**Pooled over the models** (discordant counts summed; difference in points over all pooled questions; "
              "exact McNemar p; * = p < 0.05):", "",
              "| comparison | zero-shot | models | GSM8K | models |", "|---|---|---|---|---|"]
        P7 = rec.get("pooled_without_qwen27b") or {}
        for label, _, _ in COMPARISONS:
            for tag, src in (("", P), (" — without Qwen27B", P7)):
                c = src.get(label)
                if not c or (tag and label not in WITHOUT_QWEN27B):
                    continue
                mark = " — all models" if not tag and label in WITHOUT_QWEN27B and any(
                    "qwen3.8-27b" in (c.get(ev) or {}).get("models", []) for ev in ("zeroshot", "gsm8k")) else tag
                cells = []
                for ev in ("zeroshot", "gsm8k"):
                    x = c.get(ev)
                    cells += [fmt_pair(x) if x else "TBD", ", ".join(SHORT[m] for m in x["models"]) if x else ""]
                L.append(f"| {label}{mark} | " + " | ".join(cells) + " |")
        L += ["", "\"All models\" includes Qwen3.8-27B, whose FOCUS-1k stops early on GSM8K (amendment 3); \"without "
              "Qwen27B\" pools the other seven (the coordinator's request). Qwen3.8-27B's environment-matched comparisons "
              "against its local control are in its section.", ""]
    return L


def study_cost():
    """GPU job time of the part (queue.log START / END of its cs_* jobs), by kind."""
    kinds = {"cs_fq16_": "FlipQuant calibration", "cs_focus_": "FOCUS calibration",
             "cs_gptq_": "GPTQ‡ calibration with its PPL", "cs_zs_": "zero-shot", "cs_gsm8k_": "GSM8K", "cs_ppl_": "PPL",
             "cs_probe_": "probes", "cs_theta_": "theta rerun", "cs_offload_test": "offload test"}
    out = {}
    for name, sec in cs_jobs():
        k = next((v for pre, v in kinds.items() if name.startswith(pre)), None)
        if k:
            out[k] = out.get(k, 0) + sec
    return out


def cs_jobs():
    """(name, seconds) of the part's finished jobs (queue.log END lines of cs_*)."""
    out = []
    for line in (G.RUN / "queue.log").read_text().splitlines():
        q = line.split()
        if len(q) >= 5 and q[1] == "END" and q[2].startswith("cs_") and q[4].endswith("s") and q[4][:-1].isdigit():
            out.append((q[2], int(q[4][:-1])))
    return out


def gptq_cost():
    """Amendment 5's GPU time: its calibration jobs (with their PPL) and its zero-shot / GSM8K jobs, step 0 included."""
    out = {}
    for name, sec in cs_jobs():
        if name.startswith("cs_gptq_") or name.endswith(("_gptq256", "_gptq_1024", "_gptq_2048")):
            k = "calibration with PPL" if name.startswith("cs_gptq_") else "zero-shot" if name.startswith("cs_zs_") \
                else "GSM8K"
            out[k] = out.get(k, 0) + sec
    return out


def findings_md(rec):
    P, P7 = rec.get("pooled") or {}, rec.get("pooled_without_qwen27b") or {}

    def g(label, ev, src=None):
        x = (src if src is not None else P).get(label, {}).get(ev)
        return "TBD" if not x else f"{x['diff']:+.2f}{'*' if x['p'] < 0.05 else ''} (p={x['p']:.2g})"
    cost, gc = study_cost(), gptq_cost()
    L = ["## Findings", "",
         "Pooled exact McNemar over the eight models (discordant counts summed; * = p < 0.05). The per-model tables "
         "below have every pair.", "",
         f"- **FlipQuant 16x64 gains GSM8K from more windows, and the gain saturates at 1,024.** vs 256: GSM8K "
         f"{g('FQ16-1k vs FQ16-256', 'gsm8k')} at 1,024 and {g('FQ16-2k vs FQ16-256', 'gsm8k')} at 2,048; 2,048 vs 1,024 "
         f"{g('FQ16-2k vs FQ16-1k', 'gsm8k')}. Zero-shot: {g('FQ16-1k vs FQ16-256', 'zeroshot')} / "
         f"{g('FQ16-2k vs FQ16-256', 'zeroshot')}.",
         f"- **FOCUS gains too.** vs 256 (seven models; Qwen3.8-27B's 256 reference is in another environment): GSM8K "
         f"{g('FOCUS-1k vs FOCUS-256', 'gsm8k')} at 1,024 and {g('FOCUS-2k vs FOCUS-256', 'gsm8k')} at 2,048; zero-shot "
         f"{g('FOCUS-1k vs FOCUS-256', 'zeroshot')} / {g('FOCUS-2k vs FOCUS-256', 'zeroshot')}.",
         f"- **FlipQuant vs FOCUS at the same size.**",
         f"  - GSM8K: {g('FQ16-256 vs FOCUS-256', 'gsm8k')} at 256, {g('FQ16-1k vs FOCUS-1k', 'gsm8k')} at 1,024 "
         f"({g('FQ16-1k vs FOCUS-1k', 'gsm8k', P7)} without Qwen3.8-27B's early-stopping FOCUS-1k), "
         f"{g('FQ16-2k vs FOCUS-2k', 'gsm8k')} at 2,048.",
         f"  - Zero-shot: {g('FQ16-256 vs FOCUS-256', 'zeroshot')} / {g('FQ16-1k vs FOCUS-1k', 'zeroshot')} / "
         f"{g('FQ16-2k vs FOCUS-2k', 'zeroshot')}.",
         f"  - At 2,048 the two are level on both.",
         f"- **vs IF4 / MixFP4 (Zou), simulated:** zero-shot {g('FQ16-2k vs IF4†', 'zeroshot')} / "
         f"{g('FQ16-2k vs MixFP4 (Zou)†', 'zeroshot')} at 2,048 ({g('FQ16-256 vs IF4†', 'zeroshot')} / "
         f"{g('FQ16-256 vs MixFP4 (Zou)†', 'zeroshot')} at 256); GSM8K {g('FQ16-2k vs IF4†', 'gsm8k')} / "
         f"{g('FQ16-2k vs MixFP4 (Zou)†', 'gsm8k')} (six models).",
         f"- **GPTQ‡ on the same windows (amendment 5).** vs its 256: GSM8K {g('GPTQ‡-1k vs GPTQ‡-256', 'gsm8k')} at "
         f"1,024 and {g('GPTQ‡-2k vs GPTQ‡-256', 'gsm8k')} at 2,048 (2,048 vs 1,024: {g('GPTQ‡-2k vs GPTQ‡-1k', 'gsm8k')}); "
         f"zero-shot {g('GPTQ‡-1k vs GPTQ‡-256', 'zeroshot')} / {g('GPTQ‡-2k vs GPTQ‡-256', 'zeroshot')}.",
         f"- **FlipQuant vs GPTQ‡ at the same size.** GSM8K {g('FQ16-256 vs GPTQ‡-256', 'gsm8k')} at 256, "
         f"{g('FQ16-1k vs GPTQ‡-1k', 'gsm8k')} at 1,024, {g('FQ16-2k vs GPTQ‡-2k', 'gsm8k')} at 2,048; zero-shot "
         f"{g('FQ16-256 vs GPTQ‡-256', 'zeroshot')} / {g('FQ16-1k vs GPTQ‡-1k', 'zeroshot')} / "
         f"{g('FQ16-2k vs GPTQ‡-2k', 'zeroshot')}.",
         "- **The motivating losses.** Each model's own table has the pairs.",
         "  - Phi-4 GSM8K vs MixFP4: −1.29* at 256 is gone at 1,024 / 2,048.",
         "  - Qwen3.8-27B zero-shot vs IF4: −0.45* at 256 is not significant at 1,024 / 2,048.",
         "  - Nemotron's zero-shot gap to FOCUS closes at 2,048.",
         "  - Phi-4's zero-shot gap stays against FOCUS-256 (−0.42* at 2,048). Against FOCUS-2k it is −0.11 (n.s.), "
         "but only because FOCUS-2k itself fell (−0.30* vs FOCUS-256).",
         "- **Caveats.**",
         "  - Each FOCUS cell is one run, and FOCUS is not reproducible run to run (step 0 (d); amendment 3).",
         "  - Qwen3.8-27B's FOCUS-1k stops early on GSM8K; its 256 control and its 2,048 run do not, so this is a "
         "run-specific outcome, not a window-count trend.",
         "  - FlipQuant is deterministic, but a different window draw was not tried.",
         "  - Qwen3-1.7B trades WikiText-2 / C4 NLL for GSM8K with more math / code windows, FlipQuant more than FOCUS.",
         "  - Nemotron's FQ16-2k strict-match loss is format only (flexible-extract unchanged).",
         "  - E0M3 shares grow with the steps (2,048: 6.8–17.2 %, Llama-1B 27.0 %; amendment 4).",
         "- **GPU time of the part:** " + ", ".join(f"{k} {v / 3600:.1f} h" for k, v in cost.items())
         + f"; {sum(cost.values()) / 3600:.1f} h in all."
         + (" Amendment 5 (GPTQ‡, step 0 included): " + ", ".join(f"{k} {v / 3600:.1f} h" for k, v in gc.items())
            + f"; {sum(gc.values()) / 3600:.1f} h." if gc else ""), ""]
    return L


def report_md(rec):
    L = ["# Calibration size: FlipQuant 16x64 and FOCUS at 1,024 and 2,048 windows (part AA)", "",
         "PROTOCOL.md part AA and its amendment 1. FlipQuant 16x64 (release settings: 5 epochs, top-1000 teacher, seed 0, "
         "last-step map) and FOCUS (part K's FOCUS_COMMON: 1 epoch, global batch 32, micro 8) trained on 1,024 or 2,048 "
         "windows of 512 tokens instead of the release's 256. The sets are nested, 256 ⊂ 1,024 ⊂ 2,048, and audited "
         "against every evaluation text (WINDOWS.md). FlipQuant and FOCUS read the same windows (token sha256 checked). "
         "flipquant calib-size f912a30. Evaluation exactly as parts X (zero-shot) and Y (GSM8K), plus PPL (paper "
         "convention). Step 0: STEP0.md. 2,048-window memory probes: PROBES.md.",
         "",
         "Amendment 5 adds GPTQ‡, the main table's GPTQ variant, recalibrated on the same 1,024 / 2,048 windows. As in "
         "the main table, its codes are computed inside the row's PPL job; zero-shot and GSM8K read them from the codes "
         "file. Its step 0 (STEP0.md) reproduces Llama-3.2-1B-Instruct's 256-window codes and rows bit for bit.", ""]
    L += findings_md(rec)
    L += summary_md(rec)
    for model in rec["models"]:
        m = rec["models"][model]
        L += [f"## {TITLES[model]}", "",
              "Zero-shot accuracy (%) per task and Avg.; GSM8K strict-match (%) ± 2 SE; PPL. 256-window rows: the part X / "
              "Y cells (FOCUS-256 and FQ16-256 are the existing state and map); PPL: the calib-size runs of step 0 "
              "(c2) for FourOverSix / FQ16-256 / FOCUS-256, the final table's records for the other baselines (BF16 "
              "included).",
              "The BF16, GPTQ‡, IF4† and MixFP4† rows' values are checked against their source records' own metrics, "
              "and against the protocol of the rows beside them (zero-shot digests and batch, GSM8K digest, PPL "
              "windows); a failed check would show \"mismatch\" (calib_size.json baseline_check). † simulated with "
              "each method's own rules at 1x16, calibration-free; ‡ GPTQ with BF16 propagation on the release 256 "
              "windows (the main-table variant). IF4 / MixFP4 GSM8K was not run for the Qwen3 models.",
              "GPTQ‡ 1,024 / 2,048 (amendment 5): the same GPTQ on this part's windows; its PPL is its calibration job's, "
              "as GPTQ‡ (256)'s is the main table's.", "",
              "| method | " + " | ".join(n for _, n in RX.TASK_NAMES) + " | Avg. | GSM8K | Wiki PPL | C4 PPL |",
              "|---|" + "---:|" * (len(RX.TASK_NAMES) + 4)]
        bc = m.get("bf16_check") or {}
        for label, name in m["rows"]:
            r = m["values"].get(name, {})
            zs = r.get("zeroshot")
            cells = [f2(zs[t]) for t, _ in RX.TASK_NAMES] + [f2(zs["avg"])] if zs else ["TBD"] * (len(RX.TASK_NAMES) + 1)
            g = r.get("gsm8k")
            cells.append(f"{g[0]:.2f} ± {g[1]:.2f}" if g else "TBD")
            p = r.get("ppl")
            cells += [f2(p["wiki"], 3), f2(p["c4"], 3)] if p else ["TBD", "TBD"]
            ck = (m.get("baseline_check") or {}).get(name)
            if ck is not None:
                gi = len(RX.TASK_NAMES) + 1
                if not (ck.get("zeroshot") or {}).get("ok", True):
                    cells[:gi] = ["mismatch"] * gi
                gk = ck.get("gsm8k") or {}
                if gk.get("not_run"):
                    cells[gi] = "not run"
                elif not gk.get("ok", True):
                    cells[gi] = "mismatch"
                else:
                    cells[gi] += "".join(mk for mk, on in ((" ◊", gk.get("batch_mark")), (" §", gk.get("date_pin")),
                                                           (" ¤", gk.get("expandable"))) if on)
                if not (ck.get("ppl") or {}).get("ok", True):
                    cells[-2:] = ["mismatch", "mismatch"]
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        marks = {k for ck in (m.get("baseline_check") or {}).values() for k in ("batch_mark", "date_pin", "expandable")
                 if (ck.get("gsm8k") or {}).get(k)}
        if "batch_mark" in marks:
            L += ["", f"◊ BF16 GSM8K ran at batch {bc['gsm8k']['batch']} (part Y), every other row at "
                  f"{bc['gsm8k']['batch_next']}. Generations depend on the batch composition, so its paired comparisons "
                  "would compare different batches; it enters none here."]
        if "date_pin" in marks:
            L += ["", "§ GSM8K run with the chat template's date pinned to 15 Oct 2026 (part Y amendment 3), the date "
                  "the native rows saw; this part's rows of this model are pinned too. All prompts are equal (equal "
                  "sample digests)."]
        if "expandable" in marks:
            L += ["", "¤ GSM8K run with expandable segments (part Y amendment 4; an allocator setting, which its "
                  "pre-check showed changes no generation)."]
        L += ["", "**Paired.**",
              "- Zero-shot: pooled over all questions of the six tasks.",
              "- GSM8K: strict-match.",
              "- Format: difference in points (row-only / reference-only correct, exact McNemar p).",
              "- ΔNLL: mean per-window NLL difference ± 2 SE (negative = lower NLL).",
              "- \\* = p < 0.05, or |Δ| > 2 SE.", "",
              "| comparison | zero-shot | GSM8K | ΔNLL Wiki | ΔNLL C4 |", "|---|---|---|---|---|"]
        for label, c in m["paired"].items():
            nll = c.get("nll") or {}
            L.append(f"| {label} | {fmt_pair(c.get('zeroshot'))} | {fmt_pair(c.get('gsm8k'))} | "
                     f"{fmt_nll(nll.get('wiki'))} | {fmt_nll(nll.get('c4'))} |")
        st = m.get("gsm8k_stops") or {}
        if st:
            L += ["", "GSM8K generation endings (amendment 3): strict-match no-answer count, generations cut at \"The "
                  "final [answer [is]]\", mean / max generated tokens and budget (1,024) hits, of 1,319:", "",
                  "| method | strict no-answer | cut at \"The final…\" | mean tokens | max | budget hits |",
                  "|---|---:|---:|---:|---:|---:|"]
            for name, v in st.items():
                L.append(f"| {v['label']} | {v['strict_no_answer']} | {v['cut_at_final']} | {v['mean']:.1f} | {v['max']} | "
                         f"{v['budget_hits']} |")
        main = [k for k in ("FQ16-1k vs FQ16-256", "FOCUS-1k vs FOCUS-256", "FQ16-1k vs FOCUS-1k",
                            "FQ16-2k vs FQ16-256", "FOCUS-2k vs FOCUS-256", "FQ16-2k vs FOCUS-2k",
                            "FQ16-256 vs FOCUS-256", "GPTQ‡-1k vs GPTQ‡-256", "GPTQ‡-2k vs GPTQ‡-256",
                            "FQ16-1k vs GPTQ‡-1k", "FQ16-2k vs GPTQ‡-2k") if "zeroshot_tasks" in m["paired"].get(k, {})]
        if main:
            L += ["", "Zero-shot per task (difference in points, counts, exact p):", "",
                  "| comparison | " + " | ".join(n for _, n in RX.TASK_NAMES) + " |",
                  "|---|" + "---|" * len(RX.TASK_NAMES)]
            for k in main:
                tp = m["paired"][k]["zeroshot_tasks"]
                L.append(f"| {k} | " + " | ".join(fmt_pair(tp[t]) for t, _ in RX.TASK_NAMES) + " |")
        cv = m.get("curves") or {}
        if cv:
            L += ["", "**FlipQuant 16x64 training per epoch** (amendment 4; the trainer's report.json): training KL, E0M3 "
                  "tiles at the epoch's end, and tile flips within the epoch (a tile crossing theta = 0 counts each time).",
                  ""]
            L += ["| windows | tiles | " + " | ".join(f"epoch {i + 1}" for i in range(5)) + " | final E0M3 share |",
                  "|---:|---:|" + "---|" * 5 + "---:|"]
            for n, c in cv.items():
                cells = [f"KL {e['train_kl']:.4f}; E0M3 {e['e0m3']:,}; flips {e['flips']:,}" for e in c["epochs"]]
                share = c["final_e0m3"] / c["tiles"] if c.get("tiles") else None
                L.append(f"| {n:,} | {format(c['tiles'], ',') if c.get('tiles') else '—'} | " + " | ".join(cells)
                         + f" | {f2(share * 100 if share is not None else None)} % |")
            for n, c in cv.items():
                th = c.get("theta")
                if th:
                    L += ["", f"**Theta at the end of the {n:,}-window run** (a rerun with `--save-theta` whose map equals "
                          f"the evaluated one: {'yes' if th['reproduces'] else 'NO'}):",
                          f"- E0M3 tiles with theta < 0.02 (within one lr step of the threshold): "
                          f"{th['e0m3_within_one_step']:.1%} of {th['e0m3_tiles']:,};",
                          f"- E2M1 tiles with theta in (-0.02, 0]: {th['e2m1_within_one_step']:.1%}; all tiles with "
                          f"|theta| < 0.02: {th['all_within_one_step']:.1%} of {th['tiles']:,};",
                          f"- E0M3 theta quantiles (10/25/50/75/90 %): "
                          + ", ".join(f"{q:.3f}" for q in th["quantiles_e0m3"]) + "."]
        L += ["", "**Calibration cost and E0M3 share.** GPTQ‡: the job is GPTQ plus the row's PPL evaluation (the main "
              "table's route), \"training\" is GPTQ's own time (ptq.seconds), and the bytes are the codes file's.", "",
              "| method | windows | source | steps | E0M3 share | job (s) | training (s) | peak allocated (GiB) | "
              "device peak (MiB) | host RSS (GiB) | teacher / state / codes (bytes) |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in m["cost"]:
            L.append(f"| {c['method']} | {c['windows']:,} | {c['source']} | {c.get('steps') or '—'} | "
                     f"{f2(c.get('e0m3_share'), 4) if c.get('e0m3_share') is not None else '—'} | "
                     f"{f2(c.get('job_seconds'), 0)} | {f2(c.get('training_seconds'), 0)} | "
                     f"{f2(c.get('gpu_peak_allocated_gib'))} | {c.get('device_peak_mib') or '—'} | "
                     f"{f2(c.get('cpu_peak_rss_gib'))} | "
                     f"{format(c.get('teacher_bytes') or c.get('state_bytes') or 0, ',') or '—'} |")
        if m.get("eval_seconds"):
            L += ["", "Evaluation wall times (s): " + ", ".join(f"{k.replace('cs_', '').replace('_' + model, '')} "
                                                            f"{v:.0f}" for k, v in sorted(m["eval_seconds"].items())
                                                            if v is not None) + "."]
        L.append("")
    return L


def build():
    rec = dict(models={})
    for model in CS.ORDER:
        if model not in CS.PILOT and not (CS.ROOT / model / "fq16_1024.job.json").exists():
            continue
        rows = table_rows()
        if load(CS.ROOT / model / "eval" / "zs" / "focus256_rerun.json") is not None:
            rows = rows + [RERUN]
        if model == "qwen3.8-27b":
            rows = rows + [CONTROL]
        values = {}
        for label, name in rows:
            v = {}
            z = zs_record(model, name)
            if z is not None and z.get("status") == "complete":
                v["zeroshot"] = dict({t: RX.acc(z, t) for t, _ in RX.TASK_NAMES}, avg=zs_avg(z))
            g = gsm_record(model, name)
            if g is not None:
                v["gsm8k"] = RY.acc(g)
                v["gsm8k_flex"] = RY.acc(g, "flexible_extract")
            n = ppl_nll(model, name)
            if n is not None:
                v["ppl"] = {c: math.exp(T.mean(n[c])) for c, _ in T.CORPORA}
                v["nll_mean"] = {c: T.mean(n[c]) for c, _ in T.CORPORA}
            values[name] = v
        comps = list(COMPARISONS)
        if CONTROL in rows:
            comps = [((f"{label} (cluster: a different environment)" if b == "focus256" and a.startswith("focus_")
                       else label), a, b) for label, a, b in comps] + CONTROL_COMPARISONS
        if RERUN in rows:
            comps.append(("FOCUS-256 retrained vs FOCUS-256", "focus256_rerun", "focus256"))
            comps.append(("FOCUS-1k vs FOCUS-256 retrained", "focus_1024", "focus256_rerun"))
        paired = {label: compare(model, a, b) for label, a, b in comps}
        rec["models"][model] = dict(rows=rows, values=values, paired=paired, cost=cost_rows(model),
                                    eval_seconds=eval_seconds(model), gsm8k_stops=gsm8k_stops(model, rows),
                                    curves=training_curves(model), bf16_check=bf16_check(model),
                                    baseline_check={b: bf16_check(model, b) for b in BASELINES})
    rec["pooled"] = pooled(rec)
    rec["gpu_time_seconds"] = study_cost()
    rec["pooled_without_qwen27b"] = {k: v for k, v in pooled(rec, exclude=("qwen3.8-27b",)).items()
                                     if k in WITHOUT_QWEN27B}
    return rec


def windows_md():
    nest = load(CS.WIN / "nesting.json") or {}
    audit = load(CS.WIN / "audit.json")
    L = ["# Part AA: the calibration windows", "",
         "The release rule, extended (PROTOCOL.md part AA): the calibration record's 128 windows, then N - 128 by the "
         "vendored extension rule (seed 20260930; OpenWebMath train-00000 @ fde8ef8 and CodeParrot-clean "
         "file-000000000001 @ 35a59fb; (N - 128) / 2 per source; skipping the record's fit documents, the three "
         "development records and the published C4 evaluation documents). Drawn on the CPU by calibration.fit_windows "
         "(calib-size f912a30); every window recorded by document sha256, offset and token sha256.", "",
         "| model | N | extension windows | skipped (fit / dev / C4) | contains | nesting checks | token-list sha256 | "
         "drawn in (s) |", "|---|---:|---:|---|---:|---|---|---:|"]
    for model, r in nest.items():
        for n in CS.SIZES[model]:
            e = r.get(str(n))
            w = load(CS.windows_file(model, n))
            if not e or not w:
                continue
            sk = e["skipped"]
            L.append(f"| {TITLES[model]} | {n:,} | {w['fit_extension']['windows']:,} | {sk['fit']} / {sk['development']} "
                     f"/ {sk['c4_evaluation']} | {e['contains']:,} | "
                     f"{'all pass' if e['ok'] else 'FAIL ' + str(e['checks'])} | `{e['token_list_sha256'][:16]}…` | "
                     f"{w['seconds']:.0f} |")
    L += ["", "Nesting: each source's first (n - 128) / 2 extension windows equal the smaller set's (the release 16x64 "
          "trainer report for 256), record by record; the skip counts equal the release run's; every token hash is "
          "distinct. The full token-list sha256 values are in calib_size.json.", ""]
    if audit:
        L += ["## 13-gram audit", "",
              f"Word {audit['n']}-grams ({audit['normalization']}) of every window of each model's 2,048 set, decoded "
              "with its tokenizer, against the evaluation texts (supersets of what is scored):", "",
              "| evaluation set | items | words | 13-grams |", "|---|---:|---:|---:|"]
        for s, st in audit["eval_sets"].items():
            L.append(f"| {s} | {st['items']:,} | {st['words']:,} | {st['grams']:,} |")
        L += ["", "Windows with a shared 13-gram, by evaluation set and membership (release = the 256 set; 1024 / 2048 = "
              "the windows each larger set adds):", "",
              "| model | windows | hits |", "|---|---:|---|"]
        for model, r in audit["models"].items():
            hits = "; ".join(f"{s}: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items()))
                             for s, d in sorted(r["windows_with_hits"].items())) or "none"
            L.append(f"| {TITLES[model]} | {r['windows']:,} | {hits} |")
        L += ["", f"**Pilot gate (no hit among the 1,024 sets' new windows of the pilot models): "
                  f"{'passed' if audit['pilot_ok'] else 'FAILED'}.**", ""]
        ctl = load(CS.WIN / "audit_control.json")
        if ctl:
            L += ["**Positive control** (calib_size_audit_control.py). For each model, 30 windows were chosen with a fixed "
                  "seed and re-scanned as they are, then with one evaluation item spliced into the middle of the decoded "
                  "text: " + ", ".join(f"{i['eval_set']} item {i['item']} ({i['words']} words)" for i in ctl["items"])
                  + ". A splice counts as detected when the audit attributes it to its own set and item.", "",
                  "| model | windows | 13-grams scanned | plain windows with a hit | splices detected (of 30 each) |",
                  "|---|---:|---:|---:|---|"]
            for m, v in ctl["models"].items():
                L.append(f"| {TITLES[m]} | {v['windows']:,} | {v['scanned_13grams']:,} | {v['plain_windows_with_hits']} | "
                         + ", ".join(f"{k} {d}" for k, d in v["spliced_detected"].items()) + " |")
            L += ["", f"Control: **{'pass' if ctl['ok'] else 'FAIL'}**. So the audit's result is not an artifact of the "
                      "scan: it detects evaluation text that is there.", ""]
        listed = [(m, h) for m, r in audit["models"].items() for h in r["hits"]]
        if listed:
            L += ["Every hit (window index in the trainer's order; the first shared 13-gram):", "",
                  "| model | window | membership | source | document | evaluation set | item | shared 13-grams | first "
                  "13-gram |", "|---|---:|---|---|---|---|---|---:|---|"]
            for m, h in listed:
                L.append(f"| {TITLES[m]} | {h['window']} | {h['membership']} | {h['source']} | "
                         f"{(h['document_sha256'] or 'record')[:12]} | {h['eval_set']} | {h['item']} | "
                         f"{h['shared_grams']} | {h['first_gram']} |")
    return L


def focus_diff():
    """Step 0 (d): the retrained Llama-1B FOCUS-256 state against part P's, per parameter kind."""
    import torch
    a = torch.load(CS.ROOT / "step0" / "focus_256" / "state" / "focus.pt", map_location="cpu", weights_only=False)["layers"]
    b = torch.load(Z.focus_state(CS.LLAMA1B), map_location="cpu", weights_only=False)["layers"]
    out = {}
    for k, init in (("m", 1.0), ("q", 6.0)):
        d = torch.cat([(a[n][k].float() - b[n][k].float()).abs().flatten() for n in b])
        drift = torch.cat([(b[n][k].float() - init).abs().flatten() for n in b])
        out[k] = dict(elements=d.numel(), differ=int((d > 0).sum()), max=float(d.max()), mean=float(d.mean()),
                      drift=float(drift.mean()))
    return out


def step0_md():
    s = load(CS.ROOT / "step0" / "step0.json")
    if not s:
        return ["# Part AA step 0", "", "Not run yet."]
    L = ["# Part AA step 0", "", "PROTOCOL.md part AA, step 0, on flipquant calib-size f912a30.", ""]
    a = s.get("a_code")
    if a:
        L += [f"**(a) Code.** `git diff caf69be {a['head'][:7]}` touches {', '.join(a['files'])}; evaluation/, "
              f"flipquant/, models/ and kernels/ are byte-identical: **{'pass' if a['ok'] else 'FAIL'}**.", ""]
    b = s.get("b_calibration")
    if b:
        L += ["**(b) Calibration identity.** Llama-3.2-1B-Instruct FlipQuant 16x64 at the release settings (256 windows, "
              f"part P's command) on calib-size; part P's trainer map sha256 `{b['part_p_run_map_sha256'][:16]}…`.", "",
              "| teacher | run_map_sha256 | equal | tiles equal | teacher device | peak allocated (GiB) | device peak "
              "(MiB) | job (s) |", "|---|---|---|---|---|---:|---:|---:|"]
        for k in ("gpu", "host"):
            v = b[k]
            L.append(f"| {k} | `{v['run_map_sha256'][:16]}…` | {v['equal_sha']} | {v['tiles_equal']} | "
                     f"{', '.join(v['teacher_devices'] or [])} | {f2((v['gpu_peak_allocated_gib'] or [None])[0])} | "
                     f"{v['device_peak_mib']} | {f2(v['seconds'], 0)} |")
        L += ["", f"Result: **{'pass' if b['ok'] else 'FAIL'}**.", ""]
    c = s.get("c_identity")
    if c:
        zs, g, p = c["zeroshot"], c["gsm8k"], c["ppl"]
        L += ["**(c) Evaluation identity** (Llama-3.2-1B-Instruct FlipQuant 16x64, part P's map, on calib-size):",
              "- zero-shot vs part X's record, per task (sample_digest, per-example correctness, per-choice "
              "log-likelihoods, metrics): " + ", ".join(f"{t} {'equal' if all(v.values()) else 'DIFFERENT ' + str(v)}"
                                                         for t, v in zs["tasks"].items()) + ";",
              f"- GSM8K (date pinned to 15 Oct 2026) vs part Y's samples: {g['identical']:,} / {g['samples']:,} "
              f"generations and extracted answers byte-identical;",
              f"- PPL vs part P's record: per-window NLL equal on " + ", ".join(
                  f"{k} ({p['windows'][k]} windows): {v}" for k, v in p["equal"].items()) + ".",
              f"Result: **{'pass' if c['ok'] else 'FAIL'}**.", ""]
    c2 = s.get("c2_ppl_references")
    if c2:
        L += ["**(c2) PPL references** on calib-size, against the records the final table uses (not a stop "
              "condition):", "", "| model | row | record | Wiki equal | C4 equal | max abs per-window ΔNLL |",
              "|---|---|---|---|---|---|"]
        for model, rows in c2.items():
            for name, v in rows.items():
                i = v["identity"]
                L.append(f"| {TITLES[model]} | {name} | `{v['record'].replace('/home/dev/', '~/')}` | {i['equal']['wiki']} | "
                         f"{i['equal']['c4']} | {', '.join(f'{k} {x:.2e}' for k, x in i['max_abs_diff'].items() if x is not None)} |")
        L.append("")
    d = s.get("d_focus")
    if d:
        L += [f"**(d) FOCUS reproducibility.** Llama-3.2-1B-Instruct FOCUS at 256 windows retrained with part P's command: "
              f"{d['modules']} modules; {d['differing_tensors']} tensors differ from part P's state "
              f"({'bit-identical' if d['identical'] else 'not bit-identical: evaluated as the run-to-run reference'}). "
              f"State sha256 `{d['state_sha256'][:16]}…` vs part P's `{d['part_p_state_sha256'][:16]}…`.", ""]
        if not d["identical"]:
            fd = focus_diff()
            L += ["How far apart the two states are, against how far training moved part P's state from its "
                  "initialization (m = 1, q = init_q = 6):", "",
                  "| parameter | elements | differing | max abs difference | mean abs difference | mean abs trained - init |",
                  "|---|---:|---:|---:|---:|---:|"]
            for k, v in fd.items():
                L.append(f"| {k} | {v['elements']:,} | {v['differ']:,} | {v['max']:.3g} | {v['mean']:.3g} | "
                         f"{v['drift']:.3g} |")
            L += ["", "The difference is about half the training drift, so it is not rounding noise. A plausible mechanism, "
                  "not tested here: train_focus's gradients are nondeterministic, and AdamW's first steps "
                  "(about lr x sign(g)) amplify sign changes of near-zero gradients. The run-to-run row in REPORT.md "
                  "shows the evaluation-level effect.", ""]
    L += [f"**Step 0: {'passed' if s.get('ok') else 'not passed' if 'ok' in s else 'in progress'}.**"]
    return L + gptq_step0_md()


def gptq_step0_md():
    """Amendment 5's step 0 (RUN/calib_size/step0/gptq_step0.json)."""
    s = load(CS.ROOT / "step0" / "gptq_step0.json")
    if not s:
        return []
    L = ["", "## Amendment 5, step 0: GPTQ‡", "", f"On flipquant calib-size {CS.GPTQ_CODE}.", ""]
    a = s.get("args")
    if a:
        models = [m for m in a if m != "ok"]
        L += ["**Arguments.** For each model, with the main table's windows and codes files: the PPL job's command equals "
              "the main table's logged command (parts J / P; part P ran without a cache file), and the zero-shot / GSM8K "
              "arguments equal parts X / Y's GPTQ‡ rows': "
              + ", ".join(f"{TITLES[m]} {'pass' if a[m]['ok'] else 'FAIL'}" for m in models)
              + f". **{'pass' if a['ok'] else 'FAIL'}**.", ""]
    c = s.get("calibration")
    if c:
        L += [f"**Codes.** Llama-3.2-1B-Instruct GPTQ‡ at 256 windows with part P's command: codes digest "
              f"`{c['codes_sha256'][:16]}…`, part P's `{c['part_p_codes_sha256'][:16]}…`: "
              f"**{'equal' if c['codes_equal'] else 'DIFFERENT'}** (required). The codes file's sha256 "
              f"`{c['file_sha256'][:16]}…` vs the HF upload's `{c['uploaded_file_sha256'][:16]}…`: "
              f"{'equal' if c['file_equal_uploaded'] else 'different'} (reported). GPTQ {f2(c['gptq_seconds'], 0)} s, "
              f"job {f2(c['job_seconds'], 0)} s, device peak {c['device_peak_mib']} MiB.", ""]
    i = s.get("identity")
    if i:
        zs, g, p = i["zeroshot"], i["gsm8k"], i["ppl"]
        L += ["**Evaluations** on calib-size, reading that codes file:",
              "- zero-shot vs part X's GPTQ‡ record, per task (sample_digest, per-example correctness, per-choice "
              "log-likelihoods, metrics): " + ", ".join(f"{t} {'equal' if all(v.values()) else 'DIFFERENT ' + str(v)}"
                                                         for t, v in zs["tasks"].items()) + ";",
              f"- GSM8K (date pinned to 15 Oct 2026) vs part Y's samples: {g['identical']:,} / {g['samples']:,} "
              "generations and extracted answers byte-identical;",
              "- PPL vs part P's record: per-window NLL equal on " + ", ".join(
                  f"{k} ({p['windows'][k]} windows): {v}" for k, v in p["equal"].items()) + ".", ""]
    L += [f"**Amendment 5's step 0: {'passed' if s.get('ok') else 'not passed' if 'ok' in s else 'in progress'}.**"]
    return L


def cache_files():
    """flipquant.data's cached calibration sets (calibration_release_<key>.pt, one per tokenizer and window set) and
    their sizes; the cache root is flipquant.data.cache_root() of the evaluation env (no FLIPQUANT_CACHE set)."""
    root = Path.home() / ".cache" / "flipquant" / "data"
    return {str(f.relative_to(root)): f.stat().st_size for f in sorted(root.glob("*/calibration_release_*.pt"))}


def probes_md():
    p = load(CS.ROOT / "probes" / "probes.json")
    if not p:
        return ["# Part AA: 2,048-window memory probes", "", "Not run yet."]
    L = ["# Part AA: 2,048-window memory probes", "",
         "The two worst cases of the 256-window runs, at 2,048 windows (PROTOCOL.md part AA). The device peak is "
         "nvidia-smi's memory.used every 2 s (part P's sampler); the card has 97,887 MiB (95.6 GiB).", ""]
    q = p.get("qwen3.8-27b")
    if q:
        pr = q.get("probe") or {}
        res = pr.get("resources") or {}
        ts = pr.get("teacher_storage") or {}
        L += ["## FlipQuant Qwen3.8-27B 16x64, teacher on the host",
              "", f"`{q['command']}`", "",
              f"- rc {q['rc']}, status {pr.get('status')}; job {f2(q.get('job_seconds'), 0)} s; device peak "
              f"{q.get('device_peak_mib')} MiB.",
              f"- Peak allocated / reserved: {f2((res.get('gpu_peak_allocated_gib') or [None])[0])} / "
              f"{f2((res.get('gpu_peak_reserved_gib') or [None])[0])} GiB (the 256-window run with the teacher on the "
              "GPU: 91.25 / 92.79 GiB). Host RSS peak: " + f"{f2(res.get('cpu_peak_rss_gib'))} GiB.",
              f"- Teacher: {ts.get('windows')} windows, {format(ts.get('bytes') or 0, ',')} bytes on "
              f"{', '.join(ts.get('devices') or [])}.",
              f"- Per phase (GiB allocated): " + ", ".join(
                  f"{k} {f2(v.get('gpu_peak_allocated_gib') and v['gpu_peak_allocated_gib'][0])}"
                  for k, v in ((res.get('phases') or {}).get('by_phase') or {}).items()) + ".",
              f"- Probe: {(pr.get('probe') or {}).get('steps')} optimizer steps, "
              f"{f2((pr.get('probe') or {}).get('step_seconds'), 1)} s per step.", ""]
        if pr.get("out_of_memory"):
            L += [f"- **Out of memory:** {pr['out_of_memory'].get('message')}", ""]
    f = p.get("phi4-14b")
    if f:
        r = f.get("report") or {}
        pb = r.get("probe") or {}
        L += ["## FOCUS Phi-4, micro-batch 8", "", f"`{f['command']}`", "",
              f"- rc {f['rc']}, status {r.get('status')}; job {f2(f.get('job_seconds'), 0)} s; device peak "
              f"{f.get('device_peak_mib')} MiB.",
              f"- Peak allocated / reserved: {f2((pb.get('gpu_peak_allocated_gib') or [None])[0])} / "
              f"{f2((pb.get('gpu_peak_reserved_gib') or [None])[0])} GiB (the 256-window run: 91.04 GiB allocated).",
              f"- {pb.get('steps')} optimizer steps; step times (s, cumulative): "
              f"{', '.join(f2(x, 0) for x in pb.get('step_seconds') or [])}; windows {r.get('data', {}).get('windows')}.",
              ""]
    disk = cache_files()
    if disk:
        L += ["## Disk", "", "The teacher has no disk cache: train_map keeps it in memory (GPU, or the host with "
              "--teacher-host) and FOCUS recomputes the teacher's logits per micro-batch. The windows are cached once per "
              "tokenizer and set (flipquant.data's cache):", ""]
        L += [f"- `{k}`: {v:,} bytes" for k, v in disk.items()]
        L += ["", f"Total: {len(disk)} files, {sum(disk.values()) / 1e6:.1f} MB (2,048 windows = 9.15 MB, 1,024 = 4.58 MB, "
                  "256 = 1.15 MB per tokenizer; the audit's tokenizers, loaded from their snapshot paths, have their own "
                  "cache keys)."]
    return L


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = build()
    rec["windows"] = dict(nesting=load(CS.WIN / "nesting.json"),
                          audit={k: v for k, v in (load(CS.WIN / "audit.json") or {}).items() if k != "models"},
                          audit_hits={m: v["hits"] for m, v in ((load(CS.WIN / "audit.json") or {}).get("models")
                                                                  or {}).items()})
    rec["step0"] = load(CS.ROOT / "step0" / "step0.json")
    rec["probes"] = load(CS.ROOT / "probes" / "probes.json")
    rec["gptq"] = dict(step0=load(CS.ROOT / "step0" / "gptq_step0.json"), eta=load(CS.ROOT / "gptq_eta.json"),
                       memory=load(CS.ROOT / "gptq_memory.json"), gpu_time_seconds=gptq_cost())
    (OUT / "calib_size.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (OUT / "REPORT.md").write_text("\n".join(report_md(rec)) + "\n")
    (OUT / "WINDOWS.md").write_text("\n".join(windows_md()) + "\n")
    (OUT / "STEP0.md").write_text("\n".join(step0_md()) + "\n")
    (OUT / "PROBES.md").write_text("\n".join(probes_md()) + "\n")
    print((OUT / "REPORT.md").read_text())


if __name__ == "__main__":
    main()

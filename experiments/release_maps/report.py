"""The release report: REPORT.md and summary.json in the release directory, and a draft model card (README.md) per
model, from the driver's records (records/<unit>/run.json), the PPL reports (ppl/*.json) and evaluation.significance.

    python report.py [--models a,b,...]
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

from release import BASE, PY, REL, UNITS, W

RUN_PPL = {"main_ppl": Path("/home/dev/n16k64_campaign/main_ppl/ppl"), "paper": Path("/home/dev/n16k64_campaign/paper/ppl")}
INFO = {
    "qwen3-1.7b": dict(name="Qwen3-1.7B", license="apache-2.0", license_text="Apache License 2.0",
                       ref=("main_ppl", "qwen3_1p7b")),
    "qwen3-8b": dict(name="Qwen3-8B", license="apache-2.0", license_text="Apache License 2.0", ref=("main_ppl", "qwen3_8b")),
    "mistral-7b": dict(name="Mistral-7B-Instruct-v0.3", license="apache-2.0", license_text="Apache License 2.0",
                       ref=("main_ppl", "mistral7b_ins")),
    "mistral-7b-base": dict(name="Mistral-7B-v0.3", license="apache-2.0", license_text="Apache License 2.0",
                            ref=("paper", "mistral7b")),
    "llama3.1-8b": dict(name="Llama-3.1-8B", license="llama3.1", license_text="Llama 3.1 Community License",
                        ref=("paper", "llama8b")),
    "nemotron-nano-9b-v2": dict(name="NVIDIA-Nemotron-Nano-9B-v2", license="other",
                                license_text="NVIDIA Open Model License", ref=("main_ppl", "nemotron9b"),
                                license_extra=dict(license_name="nvidia-open-model-license",
                                                   license_link="https://www.nvidia.com/en-us/agreements/enterprise-software/"
                                                                "nvidia-open-model-license/")),
    "phi4-14b": dict(name="Phi-4", license="mit", license_text="MIT License", ref=("paper", "phi4")),
    "qwen3.8-27b": dict(name="Qwen3.8-27B", license="apache-2.0", license_text="Apache License 2.0",
                        ref=("paper", "qwen27b")),
}
CORPORA = (("wiki", "WikiText-2"), ("c4", "C4"))
LLAMA_TERMS = ("The Llama 3.1 Community License (section 1.b) applies to derivative works: provide a copy of the "
               "agreement, prominently display “Built with Llama”, include “Llama” at the beginning "
               "of the name of an AI model created or improved with Llama materials or their outputs, and keep the "
               "notice “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta "
               "Platforms, Inc. All Rights Reserved.” in a NOTICE file. These maps are trained against "
               "Llama-3.1-8B's outputs (KL to its BF16 logits).")


def load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def significance(model):
    """evaluation.significance against FourOverSix (BF16 and every map as comparisons); cached in ppl/."""
    d = REL / model / "ppl"
    out = d / "significance.json"
    labels = ["bf16"] + [f"flipquant_{u}" for u in UNITS]
    have = [l for l in labels if (d / f"{l}.json").exists()]
    if not (d / "fo6.json").exists() or not have:
        return None
    sig = load(out)
    if sig is None or sorted(sig["paired"]) != sorted(have):
        cmd = [PY, "-m", "evaluation.significance", "--ref", f"fo6={d / 'fo6.json'}",
               *[a for l in have for a in ("--cmp", f"{l}={d / (l + '.json')}")], "--out", out]
        subprocess.run(cmd, cwd=W, check=True, capture_output=True, text=True,
                       env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
        sig = load(out)
    return sig


def verdict(p):
    if p is None:
        return "—"
    return "inconclusive" if not p["significant"] else ("better" if p["delta"] < 0 else "worse")


def reference(model, release_ppl):
    """The paper-setting maps' PPL under the same convention (run_ppl_deploy records), and whether the record's BF16
    and FourOverSix rows equal this release's per window (the same windows, convention and numerics)."""
    source, key = INFO[model]["ref"]
    root = RUN_PPL[source] / key

    def row(name):
        for n in (name, f"{name}-recheck"):
            r = load(root / n / "report.json")
            if r is not None:
                ev = next(iter(r["evaluations"].values()))["evaluation"]
                return dict(record=str(root / n / "report.json"),
                            ppl={c: ev[c]["ppl"] for c, _ in CORPORA if c in ev},
                            nll={c: ev[c]["nll"] for c, _ in CORPORA if c in ev})
        return None
    out = dict(source=source, key=key, maps={}, anchors={})
    for u in UNITS:
        r = row(f"ours-{u}")
        if r:
            out["maps"][u] = dict(record=r["record"], ppl=r["ppl"])
    for anchor in ("bf16", "fo6"):
        r, mine = row(anchor), release_ppl.get(anchor)
        if r and mine:
            same = {c: r["nll"].get(c) == mine["results"][c]["nll"] for c, _ in CORPORA}
            diff = {c: max(abs(a - b) for a, b in zip(r["nll"][c], mine["results"][c]["nll"]))
                    if len(r["nll"].get(c, [])) == len(mine["results"][c]["nll"]) else None for c, _ in CORPORA}
            out["anchors"][anchor] = dict(record=r["record"], ppl=r["ppl"], per_window_equal=same, max_abs_nll_diff=diff)
    return out


def model_summary(model):
    runs = {u: load(REL / model / "records" / u / "run.json") for u in UNITS}
    runs = {u: r for u, r in runs.items() if r}
    ppl = {l: load(REL / model / "ppl" / f"{l}.json") for l in ["bf16", "fo6"] + [f"flipquant_{u}" for u in UNITS]}
    ppl = {l: r for l, r in ppl.items() if r}
    smoke = load(REL / model / "ppl" / "smoke_fake_16x64.json")
    smoke_log = BASE / "logs" / f"ppl_{model}_smoke_fake_16x64.log"
    smoke_oom = smoke is None and smoke_log.exists() and "OutOfMemoryError" in smoke_log.read_text()
    sig = significance(model)
    return dict(model=model, info=INFO[model], runs=runs, ppl=ppl, smoke=smoke, smoke_oom=smoke_oom, significance=sig,
                reference=reference(model, ppl) if ppl else None)


def fmt_s(x):
    return "—" if x is None else (f"{x / 60:.1f} min" if x >= 120 else f"{x:.0f} s")


def ppl_cell(s, label, c):
    r = s["ppl"].get(label)
    return "—" if r is None else f"{r['results'][c]['ppl']:.4f}"


def delta_cell(s, label, c):
    sig = s["significance"]
    p = sig["paired"].get(label, {}).get(c) if sig else None
    return "—" if p is None else f"{p['delta']:+.4f} ± {2 * p['se']:.4f}{' *' if p['significant'] else ''}"


def ppl_table(s, with_reference=True):
    lines = ["| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict "
             "(WikiText-2 / C4) |", "|---|---:|---:|---:|---:|---|"]
    names = [("bf16", "BF16"), ("fo6", "FourOverSix (NVFP4, stock kernels)")]
    names += [(f"flipquant_{u}", f"FlipQuant {u} (this release)") for u in UNITS]
    sig = s["significance"]
    for label, name in names:
        if label not in s["ppl"]:
            continue
        if label == "fo6":
            v = "reference"
        else:
            p = sig["paired"].get(label) if sig else None
            v = " / ".join(verdict(p.get(c)) if p else "—" for c, _ in CORPORA)
        lines.append(f"| {name} | {ppl_cell(s, label, 'wiki')} | {ppl_cell(s, label, 'c4')} | {delta_cell(s, label, 'wiki')} | "
                     f"{delta_cell(s, label, 'c4')} | {v} |")
    ref = s.get("reference")
    if with_reference and ref and ref["maps"]:
        for u, r in ref["maps"].items():
            lines.append(f"| paper-setting map {u} (reference) | {r['ppl'].get('wiki', float('nan')):.4f} | "
                         f"{r['ppl'].get('c4', float('nan')):.4f} | | | |")
    return "\n".join(lines)


def ppl_costs():
    """{(model, label): (rc, seconds, device peak MiB)} of every PPL run, from the driver's queue log (the last run)."""
    import re
    pat = re.compile(r"END ppl_(?P<rest>\S+) rc=(?P<rc>\d+) (?P<s>\d+)s device_peak=(?P<mib>\d+)MiB")
    labels = ("smoke_fake_16x64", "flipquant_8x64", "flipquant_16x64", "flipquant_256x64", "bf16", "fo6")
    out = {}
    for line in (BASE / "queue.log").read_text().splitlines():
        m = pat.search(line)
        if m:
            rest = m.group("rest")
            label = next(l for l in labels if rest.endswith("_" + l))
            out[(rest[: -len(label) - 1], label)] = (int(m.group("rc")), int(m.group("s")), int(m.group("mib")))
    return out


def tree_measure(model, unit):
    """The process-tree measurement of a unit: the measured re-run (topk-cal's measure.py; the watcher's VmHWM of the
    same re-run), or the live watcher on the release run itself."""
    rm = load(REL / model / "records" / unit / "remeasure.json")
    if rm and rm.get("rc") == 0:
        w = load(BASE / "measure" / f"{model}_{unit}_remeasure.json")
        m = rm["measure"]
        return dict(source="re-run", reproduces=rm["reproduces"], tree=m["peak_rss_gib"]["tree"],
                    wrapper=w["wrapper_vmhwm_gib"] if w else m["peak_rss_gib"]["top"],
                    wrapper_kind="VmHWM" if w else "sampled", largest=m["largest_process_ru_maxrss_gib"],
                    end_to_end=m["wall_seconds"], rerun_trainer_total=rm["trainer"]["total_seconds"],
                    rerun_epoch_mean=sum(rm["trainer"]["epoch_seconds"]) / len(rm["trainer"]["epoch_seconds"]))
    w = load(BASE / "measure" / f"{model}_{unit}.json")
    if w:
        return dict(source="live, attached late" if w["attached_late"] else "live", tree=w["peak_rss_sampled_gib"]["tree"],
                    wrapper=w["wrapper_vmhwm_gib"], wrapper_kind="VmHWM", largest=w["largest_process_vmhwm_gib"],
                    vmhwm_sum=sum(x["vmhwm_gib"] for x in w["processes"]), end_to_end=w["wall_seconds"])
    return dict(source="not measured")


def unit_metrics(model, unit):
    """Time and memory of one calibration: the trainer's PhaseMonitor (the release run), and the tree measurement."""
    tr = load(REL / model / "records" / unit / "trainer_report.json")
    res = tr["resources"]
    phases = {p["name"]: p for p in res["phases"]["phases"] if p["name"] != "training"}
    epochs = [e["epoch_seconds"] for e in tr["epochs"]]
    m = dict(model_load=phases["model_load"]["seconds"], data=phases["data_load"]["seconds"],
             teacher=phases["teacher_precompute"]["seconds"], packing=phases["candidate_packing"]["seconds"],
             setup=tr["setup_seconds"], epoch_mean=sum(epochs) / len(epochs), epoch_min=min(epochs),
             epoch_max=max(epochs), training=tr["training_seconds"], trainer_total=res["total_seconds"],
             gpu_alloc=res["gpu_peak_allocated_gib"][0], gpu_reserved=res["gpu_peak_reserved_gib"][0],
             ru_maxrss=res["cpu_peak_rss_gib"], rss_after_load=phases["model_load"]["host_rss_end"],
             teacher_gib=tr["teacher_storage"]["bytes"] / 2 ** 30, batch=tr["args"]["batch"], accum=tr["args"]["accum"])
    m.update(tree_measure(model, unit))
    return m


def prep_metrics(model):
    m, c = load(BASE / "prep_measure" / f"{model}.measure.json"), load(BASE / "prep_measure" / f"{model}.compare.json")
    if not m:
        return None
    return dict(wall=m["wall_seconds"], tree=m["peak_rss_gib"]["tree"], top=m["peak_rss_gib"]["top"],
                largest=max((v["vmhwm_gib"] for v in m["vmhwm_gib"]), default=None),
                ru_maxrss=m["largest_process_ru_maxrss_gib"], rc=m["returncode"],
                identical=c["identical"] if c else None, files=len(c["files"]) if c else None)


def gib(x):
    return "—" if x is None else f"{x:.1f} GiB"


def time_table(rows):
    out = ["| model | unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) "
           "| training | trainer total | train_map end-to-end | batch × accum |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model, unit, m in rows:
        out.append(f"| {model} | {unit} | {m['model_load']:.1f} s | {m['data']:.1f} s | {m['teacher']:.1f} s | "
                   f"{m['packing']:.1f} s | {m['setup']:.1f} s | {m['epoch_mean']:.1f} s ({m['epoch_min']:.1f}–{m['epoch_max']:.1f}) | "
                   f"{fmt_s(m['training'])} | {fmt_s(m['trainer_total'])} | "
                   f"{fmt_s(m.get('end_to_end'))}{' ᴿ' if m['source'] == 're-run' else ''} | {m['batch']} × {m['accum']} |")
    return out


def late_note(m):
    if "late" not in m["source"]:
        return ""
    return " (sampled after attaching; the processes' exact VmHWM peaks sum to " + gib(m["vmhwm_sum"]) + ")"


def memory_table(rows):
    out = ["| model | unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS after model load "
           "| teacher (top-1000) | wrapper RSS | process-tree peak (10 Hz) | source of the last two |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model, unit, m in rows:
        out.append(f"| {model} | {unit} | {gib(m['gpu_alloc'])} | {gib(m['gpu_reserved'])} | {gib(m['ru_maxrss'])} | "
                   f"{gib(m['rss_after_load'])} | {m['teacher_gib']:.2f} GiB | "
                   f"{gib(m.get('wrapper'))}{' (' + m['wrapper_kind'] + ')' if m.get('wrapper_kind') else ''} | "
                   f"{gib(m.get('tree'))}{late_note(m)} | {m['source']}"
                   f"{'' if m.get('reproduces', True) else ' — MAP DIFFERS'} |")
    return out


def prep_table(models):
    out = ["| model | wall time | process-tree peak (10 Hz) | wrapper | largest process (VmHWM) | records identical to the "
           "release run's |", "|---|---:|---:|---:|---:|---|"]
    for model in models:
        p = prep_metrics(model)
        if p:
            out.append(f"| {model} | {fmt_s(p['wall'])} | {gib(p['tree'])} | {gib(p['top'])} | {gib(p['largest'])} | "
                       f"{'yes' if p['identical'] else 'NO' if p['identical'] is not None else '—'} ({p['files']} files) |")
    return out


PREP_NOTE = ("train_map's first run of a model on a new data root prepares its calibration record and development sets "
             "(`calibration.tmopt_common.prepare` and `prepare_development`; the builder rule's checks, re-tokenizing the "
             "fit windows, hashing the model's weights on the CPU unless an archived record has them, the development "
             "draws). Measured on 2026-10-06 in a fresh data root with measure.py's sampling plus each process's VmHWM; "
             "the files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in "
             "the local Hugging Face cache; a first-ever preparation also downloads them.")
MEASURE_NOTE = ("Time and memory: the trainer's numbers (phases, epochs, GPU peaks, its own `ru_maxrss`, RSS after the "
                "model load, teacher storage) come from its PhaseMonitor in the release run. The wrapper's own RSS, the "
                "process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come from topk-cal's "
                "`measure.py` method: for the first runs (ᴿ, source “re-run”) from a measured re-run of the "
                "same calibration that reproduced the same map, for the later ones from a watcher attached to the run "
                "itself (“live”; Nemotron 16x64 was attached 13 min after its start, so its tree peak covers "
                "only the rest, while its VmHWM peaks are exact). The end-to-end time excludes the one-time data "
                "preparation, which has its own row.")


def card(s):
    model, info = s["model"], s["info"]
    rep = load(REL / model / "records" / next(iter(s["runs"])) / "reproduction.json")
    settings = rep["settings"]
    import torch  # noqa: F401  (maps.load)
    import sys
    sys.path.insert(0, str(W))
    from flipquant import maps as M
    _, _, meta = M.load(REL / model / f"flipquant_{next(iter(s['runs']))}.pt")
    hf_id, revision = meta["model_id"], meta["revision"]
    commit = meta["flipquant"]["commit"]
    front = ["---", f"license: {info['license']}"]
    for k, v in info.get("license_extra", {}).items():
        front.append(f"{k}: {v}")
    front += [f"base_model: {hf_id}", "tags:", "- flipquant", "- nvfp4", "- mixfp4", "- fp4", "- quantization", "---", ""]
    title = (f"# {info['name']} FlipQuant MixFP4 maps (DRAFT)" if model == "llama3.1-8b"   # the license: "Llama" first
             else f"# FlipQuant MixFP4 maps for {info['name']} (DRAFT)")
    lines = front + [title, "",
                     "> Draft model card, for review before any upload. The repository name, the flipquant link and the "
                     "license wording are the uploader's decision.", ""]
    if model == "llama3.1-8b":
        lines += ["**Built with Llama.** " + LLAMA_TERMS + " This directory's `NOTICE` holds that notice; a copy of the "
                  "agreement (the base model's LICENSE) still has to be added by the uploader.", ""]
    lines += ["## What this is", "",
              f"Format maps for [{hf_id}](https://huggingface.co/{hf_id}) at revision `{revision}`, calibrated with "
              "FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all "
              "text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels "
              "0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and "
              "a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint "
              "when it is loaded.", "",
              "| file | tile (output rows × input columns) | E0M3 tiles | sha256 |", "|---|---|---:|---|"]
    for u, r in s["runs"].items():
        lines.append(f"| `flipquant_{u}.pt` | {u.replace('x', ' × ')} | {r['e0m3_tiles']:,} of {r['tiles']:,} "
                     f"({100 * r['e0m3_tiles'] / r['tiles']:.2f} %) | `{r['map_sha256']}` |")
    batch = f"{settings['batch']} sequences per step" if settings["accum"] == 1 else \
        f"micro-batch {settings['batch']} × accumulation {settings['accum']} (8 sequences per step)"
    lines += ["", "8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on "
              "sm_120); 16x64 and 256x64 are coarser.", "",
              "## Calibration", "",
              f"`python -m calibration.train_map --model {model} --unit <unit> --fit-windows 256 --epochs 5 "
              f"--teacher-topk 1000` (flipquant commit `{commit}`):", "",
              "- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per "
              "position plus one tail bucket, per-token FourOverSix activations.",
              f"- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, {batch}, 5 epochs, "
              "deterministic.",
              "- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ "
              "fde8ef8, CodeParrot-clean @ 35a59fb), plus 128 more by the extension rule, which skips the paper's fit, "
              "development and published C4 evaluation documents. Every window is recorded by document hash, offset and "
              "token hash (`records/<unit>/trainer_report.json`).",
              "- These are not the paper's settings (20 epochs, 128 windows, the full-vocabulary teacher). The maps' "
              "`meta` records them as `non_default_settings`.", ""]
    rows = [(model, u, unit_metrics(model, u)) for u in s["runs"]]
    lines += ["Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:", "", *time_table(rows), "",
              *memory_table(rows), ""]
    if prep_metrics(model):
        lines += ["The one-time data preparation (fit and development records, before the first unit):", "", PREP_NOTE, "",
                  *prep_table([model]), ""]
    lines += [MEASURE_NOTE, ""]
    lines += ["## Perplexity", "",
              "Native sm_120 kernels, flipquant's `--paper-convention` (per-token FourOverSix activations), "
              "WikiText-2 (146 windows of 2048 tokens) and C4 (256 windows of 2048 tokens). ΔNLL is the mean paired "
              "per-window difference in nats per token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is "
              "“better” or “worse” only when |Δ| > 2 SE.", "", ppl_table(s), ""]
    ref = s.get("reference")
    if ref and ref["maps"]:
        lines += ["The paper-setting rows are maps calibrated at the paper's settings (20 epochs, 128 windows, the full "
                  "teacher), evaluated under the same convention; they are listed for comparison only.", ""]
    cost = ppl_costs().get((model, "flipquant_16x64"))
    lines += ["## How to use (flipquant)", "",
              "On an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then evaluate with "
              "the native kernels:", "", "```bash",
              "bash kernels/sm120/build_deploy.sh",
              f"python -m evaluation.ppl --model {model} --mode native --weight mixfp4 --map flipquant_16x64.pt "
              "--paper-convention", "```", "",
              *([f"Measured here with the 16x64 map: {cost[2] / 1024:.0f} GiB of GPU memory at peak (2048-token windows, "
                 f"batch 1), {fmt_s(cost[1])} for both corpora, model load included.", ""] if cost else []),
              *(["On other GPUs, the same map with simulated (fake) quantization:", "", "```bash",
                 f"python -m evaluation.ppl --model {model} --mode fake --weight mixfp4 --map flipquant_16x64.pt "
                 "--paper-convention", "```", "",
                 "The native command produced the table above (with `--out`); the simulated one was smoke-tested on two "
                 "windows per corpus."] if not s.get("smoke_oom") else
                ["The native command produced the table above (with `--out`). flipquant's simulated (fake) path does not "
                 "fit this model on one 96 GB GPU: it builds both candidate encodings of every weight on the GPU, next "
                 "to the BF16 model, and ran out of memory while doing so (smoke test, 2026-10-06). Use the native "
                 "kernels, or a GPU with more memory."]),
              "", "flipquant: <repository link to be added by the uploader>.", "",
              "## License", "",
              f"The base model is under the {info['license_text']}. These maps are derived from it; this draft "
              "assumes the same license (the uploader's decision)." + (" " + LLAMA_TERMS if model == "llama3.1-8b" else ""), "",
              "## Files", "",
              "- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; "
              "`flipquant.maps.load` reads them).",
              "- `records/<unit>/`: `reproduction.json` and `run.json` (settings, commit, time and memory), the "
              "trainer's report (`trainer_report.json`: phases, the fit extension's windows) and its log.",
              "- `ppl/`: the evaluation reports (per-window NLLs) and `significance.json`."]
    if model == "llama3.1-8b":
        lines += ["- `NOTICE`: the Llama 3.1 attribution notice."]
    lines += [""]
    return "\n".join(lines)


def checks_section(summaries):
    """Provenance and checks, computed from the records."""
    import sys
    sys.path.insert(0, str(W))
    from flipquant import maps as M
    out = ["## Provenance and checks", ""]
    commits = {}
    for s in summaries:
        for u, r in s["runs"].items():
            c = r["checkout"]
            commits.setdefault((c["commit"][:7], c["clean"]), []).append(f"{s['model']} {u}")
    out.append("- **flipquant commit of each run** (branch optimized-defaults, pushed): " + "; ".join(
        f"`{c}`{'' if clean else ' (dirty)'}: {len(v)} maps" for (c, clean), v in commits.items()) + ".")
    ext = {s["model"]: next(iter(s["runs"].values()))["extension"]["skipped"] for s in summaries}
    ok = all(r["extension"]["ok"] for s in summaries for r in s["runs"].values())
    # the fit set (the paper's 128 windows and the 128 extension windows) against the C4 documents of the PPL evaluation
    c4_overlap = {}
    for s in summaries:
        fo6 = s["ppl"].get("fo6")
        if not fo6:
            continue
        c4 = {d["document_sha256"] for d in fo6["results"]["c4"]["data"]["documents"]}
        for u in s["runs"]:
            tr = load(REL / s["model"] / "records" / u / "trainer_report.json")
            docs = {r["document_sha256"] for r in tr["fit_extension"]["records"]}
            fit = load(BASE / "tmopt_data" / load(REL / s["model"] / "records" / u / "reproduction.json")["model_key"]
                       / "calibration" / "report.json")
            docs |= {d["document_sha256"] for src in ("math", "code") for d in fit["fit"][src]["documents"]}
            c4_overlap.setdefault(s["model"], set()).update(docs & c4)
        c4_overlap[s["model"]] = (len(c4_overlap[s["model"]]), len(c4))
    out.append(f"- **Fit extension** ({'every run passes' if ok else 'FAILURES, see run.json'}): 128 new windows (64 "
               "math, 64 code), distinct documents and tokens, none of them a paper fit, development or published C4 "
               "document, nor a development window's tokens; identical across each model's three units. Skip sets "
               "(fit / development / published C4 documents): " + "; ".join(
                   f"{m} {v['fit']}/{v['development']}/{v['c4_evaluation']}" for m, v in ext.items()) +
               ". Only Llama-3.1-8B and Qwen3.8-27B have a published C4 record, so for the others the rule skips no C4 "
               "document. Checked directly instead: the 256 fit windows' documents against the C4 documents of this "
               "release's PPL evaluation (overlap / C4 documents): " + "; ".join(
                   f"{m} {n}/{t}" for m, (n, t) in c4_overlap.items()) + ".")
    # the fit record (the first 128 windows) of each model, against the record of its paper-setting maps
    import hashlib
    paper_data = Path("/home/dev/n16k64_campaign/multimodel/data")      # experiments/paper/paper_common.DATA
    paper_record = {"phi4-14b": paper_data / "phi4", "qwen3.8-27b": paper_data / "qwen27b",
                    "mistral-7b-base": paper_data / "mistral7b"}
    parts = []
    for s in summaries:
        model, u = s["model"], next(iter(s["runs"]))
        key = load(REL / model / "records" / u / "reproduction.json")["model_key"]
        mine = hashlib.sha256((BASE / "tmopt_data" / key / "calibration" / "report.json").read_bytes()).hexdigest()
        committed = W / "maps" / model / f"flipquant_{u}.pt"
        recorded = M.load(committed)[2]["source"].get("calibration_record_sha256") if committed.exists() else None
        if recorded:
            parts.append(f"{model}: {'the same file' if recorded == mine else 'NOT the same file'} as its committed "
                         "paper-setting maps' record")
        elif model in paper_record:
            theirs = hashlib.sha256((paper_record[model] / "calibration" / "report.json").read_bytes()).hexdigest()
            parts.append(f"{model}: {'the same file' if theirs == mine else 'NOT the same file'} as the paper runs' "
                         "record (multimodel/data)")
        elif model == "llama3.1-8b":
            parts.append("llama3.1-8b: flipquant's own record of the archived Llama fit set (`" + mine[:12] + "…`), not "
                         "the paper runs' file; with it, `train_map --require-reproduction` reproduced the paper's "
                         "Llama-3.1-8B 8x64 map bit for bit on 2026-10-05")
    out.append("- **Fit record** (the first 128 windows, sha256 of the record file): " + "; ".join(parts) + ".")
    q = load(BASE / "tmopt_data" / "qwen27b" / "prepare_summary.json")
    if q and isinstance(q.get("development"), dict):
        out.append("- **Qwen3.8-27B development sets:** redrawn by the vendored preparation and accepted by hash: " + "; ".join(
            f"{n} `{v['expected'][:12]}` {'matches' if v['match'] else 'MISMATCH'} (draw {v['draw_index']})"
            for n, v in q["development"].items()) + ".")
    # the one release setting run before: Llama-3.1-8B 16x64 = topk-cal a53ce55's E256 run (and this branch's check (c))
    e256 = "7bdc6e39cce82f665e6ecf48c62cf084cd5abc8ca7dee516ff48ceece24389e4"
    llama = next((s for s in summaries if s["model"] == "llama3.1-8b"), None)
    if llama and "16x64" in llama["runs"]:
        got = llama["runs"]["16x64"]["run_map_sha256"]
        out.append(f"- **Repeat of an earlier run:** Llama-3.1-8B 16x64 at these settings was run before as topk-cal "
                   f"a53ce55's E256 (map `{e256[:16]}…`); this release's run map is `{got[:16]}…`, "
                   f"{'the same' if got == e256 else 'DIFFERENT'}.")
    v = load(BASE / "validation" / "mistral-7b-base_8x64" / "reproduction.json")
    if v:
        p = v["references"].get("paper_map", {})
        out.append(f"- **mistral-7b-base routing:** `train_map --model mistral-7b-base --unit 8x64 --require-reproduction` "
                   f"at `{v['flipquant']['commit'][:7]}` {'reproduces' if p.get('equal') else 'does NOT reproduce'} the "
                   f"paper's Mistral-7B-v0.3 8x64 map (`{p.get('sha256', '')[:16]}…`).")
    comp = [(s["model"], a, all(x["per_window_equal"].values())) for s in summaries if s.get("reference")
            for a, x in s["reference"]["anchors"].items()]
    if comp:
        bad = [f"{m} {a}" for m, a, eq in comp if not eq]
        out.append(f"- **Reference rows:** the paper-setting PPL records' BF16 and FourOverSix rows equal this release's "
                   f"per window for {sum(eq for *_, eq in comp)} of {len(comp)} rows" + (f" (not: {', '.join(bad)})" if bad else "")
                   + ", so those records use the same windows, convention and numerics.")
    reruns = []
    for s in summaries:
        for u in s["runs"]:
            m = unit_metrics(s["model"], u)
            if m["source"] == "re-run":
                reruns.append((s["model"], u, m["reproduces"], m["rerun_trainer_total"] / m["trainer_total"] - 1,
                               m["rerun_epoch_mean"] / m["epoch_mean"] - 1))
    if reruns:
        bad = [f"{mo} {u}" for mo, u, ok, *_ in reruns if not ok]
        out.append(f"- **Measurement re-runs** (topk-cal's measure.py): {len(reruns)} calibrations re-run for the process-tree "
                   f"numbers; {len(reruns) - len(bad)} of them reproduce their release map (run-map sha256 and tiles)"
                   + (f", NOT: {', '.join(bad)}" if bad else "") + ". Against the release runs, the re-runs' trainer total "
                   f"differs by at most {100 * max(abs(r[3]) for r in reruns):.1f} % and the mean epoch time by at most "
                   f"{100 * max(abs(r[4]) for r in reruns):.1f} %.")
    smokes = [(s["model"], s["smoke"]) for s in summaries]
    out.append("- **Simulated-quantization command** (model cards): smoke-tested on two windows per corpus for " +
               ", ".join(m for m, sm in smokes if sm) + "".join(
                   f"; {s['model']}: out of GPU memory (flipquant's fake path builds both candidates of every weight on "
                   "the GPU next to the BF16 model; 96 GB is not enough for it), so its card gives the native command only"
                   for s in summaries if s.get("smoke_oom")) + ".")
    return out + [""]


def report(summaries):
    lines = ["# FlipQuant release maps (DRAFT records)", "",
             "Calibrated with flipquant's `calibration.train_map --fit-windows 256 --epochs 5 --teacher-topk 1000` "
             "(learning rate 0.02, initial logit −1, seed 0, the paper's per-model batch), one job at a time on one "
             "NVIDIA RTX PRO 6000 (sm_120). PPL: native kernels, `--paper-convention`, WikiText-2 and C4 (all windows). "
             "Nothing here is uploaded anywhere.", "",
             "## Calibration: E0M3 tiles", "",
             "| model | unit | flipquant commit | E0M3 tiles | extension check |", "|---|---|---|---:|---|"]
    for s in summaries:
        for u, r in s["runs"].items():
            lines.append(f"| {s['model']} | {u} | `{r['checkout']['commit'][:7]}`{'' if r['checkout']['clean'] else ' (dirty)'} "
                         f"| {r['e0m3_tiles']:,} / {r['tiles']:,} ({100 * r['e0m3_tiles'] / r['tiles']:.2f} %) | "
                         f"{'pass' if r['extension']['ok'] else 'FAIL'} |")
    rows = [(s["model"], u, unit_metrics(s["model"], u)) for s in summaries for u in s["runs"]]
    lines += ["", "## Calibration: time", "", *time_table(rows), "", "## Calibration: memory", "", *memory_table(rows), "",
              "## One-time data preparation", "", PREP_NOTE, "", *prep_table([s["model"] for s in summaries]), "",
              MEASURE_NOTE]
    lines += ["", "## Perplexity", ""]
    for s in summaries:
        if not s["ppl"]:
            continue
        lines += [f"### {s['model']} ({s['info']['name']})", "", ppl_table(s), ""]
        ref = s.get("reference")
        if ref and ref["anchors"]:
            eq = "; ".join(f"{a}: per-window NLLs " + ("identical" if all(v["per_window_equal"].values())
                                                         else f"differ (max |Δ| {v['max_abs_nll_diff']})")
                           for a, v in ref["anchors"].items())
            lines += [f"Paper-setting reference: `{ref['source']}` records ({ref['key']}); against this release, {eq}.", ""]
    lines += checks_section(summaries)
    lines += ["## Map sha256", "", "| model | file | sha256 |", "|---|---|---|"]
    for s in summaries:
        for u, r in s["runs"].items():
            lines.append(f"| {s['model']} | flipquant_{u}.pt | `{r['map_sha256']}` |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(INFO))
    args = ap.parse_args()
    summaries = []
    for model in args.models.split(","):
        if not (REL / model).exists():
            continue
        s = model_summary(model)
        if not s["runs"]:
            continue
        summaries.append(s)
        (REL / model / "README.md").write_text(card(s))
        if model == "llama3.1-8b":
            (REL / model / "NOTICE").write_text("Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright \u00a9 "
                                                "Meta Platforms, Inc. All Rights Reserved.\n")
    (REL / "REPORT.md").write_text(report(summaries))
    slim = [{k: v for k, v in s.items() if k != "ppl"} | dict(ppl={l: {c: r["results"][c]["ppl"] for c, _ in CORPORA}
                                                                    for l, r in s["ppl"].items()}) for s in summaries]
    (REL / "summary.json").write_text(json.dumps(slim, indent=1, default=str) + "\n")
    print(f"wrote {REL / 'REPORT.md'} and {len(summaries)} model cards")


if __name__ == "__main__":
    main()

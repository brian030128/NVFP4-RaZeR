"""Part P's Hugging Face upload (the user's request of 2026-10-10, relayed by the coordinator): Llama-3.1-8B-Instruct's
FlipQuant maps and FOCUS state as two PRIVATE repositories under edgeai-lab, in the formats of the release-map
repositories (fqrel/hf_stage.py, release-maps HF_UPLOAD.md) and the FOCUS repositories (focus_hf.py).

- edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant (/home/dev/flipquant_hf/Llama-3.1-8B-Instruct-FlipQuant): the three part-P
  maps with their meta cleaned as for the release (local paths relative, no GitHub owner, no reference-implementation
  branch / commit; tiles and every tensor record bit-identical; both sha256 recorded), the card, ppl_summary.json, and
  the Llama license files.
- edgeai-lab/Llama-3.1-8B-Instruct-FOCUS-NVFP4 (/home/dev/focus_hf/Llama-3.1-8B-Instruct-FOCUS-NVFP4): focus.pt as is
  (a reflink copy; scanned like the five FOCUS states), the card, ppl_summary.json, and the Llama license files.
- Llama license files (both repositories): LICENSE and USE_POLICY.md of meta-llama/Llama-3.1-8B-Instruct at the pinned
  revision, downloaded read-only with the active token into /home/dev/flipquant_hf_licenses/llama_instruct, and the
  NOTICE of the base-model release repository (the agreement's notice, byte-identical); "Built with Llama" in the cards.
Every staged file is scanned with the release / FOCUS patterns; the records go to /home/dev/flipquant_hf/llama_instruct/.

    python llama_hf.py licenses   # read-only download of the two license files
    python llama_hf.py stage      # offline staging and scans; stage.json
    python llama_hf.py check      # read-only: the active account, no target repository exists
    python llama_hf.py upload     # create PRIVATE repos, upload, verify; hf_upload.json
    python llama_hf.py cards      # the Hub's YAML validation of each card, README equal to staged, local render
    python llama_hf.py record     # results/paper_eval/llama_instruct/HF_UPLOAD.md (+ the JSON records, cards)
"""
import copy
import datetime
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import focus_hf as FH  # noqa: E402
import run_llama_instruct as P  # noqa: E402
import tables_cpu as T  # noqa: E402

MODEL = "llama3.1-8b-instruct"
TITLE = "Llama-3.1-8B-Instruct"
HF_ID, REV = "meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"
NS = "edgeai-lab"
FQ_REPO, FOCUS_REPO = f"{NS}/Llama-3.1-8B-Instruct-FlipQuant", f"{NS}/Llama-3.1-8B-Instruct-FOCUS-NVFP4"
FQ_STAGE = Path("/home/dev/flipquant_hf") / FQ_REPO.split("/")[1]
FOCUS_STAGE = Path("/home/dev/focus_hf") / FOCUS_REPO.split("/")[1]
RECS = Path("/home/dev/flipquant_hf") / "llama_instruct"
LIC = Path("/home/dev/flipquant_hf_licenses") / "llama_instruct"
BASE_NOTICE = Path("/home/dev/flipquant_hf/Llama-3.1-8B-FlipQuant/NOTICE")
D = P.ROOT / MODEL
STATE = P.ROOT / "focus_states" / MODEL / "state"
OUT = T.ROOT / "results" / "paper_eval" / "llama_instruct"
RUN_PREFIX = str(D) + "/"
FQREL_PREFIX = "/home/dev/n16k64_campaign/fqrel/"
OWNER_PREFIX = "brian030128/flipquant "
LLAMA_NOTICE = ("Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights "
                "Reserved.")
LLAMA_TERMS = ("The Llama 3.1 Community License (section 1.b) applies to derivative works: provide a copy of the agreement, "
               "prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created "
               "or improved with Llama materials or their outputs, and keep the notice “" + LLAMA_NOTICE + "” in a NOTICE "
               "file.")
FQ_TAGS = ["quantization", "fp4", "nvfp4", "mixfp4", "flipquant"]
FQ_CODE = "The FlipQuant code will be released with the paper."
SAME_WINDOWS = ("Its 256 windows are identical, token for token, to the base model's release windows "
                "(Llama-3.1-8B-FlipQuant): the Instruct model shares the base tokenizer, the paper rule (`builder_seed0` "
                "with this model's tokenizer) reproduces the base model's 128 fit windows exactly, and the extension rule "
                "gives the same 128 more.")


# scan hits that the release-map upload already accepted (release-maps HF_UPLOAD.md): Meta's reporting address inside
# the official USE_POLICY.md (byte-identical to the base model's), and the words "commit" / "branch" in the release
# card's own sentences (the FlipQuant commit is kept; the provenance paragraph), which the release staging's scan did not
# flag only because it did not apply focus_hf's SCAN_REF
ACCEPTED = [
    lambda h: h["file"] == "USE_POLICY.md" and h["match"] == "LlamaUseReport@meta.com",
    lambda h: h["file"] == "README.md" and h["match"] in ("commit", "branch") and (
        h["text"].startswith("`python -m calibration.train_map --model llama3.1-8b-instruct") or
        h["text"].startswith("**Provenance of the map files.**")),
]


def unaccepted(hits):
    return [h for h in hits if not any(rule(h) for rule in ACCEPTED)]


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------------------------------------- licenses
def licenses():
    from huggingface_hub import hf_hub_download
    LIC.mkdir(parents=True, exist_ok=True)
    rec = dict(source=f"{HF_ID}@{REV}", retrieved=now(), files={})
    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
        for f in ("LICENSE", "USE_POLICY.md"):
            p = hf_hub_download(HF_ID, f, revision=REV, cache_dir=tmp)       # gated: the active token, read-only
            (LIC / f).write_bytes(Path(p).read_bytes())
    text = re.sub(r"\s+", " ", (LIC / "LICENSE").read_text())
    assert (LIC / "LICENSE").read_text().startswith("LLAMA 3.1 COMMUNITY LICENSE AGREEMENT")
    for s in ("Llama 3.1 Version Release Date: July 23, 2024", "1. License Rights and Redistribution", "Built with Llama",
              "2. Additional Commercial Terms", "3. Disclaimer of Warranty", "4. Limitation of Liability",
              "5. Intellectual Property", "6. Term and Termination", "7. Governing Law and Jurisdiction"):
        assert s in text, s
    assert "Acceptable Use Policy" in (LIC / "USE_POLICY.md").read_text()
    base = Path("/home/dev/flipquant_hf_licenses/llama")
    for f in ("LICENSE", "USE_POLICY.md"):
        rec["files"][f] = dict(sha256=FH.sha(LIC / f), bytes=(LIC / f).stat().st_size,
                               equal_to_base_model_file=(base / f).exists() and FH.sha(base / f) == FH.sha(LIC / f))
    (LIC / "licenses.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


def license_files(dst):
    for f in ("LICENSE", "USE_POLICY.md"):
        shutil.copyfile(LIC / f, dst / f)
    shutil.copyfile(BASE_NOTICE, dst / "NOTICE")
    assert (dst / "NOTICE").read_text().strip() == LLAMA_NOTICE
    return {f: dict(sha256=FH.sha(dst / f)) for f in ("LICENSE", "USE_POLICY.md", "NOTICE")}


# --------------------------------------------------------------------------------------------------------- FlipQuant
def tensor_records(path):
    z = zipfile.ZipFile(path)
    return {n.split("/", 1)[1]: z.read(n) for n in z.namelist() if "/data/" in n}


def neutral_meta(meta):
    """fqrel/hf_stage.neutral_meta for these maps: the run directory relative to the model's part-P directory, the
    calibration record relative to the release work directory, the extension reference without its GitHub owner, and
    the reference implementation's branch and commit removed."""
    m, changes = copy.deepcopy(meta), []
    for k, prefix in (("run", RUN_PREFIX), ("calibration_record", FQREL_PREFIX)):
        v = m["source"].get(k)
        assert isinstance(v, str) and v.startswith(prefix), (k, v)
        m["source"][k] = v[len(prefix):]
        changes.append((f"source.{k}", v, m["source"][k]))
    v = m["fit_extension"]["flipquant_reference"]
    assert v.startswith(OWNER_PREFIX), v
    m["fit_extension"]["flipquant_reference"] = "flipquant " + v[len(OWNER_PREFIX):]
    changes.append(("fit_extension.flipquant_reference", v, m["fit_extension"]["flipquant_reference"]))
    for k in ("branch", "commit"):
        if k in m:
            changes.append((k, m.pop(k), None))
    assert m["flipquant"]["commit"] and m["repo"] == "the reference implementation" and m["model_id"] == HF_ID
    return m, changes


def stage_map(unit):
    from flipquant import maps as M
    src = D / "maps" / f"flipquant_{unit}.pt"
    tiles, u, meta = M.load(src)
    new_meta, changes = neutral_meta(meta)
    out = FQ_STAGE / src.name
    M.save(out, tiles, u, new_meta)
    import torch
    t2, u2, m2 = M.load(out)
    assert u2 == u and set(t2) == set(tiles)
    assert all(t2[n].dtype == tiles[n].dtype and t2[n].shape == tiles[n].shape and torch.equal(t2[n], tiles[n])
               for n in tiles), "tiles differ"
    a, b = tensor_records(src), tensor_records(out)
    assert set(a) == set(b) and all(a[k] == b[k] for k in a), "tensor records differ"
    assert m2 == new_meta
    return dict(file=src.name, unit=unit, sha256_release=FH.sha(src), sha256_uploaded=FH.sha(out), meta_changes=changes,
                modules=len(tiles), tiles=sum(t.numel() for t in tiles.values()),
                e0m3_tiles=int(sum(int(t.sum()) for t in tiles.values())), tiles_bit_identical=True,
                tensor_records_identical=True, meta=new_meta)


def scan_map(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    strings = list(FH.walk(obj["meta"], "meta")) + [("tiles key", k) for k in obj["tiles"]] + \
        [("format", obj["format"]), ("unit", obj["unit"])]
    hits = [dict(file=path.name, where=w, match=m.group(0), text=s[:200]) for w, s in strings for m in FH.SCAN.finditer(s)]
    hits += [dict(file=path.name, where=f"raw bytes @{off}", match=m, text="") for off, m in FH.raw_hits(path).items()]
    return hits


def unit_rows():
    """Per unit: the trainer's phases and peaks (its report) and the run's wall time and device peak (run.json)."""
    rows = {}
    for unit in P.UNITS:
        run = json.loads((D / "records" / unit / "run.json").read_text())
        rep = json.loads((D / "records" / unit / "trainer_report.json").read_text())
        res, bp = rep["resources"], rep["resources"]["phases"]["by_phase"]
        ep = [e["epoch_seconds"] for e in rep["epochs"]]
        setup = sum(bp[k]["seconds"] for k in ("model_load", "data_load", "teacher_precompute", "candidate_packing"))
        rows[unit] = dict(model_load=bp["model_load"]["seconds"], data=bp["data_load"]["seconds"],
                          teacher=bp["teacher_precompute"]["seconds"], packing=bp["candidate_packing"]["seconds"],
                          setup=setup, epoch_mean=sum(ep) / len(ep), epoch_min=min(ep), epoch_max=max(ep),
                          training=bp["training"]["seconds"], trainer_total=res["total_seconds"],
                          wall=run["seconds"], train_map_wall=run["train_map_wall_seconds"],
                          batch=run["settings"]["batch"], accum=run["settings"]["accum"],
                          gpu_alloc=res["gpu_peak_allocated_gib"][0], gpu_reserved=res["gpu_peak_reserved_gib"][0],
                          rss=res["cpu_peak_rss_gib"], rss_load=bp["model_load"]["host_peak_rss_gib"],
                          teacher_gib=rep["teacher_storage"]["bytes"] / 2 ** 30, device_peak_mib=run["device_peak_mib"],
                          settings=run["settings"])
    return rows


def fmt(s):
    return f"{s:.1f} s" if s < 100 else f"{s / 60:.1f} min"


def record_prep_seconds():
    lines = (P.G.RUN / "queue.log").read_text().splitlines()
    t = [l.split()[0] for l in lines if "record llama3.1-8b-instruct: START" in l or "record llama3.1-8b-instruct: END" in l]
    a, b = (datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ") for x in t[-2:])
    return (b - a).total_seconds()


def fq_summary(staged, final):
    fo6 = json.loads((D / "ppl" / "fo6.json").read_text())
    meta = staged["16x64"]["meta"]
    labels = ["bf16", "nvfp4", "fo6"] + [f"flipquant_{u}" for u in P.UNITS]
    key = {"bf16": "bf16", "nvfp4": "nvfp4", "fo6": "fo6", **{f"flipquant_{u}": f"fq_{u}" for u in P.UNITS}}
    out = dict(model=HF_ID, revision=REV, flipquant_model_key=MODEL,
               evaluation=dict(mode="native sm_120 kernels (FlipQuant evaluation.ppl --mode native)",
                               convention=fo6["convention"], quantized_modules=fo6["quantized_modules"],
                               transformers=fo6["source"]["transformers"], torch=fo6["source"]["torch"], gpu=meta["gpu"],
                               nvfp4="per-token NVFP4 activations (--act-scope row), as the paper's NVFP4 rows",
                               corpora={c: {"dataset": FH.DATASETS[c]} | {k: v for k, v in fo6["results"][c]["data"].items()
                                                                           if k != "documents"} for c, _ in T.CORPORA},
                               c4_documents="the documents are drawn with the recorded seed from the listed shard"),
               ppl={l: final["ppl"][MODEL][key[l]] for l in labels}, paired_vs_fourover6={}, maps={},
               calibration=dict(settings=meta["settings"], non_default_settings=meta["non_default_settings"],
                                windows=SAME_WINDOWS, per_unit={u: {k: v for k, v in r.items() if k != "settings"}
                                                                for u, r in unit_rows().items()}))
    for l in labels:
        if l != "fo6":
            out["paired_vs_fourover6"][l] = {c: dict(delta_nll=p["delta"], two_se=p["two_se"], n=p["n"],
                                                     significant=p["significant"], verdict=FH.verdict(p))
                                             for c, p in final["vs_fo6"][MODEL][key[l]].items()}
    for u, st in staged.items():
        out["maps"][f"flipquant_{u}.pt"] = dict(sha256=st["sha256_uploaded"], e0m3_tiles=st["e0m3_tiles"], tiles=st["tiles"])
    return out


def fq_card(staged, summary):
    meta = staged["16x64"]["meta"]
    commit = meta["flipquant"]["commit"]
    rows = unit_rows()
    front = {"license": "llama3.1", "base_model": HF_ID, "tags": FQ_TAGS}
    L = ["---", yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=1000).strip(), "---", "",
         f"# {TITLE} FlipQuant MixFP4 maps", "", "**Built with Llama.**", "",
         f"{LLAMA_NOTICE} The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in `USE_POLICY.md`, "
         "and the notice in `NOTICE`.", "",
         "## What this is", "",
         f"Format maps for [{HF_ID}](https://huggingface.co/{HF_ID}) at revision `{REV}`, calibrated with FlipQuant "
         "(TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all text-model linears except "
         "the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels 0..7), `False` as E2M1 "
         "(FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and a per-tensor FP32 scale. The "
         "maps hold no weights: they are applied to the base model's BF16 checkpoint when it is loaded.", "",
         "| file | tile (output rows × input columns) | E0M3 tiles | sha256 |", "|---|---|---:|---|"]
    for u, st in staged.items():
        L.append(f"| `flipquant_{u}.pt` | {u.replace('x', ' × ')} | {st['e0m3_tiles']:,} of {st['tiles']:,} "
                 f"({100 * st['e0m3_tiles'] / st['tiles']:.2f} %) | `{st['sha256_uploaded']}` |")
    L += ["", "8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); "
          "16x64 and 256x64 are coarser.", "", FQ_CODE, "",
          "## Calibration", "",
          f"`python -m calibration.train_map --model {MODEL} --unit <unit> --fit-windows 256 --epochs 5 --teacher-topk 1000` "
          f"(FlipQuant commit `{commit}`):", "",
          "- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per "
          "position plus one tail bucket, per-token FourOverSix activations.",
          f"- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, "
          f"{rows['16x64']['batch']} sequences per step, 5 epochs, deterministic.",
          "- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ "
          "fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule with this model's tokenizer, plus 128 more by "
          "the extension rule, which skips the paper's fit, development and published C4 evaluation documents. The "
          "calibration records every window by document hash, offset and token hash. " + SAME_WINDOWS,
          "- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' "
          "`meta` records them as `non_default_settings`.", "",
          "Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:", "",
          "| unit | model load | data (+ extension) | teacher | lean packing | setup | per epoch (mean, min–max) | training | "
          "trainer total | train_map end-to-end | batch × accum |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for u, r in rows.items():
        L.append(f"| {u} | {fmt(r['model_load'])} | {fmt(r['data'])} | {fmt(r['teacher'])} | {fmt(r['packing'])} | "
                 f"{fmt(r['setup'])} | {fmt(r['epoch_mean'])} ({r['epoch_min']:.1f}–{r['epoch_max']:.1f}) | "
                 f"{fmt(r['training'])} | {fmt(r['trainer_total'])} | {fmt(r['train_map_wall'])} | "
                 f"{r['batch']} × {r['accum']} |")
    L += ["", "| unit | GPU peak allocated | GPU peak reserved | trainer peak RSS (ru_maxrss) | RSS peak during model load | "
          "teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for u, r in rows.items():
        L.append(f"| {u} | {r['gpu_alloc']:.1f} GiB | {r['gpu_reserved']:.1f} GiB | {r['rss']:.1f} GiB | "
                 f"{r['rss_load']:.1f} GiB | {r['teacher_gib']:.2f} GiB | {r['device_peak_mib'] / 1024:.1f} GiB |")
    L += ["", "Time and memory: the trainer's own phase timer and peak counters give the phases, the epochs, the GPU peaks, "
          "its `ru_maxrss` and the teacher storage; the end-to-end time and the device memory peak come from the run's "
          "wrapper (nvidia-smi sampled every 2 s).", "",
          f"The one-time data preparation (the calibration record: the paper rule's 128 windows re-tokenized and checked, "
          f"the weights hashed on the CPU, the development records) took {record_prep_seconds() / 60:.0f} min end to end, "
          "including the first download of the model's weights.", ""]
    rec = json.loads((D / "ppl" / "fo6.json").read_text())
    win = {c: (rec["results"][c]["windows"], rec["results"][c]["data"]["window_tokens"]) for c, _ in T.CORPORA}
    L += ["## Perplexity", "",
          f"Native sm_120 kernels, FlipQuant's `--paper-convention` (per-token FourOverSix activations; NVFP4 with "
          f"per-token NVFP4 activations), WikiText-2 ({win['wiki'][0]} windows of {win['wiki'][1]} tokens) and C4 "
          f"({win['c4'][0]} windows of {win['c4'][1]} tokens). ΔNLL is the mean paired per-window difference in nats per "
          "token against FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The "
          "numbers are also in `ppl_summary.json`.", "",
          "| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict "
          "(WikiText-2 / C4) |", "|---|---:|---:|---:|---:|---|"]
    names = [("bf16", "BF16"), ("nvfp4", "NVFP4 (per-token NVFP4 activations)"), ("fo6", "FourOverSix (NVFP4, stock kernels)"),
             *[(f"flipquant_{u}", f"FlipQuant {u} (this release)") for u in P.UNITS]]
    for l, name in names:
        p = summary["ppl"][l]
        if l == "fo6":
            d, v = ["—", "—"], "reference"
        else:
            q = summary["paired_vs_fourover6"][l]
            d = [f"{q[c]['delta_nll']:+.4f} ± {q[c]['two_se']:.4f}{' *' if q[c]['significant'] else ''}" for c, _ in T.CORPORA]
            v = " / ".join(q[c]["verdict"] for c, _ in T.CORPORA)
        L.append(f"| {name} | {p['wiki']:.4f} | {p['c4']:.4f} | {d[0]} | {d[1]} | {v} |")
    q16 = json.loads((D / "ppl" / "fq-16x64.json").read_text())["resources"]
    L += ["", "## How to use", "", FQ_CODE + " With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the "
          "kernels once, then evaluate with the native kernels:", "", "```bash", "bash kernels/sm120/build_deploy.sh",
          f"python -m evaluation.ppl --model {MODEL} --mode native --weight mixfp4 --map flipquant_16x64.pt "
          "--paper-convention", "```", "",
          f"Measured here with the 16x64 map: {q16['gpu_peak_gib'][0]:.0f} GiB of GPU memory at peak (2048-token windows, "
          f"batch 1), {q16['seconds']:.0f} s for both corpora, model load included.", "",
          "On other GPUs, the same map with simulated (fake) quantization:", "", "```bash",
          f"python -m evaluation.ppl --model {MODEL} --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention",
          "```", "", "The native command produced the table above.", "",
          "## License", "",
          "These maps are released under the same license as the base model, the Llama 3.1 Community License.", "",
          LLAMA_TERMS + f" These maps are trained against {TITLE}'s outputs (KL to its BF16 logits).", "",
          f"- `LICENSE`: the Llama 3.1 Community License Agreement, the base repository's own `LICENSE` file ({HF_ID} at "
          f"revision `{REV}`).",
          "- `USE_POLICY.md`: the Llama 3.1 Acceptable Use Policy, which the agreement incorporates; the base repository's "
          "own `USE_POLICY.md`.",
          f"- `NOTICE`: “{LLAMA_NOTICE}”", "",
          "## Files", "",
          "- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` "
          "reads them).",
          "- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and "
          "the calibration settings and cost.",
          "- `LICENSE`: see License.", "- `USE_POLICY.md`: see License.", "- `NOTICE`: see License.", "",
          "**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the run "
          "directory and calibration-record paths are relative instead of local absolute paths, the data-extension code "
          "reference names the file without its repository owner, and the internal branch and commit of the reference "
          "implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the "
          "calibration output → this repository's file:", ""]
    L += [f"- `flipquant_{u}.pt`: `{st['sha256_release']}` → `{st['sha256_uploaded']}`" for u, st in staged.items()]
    L += [""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------- FOCUS
def focus_cost():
    r = json.loads((STATE / "report.json").read_text())
    assert r["status"] == "complete" and r["steps"] == 8 and len(r["log"]) == 8
    return dict(steps=r["steps"], micro_batch=int(r["args"]["micro"]), training_seconds=r["training_seconds"],
                end_to_end_seconds=r["resources"]["seconds"], gpu=r["resources"]["gpus"][0],
                gpu_peak_allocated_gib=r["resources"]["gpu_peak_gib"][0], parameters=r["parameters"],
                checkpointed_layers=r["checkpointed_layers"], data_windows=r["data"]["windows"],
                data_tokens=r["data"]["tokens"], kl_top_first_last=(r["log"][0]["kl_top"], r["log"][-1]["kl_top"]))


def focus_card(info, ev, cs, files):
    """focus_hf.card with the Llama license items: "Built with Llama" and the notice under the title, the agreement's
    terms, USE_POLICY.md and NOTICE in the license section and the file list."""
    FH.TITLES[MODEL] = TITLE
    FH.LICENSE[MODEL] = dict(front={"license": "llama3.1"}, name="Llama 3.1 Community License",
                             files=dict(LICENSE=LIC / "LICENSE", NOTICE=BASE_NOTICE))
    text = FH.card(MODEL, info, ev, cs, files)
    head = f"# FOCUS NVFP4 scales for {TITLE} (our reproduction)\n"
    assert text.count(head) == 1
    text = text.replace(head, head + "\n**Built with Llama.**\n\n" + LLAMA_NOTICE + " The agreement is in `LICENSE`, the "
                        "Acceptable Use Policy it incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.\n")
    old = f"`LICENSE` is a copy of the base model's license file ({HF_ID} at revision `{REV}`)."
    assert text.count(old) == 1
    text = text.replace(old, LLAMA_TERMS + f" This state is trained against {TITLE}'s outputs (KL to its BF16 "
                        "logits).\n\n"
                        f"- `LICENSE`: the Llama 3.1 Community License Agreement, the base repository's own `LICENSE` file "
                        f"({HF_ID} at revision `{REV}`).\n"
                        "- `USE_POLICY.md`: the Llama 3.1 Acceptable Use Policy, which the agreement incorporates; the base "
                        "repository's own `USE_POLICY.md`.\n"
                        f"- `NOTICE`: “{LLAMA_NOTICE}”")
    old2 = "- `NOTICE`: see License.\n"
    assert text.count(old2) == 1
    text = text.replace(old2, old2 + "- `USE_POLICY.md`: see License.\n")
    return text


# ------------------------------------------------------------------------------------------------------------- stage
def stage():
    sys.path.insert(0, str(P.S.FQ2))
    RECS.mkdir(parents=True, exist_ok=True)
    final = json.loads((OUT.parent / "final" / "final.json").read_text())
    record = dict(namespace=NS, staged_utc=now(), licenses_fetched=json.loads((LIC / "licenses.json").read_text()),
                  repos={})
    # FlipQuant maps
    FQ_STAGE.mkdir(parents=True, exist_ok=True)
    staged = {u: stage_map(u) for u in P.UNITS}
    lic = license_files(FQ_STAGE)
    summary = fq_summary(staged, final)
    (FQ_STAGE / "ppl_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (FQ_STAGE / "README.md").write_text(fq_card(staged, summary))
    hits = []
    for f in sorted(FQ_STAGE.iterdir()):
        if f.suffix == ".pt":
            hits += scan_map(f)
        else:
            hits += FH.scan_text(f.name, f.read_text(), ref=f.name in ("README.md", "ppl_summary.json"))
    files = {f.name: dict(sha256=FH.sha(f), bytes=f.stat().st_size) for f in sorted(FQ_STAGE.iterdir())}
    record["repos"][FQ_REPO] = dict(model=MODEL, base_model=HF_ID, revision=REV, folder=str(FQ_STAGE),
                                    maps={u: {k: v for k, v in s.items() if k != "meta"} for u, s in staged.items()},
                                    license_files=lic, files=files, scan=hits)
    print(f"{FQ_REPO}: {list(files)} scan hits {len(hits)}", flush=True)
    # FOCUS
    FOCUS_STAGE.mkdir(parents=True, exist_ok=True)
    src, out = STATE / "focus.pt", FOCUS_STAGE / "focus.pt"
    if not out.exists():
        subprocess.run(["cp", "--reflink=always", str(src), str(out)], check=True)
    sha_src, sha_out = FH.sha(src), FH.sha(out)
    assert sha_src == sha_out, "staged focus.pt differs from the source"
    hits, info = FH.scan_state(out)
    assert info["source"]["hf_id"] == HF_ID and info["source"]["revision"] == REV and info["config"]["model"] == MODEL
    ev, cs = FH.evaluated(MODEL), focus_cost()
    assert info["modules"] == ev["record"]["quantized_modules"] == cs["parameters"]["modules"]
    assert info["block_factors_m"] == cs["parameters"]["crs"] and info["subblock_logits_q"] == cs["parameters"]["dgs"]
    lic = license_files(FOCUS_STAGE)
    fsum = FH.ppl_summary(MODEL, info, ev, cs, sha_out, out.stat().st_size)
    (FOCUS_STAGE / "ppl_summary.json").write_text(json.dumps(fsum, indent=1, ensure_ascii=False) + "\n")
    pre = {f.name: dict(bytes=f.stat().st_size, sha256=sha_out if f.name == "focus.pt" else FH.sha(f))
           for f in sorted(FOCUS_STAGE.iterdir()) if f.is_file()}
    cfiles = {"focus.pt": pre["focus.pt"], "ppl_summary.json": pre["ppl_summary.json"],
              **{n: pre[n] for n in ("LICENSE", "NOTICE", "USE_POLICY.md")}}
    (FOCUS_STAGE / "README.md").write_text(focus_card(info, ev, cs, cfiles))
    for f in sorted(FOCUS_STAGE.iterdir()):
        if f.is_file() and f.name != "focus.pt":
            hits += FH.scan_text(f.name, f.read_text(), ref=f.name in ("README.md", "ppl_summary.json"))
    files = {f.name: dict(bytes=f.stat().st_size, sha256=sha_out if f.name == "focus.pt" else FH.sha(f))
             for f in sorted(FOCUS_STAGE.iterdir()) if f.is_file()}
    record["repos"][FOCUS_REPO] = dict(
        model=MODEL, base_model=HF_ID, revision=REV, folder=str(FOCUS_STAGE),
        state=dict(source=str(src), sha256_source=sha_src, sha256_staged=sha_out, identical=True, metadata_changes=[],
                   **{k: v for k, v in info.items() if k not in ("config", "source")}, config=info["config"],
                   source_meta=info["source"]),
        license_files=lic, files=files, scan=hits, ppl=fsum["ppl"], paired_vs_fourover6=fsum["paired_vs_fourover6"],
        calibration_cost=cs)
    print(f"{FOCUS_REPO}: {list(files)} scan hits {len(hits)}", flush=True)
    (RECS / "stage.json").write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n")


# ----------------------------------------------------------------------------------------------------- check / upload
def check():
    api = FH.account()
    st = json.loads((RECS / "stage.json").read_text())
    exists = {r: api.repo_exists(r, repo_type="model") for r in st["repos"]}
    print("existing:", [r for r, e in exists.items() if e] or "none")
    if any(exists.values()):
        sys.exit("a target repository already exists; stopping (nothing is overwritten)")
    hits = {r: dict(all=len(d["scan"]), unaccepted=unaccepted(d["scan"])) for r, d in st["repos"].items()}
    print("scan hits:", {r: (v["all"], len(v["unaccepted"])) for r, v in hits.items()})
    if any(v["unaccepted"] for v in hits.values()):
        sys.exit(f"scan hits need a decision; stopping: {[v['unaccepted'] for v in hits.values()]}")


def upload():
    from huggingface_hub import hf_hub_download
    api = FH.account()
    st = json.loads((RECS / "stage.json").read_text())
    path = RECS / "hf_upload.json"
    out = json.loads(path.read_text()) if path.exists() else dict(account=NS, repos={})
    for repo_id, rec in st["repos"].items():
        if out["repos"].get(repo_id, {}).get("verified"):
            print(f"{repo_id}: already uploaded and verified")
            continue
        assert not unaccepted(rec["scan"]), f"{repo_id}: unaccepted scan hits"
        folder = Path(rec["folder"])
        names = sorted(rec["files"])
        assert {f.name for f in folder.iterdir() if f.is_file()} == set(names), f"{repo_id}: staged files changed"
        for n in names:
            assert (folder / n).stat().st_size == rec["files"][n]["bytes"], f"{repo_id}: {n} changed size"
        if not api.repo_exists(repo_id, repo_type="model"):
            api.create_repo(repo_id, repo_type="model", private=True, exist_ok=False)
        info = api.model_info(repo_id)
        anon = FH.anonymous_status(repo_id)
        if info.private is not True or anon == 200:
            sys.exit(f"{repo_id}: not private (private={info.private}, anonymous HTTP {anon}); stopping")
        api.upload_large_folder(repo_id=repo_id, folder_path=folder, repo_type="model", private=True,
                                allow_patterns=names, print_report_every=300)
        info = api.model_info(repo_id, files_metadata=True)
        anon_after = FH.anonymous_status(repo_id)
        hub = {s.rfilename: s for s in info.siblings}
        problems = []
        if info.private is not True or anon_after == 200:
            problems.append(f"not private (private={info.private}, anonymous HTTP {anon_after})")
        if set(hub) - {".gitattributes"} != set(names):
            problems.append(f"file list {sorted(hub)}")
        card_data = info.card_data.to_dict() if info.card_data else {}
        if card_data.get("base_model") != HF_ID or card_data.get("license") != "llama3.1":
            problems.append(f"card metadata {card_data}")
        files = {}
        for n in names:
            want = rec["files"][n]["sha256"]
            lfs = getattr(hub.get(n), "lfs", None)
            lfs_sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:          # one file at a time, in RAM (not /home)
                p = hf_hub_download(repo_id, n, revision=info.sha, cache_dir=tmp, force_download=True)
                got = FH.sha(p)
            files[n] = dict(sha256=want, bytes=rec["files"][n]["bytes"], hub_size=getattr(hub.get(n), "size", None),
                            hub_lfs_sha256=lfs_sha, download_sha256=got, equal=got == want,
                            lfs_equal=None if lfs_sha is None else lfs_sha == want)
            if got != want or (lfs_sha is not None and lfs_sha != want):
                problems.append(f"{n}: sha256 differs (download {got}, lfs {lfs_sha})")
            print(f"  {repo_id} {n}: download sha256 {'==' if got == want else '!='} staged", flush=True)
        commits = api.list_repo_commits(repo_id)
        out["repos"][repo_id] = dict(url=f"https://huggingface.co/{repo_id}", head=info.sha, private=info.private,
                                     anonymous_http=anon_after, hub_files=sorted(hub),
                                     commits=[dict(id=c.commit_id, title=c.title) for c in commits],
                                     card_metadata=card_data, files=files, problems=problems, verified=not problems,
                                     uploaded_utc=now())
        path.write_text(json.dumps(out, indent=1, default=str) + "\n")
        print(f"{repo_id}: head {info.sha} private={info.private} anonymous={anon_after} problems={problems or 'none'}",
              flush=True)
        if problems:
            sys.exit(f"{repo_id}: verification failed; stopping")


def cards():
    from huggingface_hub import HfApi, ModelCard
    from markdown_it import MarkdownIt
    up = json.loads((RECS / "hf_upload.json").read_text())
    st = json.loads((RECS / "stage.json").read_text())
    md = MarkdownIt("commonmark").enable("table")
    out = {}
    for repo_id in up["repos"]:
        card = ModelCard.load(repo_id, repo_type="model")
        staged = (Path(st["repos"][repo_id]["folder"]) / "README.md").read_text()
        try:
            card.validate(repo_type="model")
            valid = True
        except Exception as e:                                    # noqa: BLE001 -- recorded, not raised
            valid = f"{type(e).__name__}: {e}"
        html = md.render(card.text)
        lines = card.text.splitlines()
        n_tables = sum(1 for i, l in enumerate(lines) if re.match(r"^\|[-:| ]+\|$", l) and i and lines[i - 1].startswith("|"))
        n_fences = sum(1 for l in lines if l.startswith("```")) // 2
        info = HfApi().model_info(repo_id)
        out[repo_id] = dict(private_now=info.private, anonymous_http_now=FH.anonymous_status(repo_id),
                            hub_yaml_valid=valid, readme_equal_to_staged=card.content == staged,
                            card_data=card.data.to_dict(), tables_markdown=n_tables, tables_html=html.count("<table>"),
                            code_fences=n_fences, code_blocks_html=html.count("<pre><code"), h1=html.count("<h1>"),
                            built_with_llama="Built with Llama" in card.text,
                            renders=valid is True and card.content == staged and n_tables == html.count("<table>") and
                            n_fences == html.count("<pre><code") and html.count("<h1>") == 1)
        print(repo_id, {k: v for k, v in out[repo_id].items() if k != "card_data"}, flush=True)
    (RECS / "cards.json").write_text(json.dumps(out, indent=1) + "\n")


# ------------------------------------------------------------------------------------------------------------ record
def record():
    st = json.loads((RECS / "stage.json").read_text())
    up = json.loads((RECS / "hf_upload.json").read_text())
    cj = json.loads((RECS / "cards.json").read_text())
    dst = OUT / "hf"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("stage.json", "hf_upload.json", "cards.json"):
        shutil.copyfile(RECS / name, dst / name)
    shutil.copyfile(LIC / "licenses.json", dst / "licenses.json")
    for repo_id, d in st["repos"].items():
        c = dst / "cards" / repo_id.split("/", 1)[1]
        c.mkdir(parents=True, exist_ok=True)
        for f in ("README.md", "ppl_summary.json"):
            shutil.copyfile(Path(d["folder"]) / f, c / f)
    lic = st["licenses_fetched"]
    L = ["# Hugging Face upload of Llama-3.1-8B-Instruct's FlipQuant maps and FOCUS state (2026-10-10)", "",
         "The user's request (relayed by the coordinator, 2026-10-10): part P's Llama-3.1-8B-Instruct artifacts as two "
         "**private** repositories under `edgeai-lab`, in the formats of the release-map repositories and the FOCUS "
         "repositories, before the HF identity is switched. The active account was checked first (`edgeai-lab`; no token "
         "printed or changed). Each repository was created private and checked private (anonymous 401) before any "
         "file went up; then uploaded and verified.", "",
         "| repository | files | private | anonymous | head | every file's sha256 (download) | card YAML (Hub) | README = staged |",
         "|---|---:|---|---:|---|---|---|---|"]
    for repo_id, d in up["repos"].items():
        ok = all(f["equal"] for f in d["files"].values())
        L.append(f"| `{repo_id}` | {len(d['files'])} | {d['private']} | {d['anonymous_http']} | `{d['head'][:12]}` | "
                 f"{'equal' if ok else 'DIFFERS'} | {cj[repo_id]['hub_yaml_valid']} | {cj[repo_id]['readme_equal_to_staged']} |")
    L += ["", "## What was uploaded", ""]
    fq = st["repos"][FQ_REPO]
    L += [f"**`{FQ_REPO}`**: the three part-P maps with their meta cleaned as for the release (the run directory and "
          "calibration-record paths relative, the extension reference without its GitHub owner, the reference "
          "implementation's branch and commit removed; tiles and every tensor record bit-identical), the card, "
          "`ppl_summary.json`, `LICENSE`, `USE_POLICY.md`, `NOTICE`. sha256 calibration output → uploaded:", ""]
    L += [f"- `{m['file']}`: `{m['sha256_release']}` → `{m['sha256_uploaded']}` ({m['e0m3_tiles']:,} E0M3 tiles of "
          f"{m['tiles']:,})" for m in fq["maps"].values()]
    fo = st["repos"][FOCUS_REPO]
    L += ["", f"**`{FOCUS_REPO}`**: `focus.pt` as is (sha256 `{fo['state']['sha256_staged']}`, equal to part P's state; "
          "metadata, record names and raw bytes scanned, nothing found), the card (our reproduction of FOCUS, "
          "arXiv:2608.01847; not an official release; hyperparameters, the calibration-data difference, per-token "
          "FourOverSix activations, what m / q / s2 are; not a quantized checkpoint; PPL, training time and memory), "
          "`ppl_summary.json`, `LICENSE`, `USE_POLICY.md`, `NOTICE`.", "",
          "## License files", "",
          f"`LICENSE` and `USE_POLICY.md` are {lic['source']}'s own files, downloaded read-only with the active token on "
          f"{lic['retrieved']} (sha256 " + ", ".join(f"{f} `{v['sha256'][:16]}`" for f, v in lic["files"].items()) +
          "; " + ", ".join(f"{f} equal to the base model's: {v['equal_to_base_model_file']}" for f, v in lic["files"].items()) +
          "). `NOTICE` is the base-model release repository's (the agreement's notice). Both cards say “Built with Llama”; "
          "both repository names start with “Llama”.", "",
          "## Scans", "",
          "Every staged file, with the release / FOCUS patterns (local paths, the host, user names, private repository "
          "names, tokens, e-mail addresses, pointers to a reference implementation): " +
          ", ".join(f"`{r}` {len(d['scan'])} hit{'' if len(d['scan']) == 1 else 's'}, {len(unaccepted(d['scan']))} not "
                    "accepted" for r, d in st["repos"].items())
          + ". Accepted, as in the release-map upload: Meta's reporting address `LlamaUseReport@meta.com` inside the "
          "official `USE_POLICY.md` (byte-identical to the base model's), and the words “commit” / “branch” in the release "
          "card's own sentences (the FlipQuant commit is kept; the provenance paragraph).", ""]
    (OUT / "HF_UPLOAD.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    {"licenses": licenses, "stage": stage, "check": check, "upload": upload, "cards": cards, "record": record}.get(
        what, lambda: sys.exit(__doc__))()

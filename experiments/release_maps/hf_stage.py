"""Stage the Hugging Face repos of the release maps; NOTHING is uploaded here (hf_upload.py does that).

/home/dev/flipquant_hf/<repo>/ per model:
- flipquant_{8x64,16x64,256x64}.pt: the release maps with the meta's local paths made relative, the data-extension code
  reference without its GitHub owner, and the reference implementation's branch and commit removed (the user's
  decisions); flipquant.maps.save, tiles bit-identical (torch.equal per module and every tensor record byte-identical),
  the release directory's own files untouched;
- README.md: the final model card (report.py's draft, finalized: front matter, no draft banner or placeholders, the code
  statement, the license, the corrected window counts, the uploaded files' sha256 and their provenance);
- ppl_summary.json: PPL per unit with BF16 / FourOverSix, the paired ΔNLL ± 2 SE against FourOverSix, the settings;
- license files: Qwen3 LICENSE (the base snapshot's); Llama LICENSE + USE_POLICY.md (the base repository's own files,
  downloaded read-only into /home/dev/flipquant_hf_licenses/llama) + NOTICE; Nemotron LICENSE (the NVIDIA Open Model
  License Agreement, verbatim from the page the base card links, rendered by /home/dev/flipquant_hf_licenses/nemotron)
  + NOTICE (the attribution); the other models: the card's license field only (their repositories hold no LICENSE).
Then every staged file is scanned (local paths, host, user names, the private repos, tokens); stage.json records the
files, the sha256 before -> after and the scan.

    python hf_stage.py
"""
import copy
import hashlib
import json
import re
import socket
import sys
import zipfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as RP  # noqa: E402
from release import REL, UNITS, W  # noqa: E402

sys.path.insert(0, str(W))
from flipquant import maps as M  # noqa: E402

OUT = Path("/home/dev/flipquant_hf")
HUB = Path("/home/dev/.cache/huggingface/hub")
REPOS = {"qwen3-1.7b": "Qwen3-1.7B-FlipQuant", "qwen3-8b": "Qwen3-8B-FlipQuant",
         "mistral-7b": "Mistral-7B-Instruct-v0.3-FlipQuant", "mistral-7b-base": "Mistral-7B-v0.3-FlipQuant",
         "llama3.1-8b": "Llama-3.1-8B-FlipQuant", "nemotron-nano-9b-v2": "NVIDIA-Nemotron-Nano-9B-v2-FlipQuant",
         "phi4-14b": "phi-4-FlipQuant", "qwen3.8-27b": "Qwen3.8-27B-FlipQuant"}
NAMESPACE = "edgeai-lab"
LOCAL_PREFIX = "/home/dev/n16k64_campaign/fqrel/"
OWNER_PREFIX = "brian030128/flipquant "
SCAN = re.compile("|".join([r"/home", r"/tmp", r"/root", r"/mnt", r"/scratch", r"n16k64_campaign", r"fqrel", r"fqopt",
                            r"razer", r"brian030128", r"chenjiaj", r"u4320956", r"nycu", r"hf_[A-Za-z0-9]{20,}",
                            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}", re.escape(socket.gethostname()),
                            r"\bgputw\b", r"topk-cal"]), re.I)
NVIDIA_NOTICE = "Licensed by NVIDIA Corporation under the NVIDIA Open Model License\n"
LICENSES = Path("/home/dev/flipquant_hf_licenses")
NVIDIA_URL = "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/"
NVIDIA_RETRIEVED = "2026-10-07 05:26 UTC"
NVIDIA_PAGE_SHA = "624c40d8cc6da3e6636a7a05bf34cf6e925a4f9f23666f7095468865917013f2"
CODE = "The FlipQuant code will be released with the paper."
TAGS = ["quantization", "fp4", "nvfp4", "mixfp4", "flipquant"]
DATASETS = {"wiki": "Salesforce/wikitext, wikitext-2-raw-v1, test split",
            "c4": "allenai/c4, validation shard en/c4-validation.00000-of-00008.json.gz"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_records(path):
    z = zipfile.ZipFile(path)
    return {n.split("/", 1)[1]: z.read(n) for n in z.namelist() if "/data/" in n}


def walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(x, str):
        yield path, x


def neutral_meta(meta):
    """The meta with its local paths relative to the release work directory, and the extension code's reference without
    its GitHub owner; returns (meta, [(field, old, new)])."""
    m, changes = copy.deepcopy(meta), []
    for k in ("run", "calibration_record"):
        v = m["source"].get(k)
        if isinstance(v, str) and v.startswith("/"):
            assert v.startswith(LOCAL_PREFIX), v
            m["source"][k] = v[len(LOCAL_PREFIX):]
            changes.append((f"source.{k}", v, m["source"][k]))
    v = m.get("fit_extension", {}).get("flipquant_reference")
    if isinstance(v, str) and v.startswith(OWNER_PREFIX):
        m["fit_extension"]["flipquant_reference"] = "flipquant " + v[len(OWNER_PREFIX):]
        changes.append(("fit_extension.flipquant_reference", v, m["fit_extension"]["flipquant_reference"]))
    for k in ("branch", "commit"):                      # the reference implementation's (the user's decision (b))
        if k in m:
            changes.append((k, m.pop(k), None))
    assert m["flipquant"]["commit"] and m["repo"] == "the reference implementation" and "trainer" in m
    return m, changes


def stage_map(model, unit, dst):
    src = REL / model / f"flipquant_{unit}.pt"
    tiles, u, meta = M.load(src)
    new_meta, changes = neutral_meta(meta)
    out = dst / src.name
    M.save(out, tiles, u, new_meta)
    t2, u2, m2 = M.load(out)
    assert u2 == u and set(t2) == set(tiles)
    assert all(t2[n].dtype == tiles[n].dtype and t2[n].shape == tiles[n].shape and torch.equal(t2[n], tiles[n])
               for n in tiles), "tiles differ"
    a, b = tensor_records(src), tensor_records(out)
    assert set(a) == set(b) and all(a[k] == b[k] for k in a), "tensor records differ"
    assert m2 == new_meta
    return dict(file=src.name, unit=unit, sha256_release=sha(src), sha256_uploaded=sha(out), meta_changes=changes,
                modules=len(tiles), tiles=sum(t.numel() for t in tiles.values()),
                e0m3_tiles=int(sum(int(t.sum()) for t in tiles.values())), tiles_bit_identical=True,
                tensor_records_identical=True)


def llama_license(dst, meta):
    """The base repository's own LICENSE and USE_POLICY.md (meta-llama/Llama-3.1-8B at the calibration revision),
    downloaded read-only on 2026-10-07 into LICENSES/llama (the local snapshot holds neither), and the NOTICE."""
    src = LICENSES / "llama"
    agreement = (src / "LICENSE").read_text()
    assert agreement.startswith("LLAMA 3.1 COMMUNITY LICENSE AGREEMENT")
    for s in ("Llama 3.1 Version Release Date: July 23, 2024", "1. License Rights and Redistribution",
              "Built with Llama", "2. Additional Commercial Terms", "3. Disclaimer of Warranty", "4. Limitation of Liability",
              "5. Intellectual Property", "6. Term and Termination", "7. Governing Law and Jurisdiction"):
        assert s in re.sub(r"\s+", " ", agreement), s      # the official file is hard-wrapped
    for f in ("LICENSE", "USE_POLICY.md"):
        (dst / f).write_bytes((src / f).read_bytes())
    (dst / "NOTICE").write_bytes((REL / "llama3.1-8b" / "NOTICE").read_bytes())
    return dict(LICENSE=f"meta-llama/Llama-3.1-8B@{meta['revision']}: LICENSE (downloaded 2026-10-07)",
                **{"USE_POLICY.md": f"meta-llama/Llama-3.1-8B@{meta['revision']}: USE_POLICY.md (downloaded 2026-10-07)"},
                NOTICE="the release directory's NOTICE (as drafted)")


def nvidia_license(dst):
    """The NVIDIA Open Model License Agreement, verbatim from the page the base model card links (the repository has no
    LICENSE file), rendered to text with the section numbers the page displays; and the attribution NOTICE."""
    text = (LICENSES / "nemotron" / "LICENSE").read_text()
    assert text.startswith("NVIDIA Open Model License Agreement") and "Version Release Date: October 24, 2025" in text
    assert "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" in text
    (dst / "LICENSE").write_text(text)
    (dst / "NOTICE").write_text(NVIDIA_NOTICE)
    return dict(LICENSE=f"{NVIDIA_URL} (retrieved {NVIDIA_RETRIEVED}; page sha256 {NVIDIA_PAGE_SHA}; Last Modified: "
                        "October 24, 2025), the agreement's text with the page's displayed section numbers",
                NOTICE="the NVIDIA Open Model License's attribution notice (section 3.1)")


def snapshot_license(dst, meta):
    lic = HUB / ("models--" + meta["model_id"].replace("/", "--")) / "snapshots" / meta["revision"] / "LICENSE"
    if lic.exists():
        (dst / "LICENSE").write_bytes(lic.read_bytes())
        return dict(LICENSE=f"{meta['model_id']}@{meta['revision']} LICENSE (the base snapshot's file)")
    return {}


def windows(s):
    r = s["ppl"]["fo6"]["results"]
    return {c: (r[c]["windows"], r[c]["data"]["window_tokens"]) for c, _ in RP.CORPORA}


MEASURE_NOTE = ("Time and memory: the trainer's own phase timer and peak counters give the phases, the epochs, the GPU "
                "peaks, its `ru_maxrss`, the RSS after the model load and the teacher storage (the release run). The "
                "wrapper's RSS, the process-tree peak (wrapper + trainer, sampled at 10 Hz) and the end-to-end time come "
                "from an external process sampler: for the runs marked ᴿ (source “re-run”) from a measured re-run of "
                "the same calibration that reproduced the same map, otherwise from a sampler attached to the run itself "
                "(“live”).{late} The end-to-end time excludes the one-time data preparation, which has its own row.")
LATE = (" Nemotron 16x64's sampler was attached 13 min after the start, so its tree peak covers only the rest, while "
        "its VmHWM peaks are exact.")
PREP_NOTE = ("train_map's first run of a model on a new data root prepares its calibration record and development sets "
             "(`calibration.tmopt_common.prepare` and `prepare_development`: the builder rule's checks, re-tokenizing the "
             "fit windows, hashing the model's weights on the CPU unless an archived record has them, the development "
             "draws). Measured on 2026-10-06 in a fresh data root with the same sampler plus each process's VmHWM; the "
             "files it wrote are byte-identical to the ones the release runs used. The dataset shards were already in "
             "the local Hugging Face cache; a first-ever preparation also downloads them.")
LLAMA_NOTICE = "Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved."


def card(s, staged, license_files):
    model, info = s["model"], s["info"]
    meta = M.load(REL / model / f"flipquant_{next(iter(s['runs']))}.pt")[2]
    hf_id, revision, commit = meta["model_id"], meta["revision"], meta["flipquant"]["commit"]
    rep = RP.load(REL / model / "records" / next(iter(s["runs"])) / "reproduction.json")
    settings = rep["settings"]
    front = {"license": info["license"], **info.get("license_extra", {}), "base_model": hf_id, "tags": TAGS}
    lines = ["---", yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=1000).strip(), "---", ""]
    if model == "llama3.1-8b":
        lines += [f"# {info['name']} FlipQuant MixFP4 maps", "", "**Built with Llama.**", "",
                  f"{LLAMA_NOTICE} The agreement is in `LICENSE`, the Acceptable Use Policy it incorporates in "
                  "`USE_POLICY.md`, and the notice in `NOTICE`.", ""]
    else:
        lines += [f"# FlipQuant MixFP4 maps for {info['name']}", ""]
    lines += ["## What this is", "",
              f"Format maps for [{hf_id}](https://huggingface.co/{hf_id}) at revision `{revision}`, calibrated with "
              "FlipQuant (TM-OPT+TC). A map holds one bit per weight tile of every quantized linear layer (all "
              "text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform levels "
              "0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale and "
              "a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 checkpoint "
              "when it is loaded.", "",
              "| file | tile (output rows × input columns) | E0M3 tiles | sha256 |", "|---|---|---:|---|"]
    for u, r in s["runs"].items():
        st = staged[u]
        assert (st["e0m3_tiles"], st["tiles"]) == (r["e0m3_tiles"], r["tiles"])
        lines.append(f"| `flipquant_{u}.pt` | {u.replace('x', ' × ')} | {r['e0m3_tiles']:,} of {r['tiles']:,} "
                     f"({100 * r['e0m3_tiles'] / r['tiles']:.2f} %) | `{st['sha256_uploaded']}` |")
    batch = f"{settings['batch']} sequences per step" if settings["accum"] == 1 else \
        f"micro-batch {settings['batch']} × accumulation {settings['accum']} (8 sequences per step)"
    lines += ["", "8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on "
              "sm_120); 16x64 and 256x64 are coarser.", "", CODE, "",
              "## Calibration", "",
              f"`python -m calibration.train_map --model {model} --unit <unit> --fit-windows 256 --epochs 5 "
              f"--teacher-topk 1000` (FlipQuant commit `{commit}`):", "",
              "- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per "
              "position plus one tail bucket, per-token FourOverSix activations.",
              f"- Adam, learning rate 0.02, initial logit −1 (every tile starts as E2M1), seed 0, {batch}, 5 epochs, "
              "deterministic.",
              "- Fit set: 256 windows of 512 tokens. They are the FlipQuant paper's 128 math/code windows (OpenWebMath @ "
              "fde8ef8, CodeParrot-clean @ 35a59fb), plus 128 more by the extension rule, which skips the paper's fit, "
              "development and published C4 evaluation documents. The calibration records every window by document "
              "hash, offset and token hash.",
              "- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' "
              "`meta` records them as `non_default_settings`.", ""]
    rows = [(model, u, RP.unit_metrics(model, u)) for u in s["runs"]]
    lines += ["Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:", "", *RP.time_table(rows), "",
              *RP.memory_table(rows), ""]
    if RP.prep_metrics(model):
        lines += ["The one-time data preparation (fit and development records, before the first unit):", "", PREP_NOTE,
                  "", *RP.prep_table([model]), ""]
    lines += [MEASURE_NOTE.format(late=LATE if model == "nemotron-nano-9b-v2" else ""), ""]
    win = windows(s)
    lines += ["## Perplexity", "",
              f"Native sm_120 kernels, FlipQuant's `--paper-convention` (per-token FourOverSix activations), "
              f"WikiText-2 ({win['wiki'][0]} windows of {win['wiki'][1]} tokens) and C4 ({win['c4'][0]} windows of "
              f"{win['c4'][1]} tokens). ΔNLL is the mean paired per-window difference in nats per token against "
              "FourOverSix; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers "
              "are also in `ppl_summary.json`.", "", RP.ppl_table(s), ""]
    if s.get("reference") and s["reference"]["maps"]:
        lines += ["The paper-setting rows are maps calibrated at the paper's settings (20 epochs, 128 windows, the full "
                  "teacher), evaluated under the same convention; they are listed for comparison only.", ""]
    cost = RP.ppl_costs().get((model, "flipquant_16x64"))
    lines += ["## How to use", "", CODE + " With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the "
              "kernels once, then evaluate with the native kernels:", "", "```bash",
              "bash kernels/sm120/build_deploy.sh",
              f"python -m evaluation.ppl --model {model} --mode native --weight mixfp4 --map flipquant_16x64.pt "
              "--paper-convention", "```", "",
              *([f"Measured here with the 16x64 map: {cost[2] / 1024:.0f} GiB of GPU memory at peak (2048-token windows, "
                 f"batch 1), {RP.fmt_s(cost[1])} for both corpora, model load included.", ""] if cost else []),
              *(["On other GPUs, the same map with simulated (fake) quantization:", "", "```bash",
                 f"python -m evaluation.ppl --model {model} --mode fake --weight mixfp4 --map flipquant_16x64.pt "
                 "--paper-convention", "```", "",
                 "The native command produced the table above; the simulated one was smoke-tested on two windows per "
                 "corpus."] if not s.get("smoke_oom") else
                ["The native command produced the table above. The simulated (fake) path does not fit this model on one "
                 "96 GB GPU: it builds both candidate encodings of every weight on the GPU, next to the BF16 model, and "
                 "ran out of memory while doing so (smoke test, 2026-10-06). Use the native kernels, or a GPU with more "
                 "memory."]), ""]
    lines += ["## License", "",
              f"These maps are released under the same license as the base model, the {info['license_text']}."]
    if model == "llama3.1-8b":
        lines += ["", RP.LLAMA_TERMS, "",
                  f"- `LICENSE`: the Llama 3.1 Community License Agreement, the base repository's own `LICENSE` file "
                  f"({hf_id} at revision `{revision}`).",
                  "- `USE_POLICY.md`: the Llama 3.1 Acceptable Use Policy, which the agreement incorporates; the base "
                  "repository's own `USE_POLICY.md`.",
                  f"- `NOTICE`: “{LLAMA_NOTICE}”"]
    elif model == "nemotron-nano-9b-v2":
        lines += ["", f"Governing terms: the [NVIDIA Open Model License Agreement]({info['license_extra']['license_link']}).",
                  "",
                  f"- `LICENSE`: the NVIDIA Open Model License Agreement (Last Modified: October 24, 2025), verbatim from "
                  f"{NVIDIA_URL}, the page the base model card links (the base repository has no license file); retrieved "
                  f"{NVIDIA_RETRIEVED}. The section numbers are the ones the page displays.",
                  "- `NOTICE`: the attribution notice “Licensed by NVIDIA Corporation under the NVIDIA Open Model "
                  "License” (section 3.1)."]
    elif "LICENSE" in license_files:
        lines += ["", "`LICENSE` is a copy of the base model's license file."]
    lines += ["", "## Files", "",
              "- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; "
              "`flipquant.maps.load` reads them).",
              "- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix and the evaluation "
              "settings."]
    lines += [f"- `{f}`: see License." for f in ("LICENSE", "USE_POLICY.md", "NOTICE") if f in license_files]
    lines += ["", "**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the "
              "run directory and calibration-record paths are relative instead of local absolute paths, the "
              "data-extension code reference names the file without its repository owner, and the internal branch and "
              "commit of the reference implementation are removed (the FlipQuant commit is kept). The tiles are "
              "bit-identical. sha256 of the calibration output → this repository's file:", ""]
    lines += [f"- `flipquant_{u}.pt`: `{staged[u]['sha256_release']}` → `{staged[u]['sha256_uploaded']}`" for u in s["runs"]]
    lines += [""]
    return "\n".join(lines)


def ppl_summary(s, staged):
    sig, meta = s["significance"], M.load(REL / s["model"] / f"flipquant_{next(iter(s['runs']))}.pt")[2]
    fo6 = s["ppl"]["fo6"]
    labels = ["bf16", "fo6"] + [f"flipquant_{u}" for u in UNITS if f"flipquant_{u}" in s["ppl"]]
    out = dict(model=meta["model_id"], revision=meta["revision"], flipquant_model_key=s["model"],
               evaluation=dict(mode="native sm_120 kernels (FlipQuant evaluation.ppl --mode native)",
                               convention=fo6["convention"], quantized_modules=fo6["quantized_modules"],
                               transformers=fo6["source"]["transformers"], torch=fo6["source"]["torch"],
                               gpu=meta["gpu"],
                               corpora={c: {"dataset": DATASETS[c]} | {k: v for k, v in fo6["results"][c]["data"].items()
                                                                        if k != "documents"} for c, _ in RP.CORPORA},
                               c4_documents="the documents are drawn with the recorded seed from the listed shard"),
               ppl={l: {c: s["ppl"][l]["results"][c]["ppl"] for c, _ in RP.CORPORA} for l in labels},
               paired_vs_fourover6={}, maps={})
    for l in labels:
        if l == "fo6":
            continue
        out["paired_vs_fourover6"][l] = {c: dict(delta_nll=p["delta"], two_se=2 * p["se"], n=p["n"],
                                                 significant=p["significant"], verdict=RP.verdict(p))
                                         for c, p in sig["paired"][l].items()}
    for u, st in staged.items():
        out["maps"][f"flipquant_{u}.pt"] = dict(sha256=st["sha256_uploaded"], e0m3_tiles=st["e0m3_tiles"], tiles=st["tiles"])
    ref = s.get("reference")
    if ref and ref["maps"]:
        out["paper_setting_maps_ppl"] = {u: r["ppl"] for u, r in ref["maps"].items()}
        out["paper_setting_note"] = ("maps calibrated at the paper's settings (20 epochs, 128 windows, full-vocabulary "
                                     "teacher), evaluated under the same convention; for comparison only")
    return out


def scan(dst):
    hits = []
    for f in sorted(dst.iterdir()):
        if f.suffix == ".pt":
            obj = torch.load(f, map_location="cpu", weights_only=False)
            strings = list(walk(obj["meta"], "meta")) + [(f"tiles key", k) for k in obj["tiles"]] + \
                [("format", obj["format"]), ("unit", obj["unit"])]
            raw = f.read_bytes()
            strings += [("raw bytes", m.group(0).decode(errors="replace"))
                        for m in re.finditer(rb"(/home/|/tmp/|brian030128|razer|hf_[A-Za-z0-9]{20,})", raw, re.I)]
        else:
            strings = [(f"line {i + 1}", line) for i, line in enumerate(f.read_text().splitlines())]
        for where, sv in strings:
            for m in SCAN.finditer(sv):
                hits.append(dict(file=f.name, where=where, match=m.group(0), text=sv[:200]))
    return hits


def main():
    OUT.mkdir(exist_ok=True)
    record = dict(namespace=NAMESPACE, repos={})
    for model, repo in REPOS.items():
        dst = OUT / repo
        dst.mkdir(exist_ok=True)
        s = RP.model_summary(model)
        staged = {u: stage_map(model, u, dst) for u in s["runs"]}
        meta = M.load(REL / model / f"flipquant_{next(iter(s['runs']))}.pt")[2]
        if model == "llama3.1-8b":
            lic = llama_license(dst, meta)
        elif model == "nemotron-nano-9b-v2":
            lic = nvidia_license(dst)
        else:
            lic = snapshot_license(dst, meta)
        (dst / "ppl_summary.json").write_text(json.dumps(ppl_summary(s, staged), indent=1, ensure_ascii=False) + "\n")
        (dst / "README.md").write_text(card(s, staged, lic))
        files = {f.name: dict(sha256=sha(f), bytes=f.stat().st_size) for f in sorted(dst.iterdir())}
        record["repos"][f"{NAMESPACE}/{repo}"] = dict(model=model, base_model=meta["model_id"], revision=meta["revision"],
                                                      maps=staged, license_files=lic, files=files, scan=scan(dst))
        print(f"{NAMESPACE}/{repo}: {len(files)} files, scan hits {len(record['repos'][f'{NAMESPACE}/{repo}']['scan'])}")
    (OUT / "stage.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

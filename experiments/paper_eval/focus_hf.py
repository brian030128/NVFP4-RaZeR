"""The FOCUS states on the Hugging Face Hub (the user's request of 2026-10-09, relayed by the coordinator): five PRIVATE
repositories under edgeai-lab, one per model with a FOCUS state (part K; Qwen3.8-27B has none).

Staging (/home/dev/focus_hf/<repo>/, outside git; `stage` uploads nothing):
- focus.pt: part K's state as is (a reflink copy; sha256 equal to the source). Scanned: its metadata (config, source, the
  layer and field names), the archive's record names and its raw bytes. The metadata is rewritten only if a scan finds
  something (then every tensor is checked bit-identical and both sha256 are recorded).
- README.md: the model card.
- ppl_summary.json: BF16 / NVFP4 / FourOverSix / FOCUS PPL, the paired ΔNLL ± 2 SE against FourOverSix, the evaluation
  settings, the calibration settings and cost; every number from the evaluated records (final.json's sources) and the
  calibration's report.json.
- license files: the Qwen3 LICENSE and the Nemotron LICENSE + NOTICE as staged for the FlipQuant repositories
  (/home/dev/flipquant_hf); phi-4's LICENSE (the base repository's file at the pinned revision); for
  Mistral-7B-Instruct-v0.3, whose repository has no license file, the Apache License 2.0 text its card declares
  (`licenses` fetches both, read-only and anonymously, into /home/dev/focus_hf_licenses).
Every staged file is scanned with the FlipQuant upload's patterns (fqrel/hf_stage.py), plus a few for pointers to a
reference implementation; stage.json records the files, their sha256, the scan hits and the metadata changes.

    python focus_hf.py licenses    # network (anonymous, read-only): phi-4 LICENSE, the Apache License 2.0 text
    python focus_hf.py stage       # offline: /home/dev/focus_hf/<repo>/ and stage.json
    python focus_hf.py check       # read-only: the active account, and that no target repository exists
    python focus_hf.py upload      # create PRIVATE repos, upload_large_folder (resumable), verify; hf_upload.json
    python focus_hf.py record      # results/paper_eval/focus_hf/{HF_UPLOAD.md, stage.json, hf_upload.json}
"""
import datetime
import hashlib
import json
import math
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables_cpu as T  # noqa: E402

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
STAGE = Path("/home/dev/focus_hf")
LIC = Path("/home/dev/focus_hf_licenses")
OLD = Path("/home/dev/flipquant_hf")
OUT = T.ROOT / "results" / "paper_eval" / "focus_hf"
NAMESPACE = "edgeai-lab"
REPOS = {"qwen3-1.7b": "Qwen3-1.7B-FOCUS-NVFP4", "qwen3-8b": "Qwen3-8B-FOCUS-NVFP4",
         "mistral-7b": "Mistral-7B-Instruct-v0.3-FOCUS-NVFP4", "nemotron-nano-9b-v2": "NVIDIA-Nemotron-Nano-9B-v2-FOCUS-NVFP4",
         "phi4-14b": "phi-4-FOCUS-NVFP4"}
TITLES = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B", "mistral-7b": "Mistral-7B-Instruct-v0.3",
          "nemotron-nano-9b-v2": "NVIDIA-Nemotron-Nano-9B-v2", "phi4-14b": "phi-4"}
PHI4 = ("microsoft/phi-4", "2db69c1c3e91a05d2c64a3185acfbaf36f744e25")
APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
NVIDIA_URL = "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/"
NVIDIA_RETRIEVED = "2026-10-07 05:26 UTC"
LICENSE = {
    "qwen3-1.7b": dict(front={"license": "apache-2.0"}, name="Apache License 2.0", files=dict(LICENSE=OLD / "Qwen3-1.7B-FlipQuant" / "LICENSE")),
    "qwen3-8b": dict(front={"license": "apache-2.0"}, name="Apache License 2.0", files=dict(LICENSE=OLD / "Qwen3-8B-FlipQuant" / "LICENSE")),
    "mistral-7b": dict(front={"license": "apache-2.0"}, name="Apache License 2.0", files=dict(LICENSE=LIC / "mistral" / "LICENSE")),
    "nemotron-nano-9b-v2": dict(front={"license": "other", "license_name": "nvidia-open-model-license",
                                       "license_link": NVIDIA_URL},
                                name="NVIDIA Open Model License Agreement",
                                files=dict(LICENSE=OLD / "NVIDIA-Nemotron-Nano-9B-v2-FlipQuant" / "LICENSE",
                                           NOTICE=OLD / "NVIDIA-Nemotron-Nano-9B-v2-FlipQuant" / "NOTICE")),
    "phi4-14b": dict(front={"license": "mit"}, name="MIT License", files=dict(LICENSE=LIC / "phi4" / "LICENSE")),
}
TAGS = ["quantization", "fp4", "nvfp4", "focus"]
CODE = "The code (our FOCUS implementation and the evaluation) will be released with the paper."
DATASETS = {"wiki": "Salesforce/wikitext, wikitext-2-raw-v1, test split",
            "c4": "allenai/c4, validation shard en/c4-validation.00000-of-00008.json.gz"}
# the FlipQuant upload's patterns (fqrel/hf_stage.py SCAN), plus pointers to a reference implementation
SCAN = re.compile("|".join([r"/home", r"/tmp", r"/root", r"/mnt", r"/scratch", r"n16k64_campaign", r"fqrel", r"fqopt",
                            r"razer", r"brian030128", r"chenjiaj", r"u4320956", r"nycu", r"hf_[A-Za-z0-9]{20,}",
                            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}", re.escape(socket.gethostname()),
                            r"\bgputw\b", r"topk-cal"]), re.I)
SCAN_REF = re.compile(r"angelslim|tencent|github|gitlab|\bbranch\b|\bcommit\b", re.I)
RAW = re.compile(rb"(/home/|/tmp/|brian030128|razer|hf_[A-Za-z0-9]{20,}|chenjiaj|u4320956|n16k64_campaign|angelslim)",
                 re.I)


def sha(path, bufsize=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            p = f"{path}.{k}" if path else str(k)
            yield f"{p} (key)", str(k)
            yield from walk(v, p)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(x, str):
        yield path, x


def raw_hits(path, bufsize=64 << 20, overlap=96):
    """RAW matches over the whole file, streamed with an overlap; {offset: match}."""
    hits, prev, pos = {}, b"", 0
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            buf = prev + chunk
            for m in RAW.finditer(buf):
                hits[pos - len(prev) + m.start()] = m.group(0).decode(errors="replace")
            prev, pos = buf[-overlap:], pos + len(chunk)
    return hits


def scan_text(name, text, ref=True):
    """SCAN on every line; SCAN_REF too on our own files (not on the third-party license texts)."""
    hits = []
    for i, line in enumerate(text.splitlines()):
        for rx in ((SCAN, SCAN_REF) if ref else (SCAN,)):
            hits += [dict(file=name, where=f"line {i + 1}", match=m.group(0), text=line[:200]) for m in rx.finditer(line)]
    return hits


def scan_state(path):
    """The state's metadata strings, its field names, the archive's record names and its raw bytes."""
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert set(obj) == {"config", "layers", "source"}, sorted(obj)
    fields = {k for v in obj["layers"].values() for k in v}
    assert fields == {"m", "q", "s2", "act_gscale"}, fields
    assert all(v["act_gscale"] is None for v in obj["layers"].values())
    strings = list(walk(obj["config"], "config")) + list(walk(obj["source"], "source"))
    strings += [("layers (key)", k) for k in obj["layers"]] + [("layer field", k) for k in sorted(fields)]
    with zipfile.ZipFile(path) as z:
        records = z.namelist()
    strings += [("archive record", n) for n in records]
    hits = []
    for where, s in strings:
        for rx in (SCAN, SCAN_REF):
            hits += [dict(file=path.name, where=where, match=m.group(0), text=s[:200]) for m in rx.finditer(s)]
    raw = raw_hits(path)
    hits += [dict(file=path.name, where=f"raw bytes @{off}", match=m, text="") for off, m in sorted(raw.items())]
    info = dict(config=obj["config"], source=obj["source"], modules=len(obj["layers"]),
                block_factors_m=sum(v["m"].numel() for v in obj["layers"].values()),
                subblock_logits_q=sum(v["q"].numel() for v in obj["layers"].values()),
                tensor_scales_s2=sum(v["s2"].numel() for v in obj["layers"].values()),
                dtypes=sorted({str(v[f].dtype) for v in obj["layers"].values() for f in ("m", "q", "s2")}),
                first_layer=next(iter(obj["layers"])), archive_records=len(records),
                archive_prefix=sorted({n.split("/", 1)[0] for n in records}),
                metadata_strings_scanned=len(strings))
    return hits, info


# ---------------------------------------------------------------------------------------------------------- licenses
def licenses():
    """phi-4's LICENSE at the pinned revision and the Apache License 2.0 text, read-only and without a token."""
    from huggingface_hub import hf_hub_download
    rec = {}
    (LIC / "phi4").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        p = hf_hub_download(PHI4[0], "LICENSE", revision=PHI4[1], cache_dir=tmp, token=False)
        text = Path(p).read_text()
    assert "MIT License" in text and "Copyright (c) Microsoft Corporation." in text and "Permission is hereby granted" in text, text[:200]
    (LIC / "phi4" / "LICENSE").write_text(text)
    when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rec["phi4"] = dict(source=f"{PHI4[0]}@{PHI4[1]}: LICENSE", retrieved=when, sha256=sha(LIC / "phi4" / "LICENSE"))
    (LIC / "mistral").mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(APACHE_URL, timeout=60) as r:
        body = r.read()
    text = body.decode("utf-8")
    assert text.lstrip().startswith("Apache License") and "Version 2.0, January 2004" in text
    assert "END OF TERMS AND CONDITIONS" in text and "APPENDIX: How to apply the Apache License to your work." in text
    (LIC / "mistral" / "LICENSE").write_bytes(body)
    rec["mistral"] = dict(source=APACHE_URL, retrieved=when, sha256=sha(LIC / "mistral" / "LICENSE"))
    (LIC / "licenses.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


# ------------------------------------------------------------------------------------------------------------ records
def evaluated(model):
    """PPL and paired ΔNLL vs FourOverSix of BF16 / NVFP4 / FourOverSix / FOCUS, recomputed from the records final.json
    names, and checked equal to final.json."""
    final = json.loads((OUT.parent / "final" / "final.json").read_text())
    src = {r: final["sources"][model][r] for r in ("bf16", "nvfp4", "fo6", "focus")}
    nll = {r: T.nll_of(p) for r, p in src.items()}
    out = dict(ppl={}, paired={})
    for r, v in nll.items():
        out["ppl"][r] = {c: math.exp(T.mean(v[c])) for c, _ in T.CORPORA}
        assert all(out["ppl"][r][c] == final["ppl"][model][r][c] for c, _ in T.CORPORA), (model, r)
        if r != "fo6":
            out["paired"][r] = {c: T.paired(v[c], nll["fo6"][c]) for c, _ in T.CORPORA}
            for c, _ in T.CORPORA:
                assert out["paired"][r][c]["delta"] == final["vs_fo6"][model][r][c]["delta"], (model, r, c)
    rec = json.loads(Path(src["focus"]).read_text())
    assert rec["mode"] == "native" and rec["focus"]["deploy"] is True and rec["quant"]["weight"] == "focus"
    assert (rec["quant"]["act"], rec["quant"]["act_scope"]) == ("fourover6", "row")
    assert rec["native"]["weight"] == "nvfp4" and rec["native"]["e0m3_tiles"] == 0
    out["record"] = rec
    out["windows"] = {c: rec["results"][c]["windows"] for c, _ in T.CORPORA}
    assert all(len(nll[r][c]) == out["windows"][c] for r in nll for c, _ in T.CORPORA)
    return out


def verdict(p):
    return "inconclusive" if not p["significant"] else ("better" if p["delta"] < 0 else "worse")


def cost(model):
    r = json.loads((RUN / "focus" / model / "state" / "report.json").read_text())
    assert r["status"] == "complete" and r["steps"] == 8 and len(r["log"]) == 8
    return dict(steps=r["steps"], micro_batch=int(r["args"]["micro"]), training_seconds=r["training_seconds"],
                end_to_end_seconds=r["resources"]["seconds"], gpu=r["resources"]["gpus"][0],
                gpu_peak_allocated_gib=r["resources"]["gpu_peak_gib"][0], parameters=r["parameters"],
                checkpointed_layers=r["checkpointed_layers"], data_windows=r["data"]["windows"],
                data_tokens=r["data"]["tokens"], kl_top_first_last=(r["log"][0]["kl_top"], r["log"][-1]["kl_top"]))


def fmt_s(s):
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min"


# ------------------------------------------------------------------------------------------------------------- card
def card(model, info, ev, cs, files):
    hf_id, rev = info["source"]["hf_id"], info["source"]["revision"]
    lic = LICENSE[model]
    front = {**lic["front"], "base_model": hf_id, "tags": TAGS}
    title = TITLES[model]
    rec = ev["record"]
    wiki, c4 = rec["results"]["wiki"]["data"], rec["results"]["c4"]["data"]
    L = ["---", yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=1000).strip(), "---", "",
         f"# FOCUS NVFP4 scales for {title} (our reproduction)", "",
         "**This is our reproduction of FOCUS** ([“FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity "
         "Scaling”](https://arxiv.org/abs/2608.01847), arXiv:2608.01847), **made as a baseline for the FlipQuant paper. "
         "It is not an official release by the FOCUS authors.**", "",
         "## What this is", "",
         f"`focus.pt` holds the learned FOCUS scale parameters for [{hf_id}](https://huggingface.co/{hf_id}) at revision "
         f"`{rev}`, for every quantized linear layer ({info['modules']} layers: all text-model linear layers except the "
         "LM head). Per layer, in FP32:", "",
         "| field | count | role |", "|---|---|---|",
         f"| `m` | one per 16-element block ({info['block_factors_m']:,} in total) | the block-scale factor (Coupled-Relaxation "
         "Scaling); starts at 1 |",
         f"| `q` | one per 8-element sub-block, 2 per block ({info['subblock_logits_q']:,} in total) | the sub-block "
         "relaxation logit (Dual-Granularity Scaling); coefficient `sigmoid(q)`, starts at `q = 6` |",
         "| `s2` | one per tensor | the NVFP4 tensor scale `amax(abs(W)) / (6 · 448)`, frozen (it follows from the base "
         "weights) |", "",
         "`m` and `q` are in row-major order of the weight `[out_features, in_features]`, 16-element blocks along "
         "`in_features`. Each layer also has `act_gscale = None`: activations use one scale per token (below), so "
         "there is no static activation scale. The file also stores the calibration settings (`config`) and the base "
         "model's id and revision (`source`).", "",
         "**It is not a quantized checkpoint**: it holds no weights. Using it needs the BF16 base model at the pinned "
         "revision and the FOCUS implementation, which turns the base weights and these parameters into standard NVFP4 "
         "codes. For a block `b` of 16 weights with sub-blocks `k = 1, 2` of 8:", "",
         "```",
         "S_b   = e4m3( clamp( max|w_b| · m_b / (6 · s2), 2^-6, 448 ) )      # the block's E4M3 scale",
         "c_i   = E2M1( clamp( w_i / (S_b · sigmoid(q_k) · s2), -6, 6 ) )      # element i in sub-block k",
         "w'_i  = c_i · S_b · s2                                               # the deployed (dequantized) weight",
         "```", "",
         "`E2M1` rounds to the nearest FP4 value (ties to even); a block whose scaled maximum is 0 gets scale 1 before the "
         "clamp. `sigmoid(q)` only changes which code each element gets: the stored scales are `S_b` and `s2`, so the "
         "deployed model is plain NVFP4 (E2M1 elements, one E4M3 scale per 16 elements, one FP32 scale per tensor) and "
         "runs on standard NVFP4 kernels with no extra metadata or cost.", "",
         CODE, "",
         "## Calibration", "",
         "The hyperparameters follow the FOCUS paper's NVFP4 setting:", "",
         "- learning rate 5e-3 for the block-scale factors `m`, 1e-3 for the relaxation logits `q`;",
         "- 2 sub-blocks of 8 elements per 16-element block;",
         "- KL-Top loss with k = 1000 (the BF16 model's top-1000 tokens per position, both distributions renormalized "
         "over them);",
         "- AdamW (betas 0.9 / 0.999, weight decay 0) with a constant learning rate;",
         "- 1 epoch, global batch 32;",
         "- `q` initialized to 6 (and `m` to 1); seed 42 (the data order).", "",
         "Only the scale parameters train; the model weights are frozen. The teacher is the unquantized BF16 model.", "",
         "**Differences from the paper:**", "",
         f"- **Calibration data.** The FlipQuant release set: {cs['data_windows']} windows of 512 tokens of math and code "
         f"text ({cs['data_tokens']:,} tokens; OpenWebMath and CodeParrot-clean), the same fit set as the FlipQuant "
         f"release maps, so {cs['steps']} optimizer steps. The paper uses 1,248 WikiText-2 training samples of 2,048 "
         "tokens (39 steps).",
         "- **Activations**, during calibration and evaluation: FourOverSix with one scale per token, as in the FlipQuant "
         "paper's protocol (NVFP4 activations in which each 16-element block's scale maps the block maximum to 6 or to "
         "4, whichever has the lower error, and one FP32 scale per token). The paper quantizes activations to NVFP4 "
         "with a static per-tensor scale.", "",
         f"Cost on one {cs['gpu']} (sm_120), micro-batch {cs['micro_batch']} ({32 // cs['micro_batch']} forward/backward "
         "passes per step), "
         "activation checkpointing per decoder block:", "",
         "| training ({} steps) | end to end (model load, data, training, save) | peak GPU memory |".format(cs["steps"]),
         "|---:|---:|---:|",
         f"| {fmt_s(cs['training_seconds'])} | {fmt_s(cs['end_to_end_seconds'])} | "
         f"{cs['gpu_peak_allocated_gib']:.1f} GiB |", "",
         "The peak is PyTorch's peak allocated memory on the GPU (`torch.cuda.max_memory_allocated`). The training "
         "time includes the BF16 teacher's forward passes.", "",
         "## Perplexity", "",
         f"Native sm_120 kernels on the {cs['gpu']}, with FOCUS deployed as standard NVFP4 codes and per-token FourOverSix "
         f"activations. WikiText-2 test ({ev['windows']['wiki']} windows of {wiki['window_tokens']} tokens) and C4 "
         f"validation ({ev['windows']['c4']} windows of {c4['window_tokens']} tokens). Every row is evaluated on the same "
         "windows:", "",
         "- BF16: the unquantized model;",
         "- NVFP4: round-to-nearest NVFP4 weights, NVFP4 activations with one scale per token;",
         "- FourOverSix: round-to-nearest FourOverSix weights and per-token FourOverSix activations (the reference);",
         "- FOCUS: this state, with per-token FourOverSix activations.", "",
         "ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; ± 2 SE over windows; "
         "`*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in "
         "`ppl_summary.json`.", "",
         "| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict "
         "(WikiText-2 / C4) |", "|---|---:|---:|---:|---:|---|"]
    names = [("bf16", "BF16"), ("nvfp4", "NVFP4"), ("fo6", "FourOverSix (reference)"), ("focus", "FOCUS (this state)")]
    for r, name in names:
        p = ev["ppl"][r]
        if r == "fo6":
            d, v = ["—", "—"], "reference"
        else:
            q = ev["paired"][r]
            d = [f"{q[c]['delta']:+.4f} ± {q[c]['two_se']:.4f}{' *' if q[c]['significant'] else ''}" for c, _ in T.CORPORA]
            v = " / ".join(verdict(q[c]) for c, _ in T.CORPORA)
        L.append(f"| {name} | {p['wiki']:.4f} | {p['c4']:.4f} | {d[0]} | {d[1]} | {v} |")
    if ev["ppl"]["focus"]["wiki"] < ev["ppl"]["bf16"]["wiki"]:
        L += ["", "On this model FOCUS's WikiText-2 perplexity is below BF16's, while its C4 perplexity is above it; both "
              "are as measured on the same windows."]
    L += ["", "## How to read the file", "", "With PyTorch alone:", "", "```python", "import torch", "",
          'state = torch.load("focus.pt", map_location="cpu", weights_only=True)',
          'state["config"]   # the calibration settings',
          'state["source"]   # the base model id and revision',
          f'layer = state["layers"]["{info["first_layer"]}"]',
          'layer["m"], layer["q"], layer["s2"]   # FP32 tensors (see above)', "```", "",
          "## License", "",
          f"This state is released under the base model's license, the {lic['name']}."]
    if model == "nemotron-nano-9b-v2":
        L += ["", f"Governing terms: the [NVIDIA Open Model License Agreement]({NVIDIA_URL}).", "",
              f"- `LICENSE`: the NVIDIA Open Model License Agreement (Last Modified: October 24, 2025), verbatim from "
              f"{NVIDIA_URL}, the page the base model card links (the base repository has no license file); retrieved "
              f"{NVIDIA_RETRIEVED}. The section numbers are the ones the page displays.",
              "- `NOTICE`: the attribution notice “Licensed by NVIDIA Corporation under the NVIDIA Open Model License” "
              "(section 3.1)."]
    elif model == "mistral-7b":
        L += ["", "The base model card declares Apache-2.0, and the base repository has no license file; `LICENSE` is the "
              f"Apache License 2.0 text from {APACHE_URL}."]
    elif model == "phi4-14b":
        L += ["", f"`LICENSE` is a copy of the base repository's license file ({hf_id} at revision `{rev}`)."]
    else:
        L += ["", f"`LICENSE` is a copy of the base model's license file ({hf_id} at revision `{rev}`)."]
    L += ["", "## Files", "", "| file | bytes | sha256 |", "|---|---:|---|"]
    L += [f"| `{n}` | {d['bytes']:,} | `{d['sha256']}` |" for n, d in files.items()]
    L += ["", "- `focus.pt`: the FOCUS state (`{config, layers, source}`; above). It is the calibration output, "
          "unchanged.",
          "- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings "
          "and the calibration's settings and cost."]
    L += [f"- `{f}`: see License." for f in ("LICENSE", "NOTICE") if f in lic["files"]]
    L += [""]
    return "\n".join(L)


def ppl_summary(model, info, ev, cs, state_sha, state_bytes):
    rec = ev["record"]
    out = dict(model=info["source"]["hf_id"], revision=info["source"]["revision"], model_key=info["config"]["model"],
               method="FOCUS (arXiv:2608.01847), our reproduction; not an official release by the FOCUS authors",
               evaluation=dict(mode="native sm_120 kernels; FOCUS deployed as standard NVFP4 codes",
                               convention=rec["convention"], quantized_modules=rec["quantized_modules"],
                               transformers=rec["source"]["transformers"], torch=rec["source"]["torch"],
                               gpu=rec["resources"]["gpus"][0],
                               corpora={c: {"dataset": DATASETS[c]} | {k: v for k, v in rec["results"][c]["data"].items()
                                                                        if k != "documents"} for c, _ in T.CORPORA},
                               c4_documents="the documents are drawn with the recorded seed from the listed shard",
                               rows=dict(bf16="the unquantized model",
                                         nvfp4="round-to-nearest NVFP4 weights, NVFP4 activations with one scale per token",
                                         fo6="round-to-nearest FourOverSix weights, per-token FourOverSix activations",
                                         focus="this state deployed as NVFP4 codes, per-token FourOverSix activations")),
               ppl={r: ev["ppl"][r] for r in ("bf16", "nvfp4", "fo6", "focus")},
               paired_vs_fourover6={r: {c: dict(delta_nll=p["delta"], two_se=p["two_se"], n=p["n"],
                                                significant=p["significant"], verdict=verdict(p))
                                        for c, p in ev["paired"][r].items()} for r in ("bf16", "nvfp4", "focus")},
               state={"focus.pt": dict(sha256=state_sha, bytes=state_bytes, modules=info["modules"],
                                       block_factors_m=info["block_factors_m"], subblock_logits_q=info["subblock_logits_q"],
                                       tensor_scales_s2=info["tensor_scales_s2"], dtype="float32")},
               calibration=dict(info["config"], steps=cs["steps"], micro_batch=cs["micro_batch"],
                                data_description="the FlipQuant release fit set: 256 windows of 512 tokens of math and "
                                                 "code text (OpenWebMath, CodeParrot-clean)",
                                data_tokens=cs["data_tokens"], optimizer="AdamW (0.9, 0.999), weight decay 0, constant LR",
                                teacher="the unquantized BF16 model", training_seconds=cs["training_seconds"],
                                end_to_end_seconds=cs["end_to_end_seconds"],
                                gpu_peak_allocated_gib=cs["gpu_peak_allocated_gib"], gpu=cs["gpu"]))
    return out


# ------------------------------------------------------------------------------------------------------------- stage
def stage():
    STAGE.mkdir(exist_ok=True)
    lic_rec = json.loads((LIC / "licenses.json").read_text())
    record = dict(namespace=NAMESPACE, staged_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                  licenses_fetched=lic_rec, repos={})
    for model, repo in REPOS.items():
        dst = STAGE / repo
        dst.mkdir(exist_ok=True)
        src = RUN / "focus" / model / "state" / "focus.pt"
        out = dst / "focus.pt"
        if not out.exists():
            subprocess.run(["cp", "--reflink=always", str(src), str(out)], check=True)
        sha_src, sha_out = sha(src), sha(out)
        assert sha_src == sha_out, f"{model}: staged focus.pt differs from the source"
        hits, info = scan_state(out)
        assert info["source"]["hf_id"] and info["config"]["model"] == model
        ev, cs = evaluated(model), cost(model)
        assert info["modules"] == ev["record"]["quantized_modules"] == cs["parameters"]["modules"]
        assert info["block_factors_m"] == cs["parameters"]["crs"] and info["subblock_logits_q"] == cs["parameters"]["dgs"]
        lic_files = {}
        if model.startswith("qwen3"):                     # the FlipQuant staging's copy is the pinned snapshot's file
            snap = (Path("/home/dev/.cache/huggingface/hub") / ("models--" + info["source"]["hf_id"].replace("/", "--"))
                    / "snapshots" / info["source"]["revision"] / "LICENSE")
            assert sha(snap) == sha(LICENSE[model]["files"]["LICENSE"]), model
        for name, path in LICENSE[model]["files"].items():
            shutil.copyfile(path, dst / name)
            lic_files[name] = dict(source=str(path), sha256=sha(dst / name))
        summary = ppl_summary(model, info, ev, cs, sha_out, out.stat().st_size)
        (dst / "ppl_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
        pre = {f.name: dict(bytes=f.stat().st_size, sha256=sha_out if f.name == "focus.pt" else sha(f))
               for f in sorted(dst.iterdir()) if f.is_file()}
        files = {"focus.pt": pre["focus.pt"], "ppl_summary.json": pre["ppl_summary.json"],
                 **{n: pre[n] for n in ("LICENSE", "NOTICE") if n in pre}}
        (dst / "README.md").write_text(card(model, info, ev, cs, files))
        for f in sorted(dst.iterdir()):
            if f.is_file() and f.name != "focus.pt":
                hits += scan_text(f.name, f.read_text(), ref=f.name in ("README.md", "ppl_summary.json"))
        staged = {f.name: dict(bytes=f.stat().st_size, sha256=sha_out if f.name == "focus.pt" else sha(f))
                  for f in sorted(dst.iterdir()) if f.is_file()}
        record["repos"][f"{NAMESPACE}/{repo}"] = dict(
            model=model, base_model=info["source"]["hf_id"], revision=info["source"]["revision"],
            state=dict(source=str(src), sha256_source=sha_src, sha256_staged=sha_out, identical=True,
                       metadata_changes=[], **{k: v for k, v in info.items() if k not in ("config", "source")},
                       config=info["config"], source_meta=info["source"]),
            license_files=lic_files, files=staged, scan=hits,
            ppl=summary["ppl"], paired_vs_fourover6=summary["paired_vs_fourover6"], calibration_cost=cs)
        print(f"{NAMESPACE}/{repo}: files {list(staged)} scan hits {len(hits)} "
              f"FOCUS {ev['ppl']['focus']['wiki']:.4f}/{ev['ppl']['focus']['c4']:.4f}", flush=True)
    (STAGE / "stage.json").write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n")


# ----------------------------------------------------------------------------------------------------- check / upload
def anonymous_status(repo_id):
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/models/{repo_id}"), timeout=30)
        return 200
    except urllib.error.HTTPError as e:
        return e.code


def account():
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print("account:", who.get("name"))
    if who.get("name") != NAMESPACE:
        sys.exit(f"the active account is not {NAMESPACE}; stopping")
    return api


def check():
    api = account()
    stage_rec = json.loads((STAGE / "stage.json").read_text())
    exists = {r: api.repo_exists(r, repo_type="model") for r in stage_rec["repos"]}
    print("existing:", [r for r, e in exists.items() if e] or "none")
    if any(exists.values()):
        sys.exit("a target repository already exists; stopping (nothing is overwritten)")


def upload():
    from huggingface_hub import hf_hub_download
    api = account()
    stage_rec = json.loads((STAGE / "stage.json").read_text())
    path = STAGE / "hf_upload.json"
    out = json.loads(path.read_text()) if path.exists() else dict(account=NAMESPACE, repos={})
    for repo_id, rec in stage_rec["repos"].items():
        if out["repos"].get(repo_id, {}).get("verified"):
            print(f"{repo_id}: already uploaded and verified")
            continue
        folder = STAGE / repo_id.split("/", 1)[1]
        names = sorted(rec["files"])
        current = {f.name for f in folder.iterdir() if f.is_file()}
        assert current == set(names), f"{repo_id}: staged files changed: {sorted(current)}"
        for n in names:                                            # cheap check here; the full hash is after the upload
            assert (folder / n).stat().st_size == rec["files"][n]["bytes"], f"{repo_id}: {n} changed size"
        if not api.repo_exists(repo_id, repo_type="model"):
            api.create_repo(repo_id, repo_type="model", private=True, exist_ok=False)
        info = api.model_info(repo_id)
        anon = anonymous_status(repo_id)
        if info.private is not True or anon == 200:
            sys.exit(f"{repo_id}: not private (private={info.private}, anonymous HTTP {anon}); stopping")
        api.upload_large_folder(repo_id=repo_id, folder_path=folder, repo_type="model", private=True,
                                allow_patterns=names, print_report_every=300)
        info = api.model_info(repo_id, files_metadata=True)
        anon_after = anonymous_status(repo_id)
        hub = {s.rfilename: s for s in info.siblings}
        problems = []
        if info.private is not True or anon_after == 200:
            problems.append(f"not private (private={info.private}, anonymous HTTP {anon_after})")
        if set(hub) - {".gitattributes"} != set(names):
            problems.append(f"file list {sorted(hub)}")
        card_data = info.card_data.to_dict() if info.card_data else {}
        if card_data.get("base_model") != rec["base_model"] or not card_data.get("license"):
            problems.append(f"card metadata {card_data}")
        files = {}
        for n in names:
            want = rec["files"][n]["sha256"]
            lfs = getattr(hub.get(n), "lfs", None)
            lfs_sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            with tempfile.TemporaryDirectory(dir=STAGE) as tmp:          # one file at a time (disk)
                p = hf_hub_download(repo_id, n, revision=info.sha, cache_dir=tmp, force_download=True)
                got = sha(p)
            files[n] = dict(sha256=want, bytes=rec["files"][n]["bytes"], hub_size=getattr(hub.get(n), "size", None),
                            hub_lfs_sha256=lfs_sha, download_sha256=got, equal=got == want,
                            lfs_equal=None if lfs_sha is None else lfs_sha == want)
            if got != want or (lfs_sha is not None and lfs_sha != want):
                problems.append(f"{n}: sha256 differs (download {got}, lfs {lfs_sha})")
            print(f"  {repo_id} {n}: download sha256 {'==' if got == want else '!='} staged; lfs {lfs_sha == want if lfs_sha else '-'}",
                  flush=True)
        commits = api.list_repo_commits(repo_id)
        out["repos"][repo_id] = dict(url=f"https://huggingface.co/{repo_id}", head=info.sha, private=info.private,
                                     anonymous_http=anon_after, hub_files=sorted(hub),
                                     commits=[dict(id=c.commit_id, title=c.title) for c in commits],
                                     card_metadata=card_data, page=rendered(repo_id), files=files, problems=problems,
                                     verified=not problems)
        path.write_text(json.dumps(out, indent=1, default=str) + "\n")
        print(f"{repo_id}: head {info.sha} private={info.private} anonymous={anon_after} problems={problems or 'none'}",
              flush=True)
        if problems:
            sys.exit(f"{repo_id}: verification failed; stopping")


def cards():
    """Each uploaded card: the Hub's own YAML validation of the card loaded from the repository (ModelCard.validate),
    the README downloaded from the Hub equal to the staged one, and a local CommonMark + tables render (markdown-it-py):
    every Markdown table becomes an HTML table and every code fence a code block. The web page of a private repository
    cannot be fetched with a token (HTTP 401), so this stands in for "the card renders". Writes STAGE/cards.json."""
    from huggingface_hub import ModelCard
    from markdown_it import MarkdownIt
    up = json.loads((STAGE / "hf_upload.json").read_text())
    md = MarkdownIt("commonmark").enable("table")
    out = {}
    for repo_id, d in up["repos"].items():
        card = ModelCard.load(repo_id, repo_type="model")
        staged = (STAGE / repo_id.split("/", 1)[1] / "README.md").read_text()
        try:
            card.validate(repo_type="model")
            valid = True
        except Exception as e:                                    # noqa: BLE001 -- recorded, not raised
            valid = f"{type(e).__name__}: {e}"
        body = card.text
        html = md.render(body)
        lines = body.splitlines()
        n_tables = sum(1 for i, l in enumerate(lines) if re.match(r"^\|[-:| ]+\|$", l) and i and lines[i - 1].startswith("|"))
        n_fences = sum(1 for l in lines if l.startswith("```")) // 2
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id)
        out[repo_id] = dict(private_now=info.private, anonymous_http_now=anonymous_status(repo_id),
                            hub_yaml_valid=valid, readme_equal_to_staged=card.content == staged,
                            card_data=card.data.to_dict(), tables_markdown=n_tables, tables_html=html.count("<table>"),
                            code_fences=n_fences, code_blocks_html=html.count("<pre><code"),
                            h1=html.count("<h1>"), h2=html.count("<h2>"), html_bytes=len(html),
                            renders=valid is True and card.content == staged and n_tables == html.count("<table>") and
                            n_fences == html.count("<pre><code") and html.count("<h1>") == 1)
        print(repo_id, {k: v for k, v in out[repo_id].items() if k != "card_data"}, flush=True)
    (STAGE / "cards.json").write_text(json.dumps(out, indent=1) + "\n")


# ------------------------------------------------------------------------------------------------------------- record
def rendered(repo_id):
    """The repository's web page fetched with the active token in the request header only (never printed): HTTP status,
    and whether the page shows the card's title and its perplexity table."""
    from huggingface_hub import get_token
    token = get_token()
    req = urllib.request.Request(f"https://huggingface.co/{repo_id}", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            html = r.read().decode("utf-8", errors="replace")
            return dict(http=r.status, title_shown="FOCUS NVFP4 scales for" in html,
                        table_shown="FOCUS (this state)" in html, license_shown="License" in html)
    except urllib.error.HTTPError as e:
        return dict(http=e.code)


def record():
    """results/paper_eval/focus_hf/: HF_UPLOAD.md, stage.json, hf_upload.json and each repository's card and summary."""
    st = json.loads((STAGE / "stage.json").read_text())
    up = json.loads((STAGE / "hf_upload.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("stage.json", "hf_upload.json"):
        shutil.copyfile(STAGE / name, OUT / name)
    for repo_id in st["repos"]:
        d = OUT / "cards" / repo_id.split("/", 1)[1]
        d.mkdir(parents=True, exist_ok=True)
        for f in ("README.md", "ppl_summary.json"):
            shutil.copyfile(STAGE / repo_id.split("/", 1)[1] / f, d / f)
    lic = st["licenses_fetched"]
    cj = json.loads((STAGE / "cards.json").read_text())
    shutil.copyfile(STAGE / "cards.json", OUT / "cards.json")
    L = ["# Hugging Face upload of the FOCUS states (2026-10-09)", "",
         "The user's request (relayed by the coordinator, 2026-10-09): the FOCUS states behind the main-ppl FOCUS row "
         "(paper-eval dbe3a48, part K) as five **private** repositories under `edgeai-lab`, `focus.pt` as is (no NVFP4 "
         "code export). Qwen3.8-27B has no FOCUS state and no repository. The active account was checked first "
         "(`edgeai-lab`; no token printed or changed); every repository was created private and checked private before "
         "any file went up. Nothing is public.", "",
         "| model | repository | HF head commit | files |", "|---|---|---|---|"]
    for repo_id, d in up["repos"].items():
        L.append(f"| {st['repos'][repo_id]['model']} | [{repo_id}]({d['url']}) | `{d['head']}` | "
                 f"{', '.join(f'`{f}`' for f in d['hub_files'])} |")
    L += ["", "Upload path: huggingface_hub 1.31 `create_repo(private=True)`, then `upload_large_folder` (the resumable, "
          "multi-worker large-file path; Xet storage) with exactly the staged files; each repository has two commits, "
          "the empty `initial commit` and `Add files using upload-large-folder tool`.", "",
          "## Files and sha256", "",
          "Every file was downloaded again from the Hub at the head commit; its sha256 equals the staged file's (and, for "
          "`focus.pt`, the Hub's LFS sha256 too).", "",
          "| repository | file | bytes | sha256 (staged = remote) | download | LFS |", "|---|---|---:|---|---|---|"]
    for repo_id, d in up["repos"].items():
        for n, f in d["files"].items():
            L.append(f"| {repo_id.split('/', 1)[1]} | `{n}` | {f['bytes']:,} | `{f['sha256']}` | "
                     f"{'equal' if f['equal'] else 'DIFFERS'} | "
                     f"{'-' if f['lfs_equal'] is None else ('equal' if f['lfs_equal'] else 'DIFFERS')} |")
    L += ["", "## focus.pt: the state as is", "",
          "`focus.pt` is part K's calibration output, copied unchanged (a reflink copy; sha256 of the source = the staged "
          "file = the uploaded file). Its metadata needed no change: the scan below found no local path, user name, email, "
          "GitHub owner, reference-implementation branch or commit, or \"razer\" in its `config` and `source`, its layer "
          "and field names, its archive record names or its raw bytes.", "",
          "| model | source (part K) | sha256 (source = uploaded) | layers | m | q | config |", "|---|---|---|---:|---:|---:|---|"]
    for repo_id, r in st["repos"].items():
        s = r["state"]
        assert s["sha256_source"] == s["sha256_staged"] == up["repos"][repo_id]["files"]["focus.pt"]["sha256"]
        cfg = s["config"]
        L.append(f"| {r['model']} | `{s['source']}` | `{s['sha256_source']}` | {s['modules']} | {s['block_factors_m']:,} | "
                 f"{s['subblock_logits_q']:,} | data {cfg['data']}, {cfg['windows']} x {cfg['seqlen']}, batch {cfg['batch']}, "
                 f"lr {cfg['lr_scale']} / {cfg['lr_sub']}, top-{cfg['topk']}, seed {cfg['seed']} |")
    L += ["", "## Scans", "",
          "The FlipQuant upload's patterns (hf_stage.py: local path prefixes, the run directories, \"razer\", the "
          "GitHub owner, the user names, the university domain, HF tokens, email addresses, the host name, the internal "
          "branch name), plus `angelslim|tencent|github|gitlab|branch|commit` on the state's metadata and on our own text "
          "files (README.md, ppl_summary.json), over every staged file: the state's metadata strings and names, its "
          "archive record names and its raw bytes (streamed; the patterns /home/, /tmp/, razer, the owner, HF tokens, the "
          "user names, the run directory, angelslim), and every line of the text files.", "",
          "| repository | hits |", "|---|---:|"]
    L += [f"| {repo_id.split('/', 1)[1]} | {len(r['scan'])} |" for repo_id, r in st["repos"].items()]
    L += ["", "## License files", "",
          "- Qwen3-1.7B, Qwen3-8B: `LICENSE`, the base snapshot's file at the pinned revision (as staged for the FlipQuant "
          "repositories; Apache-2.0).",
          f"- Mistral-7B-Instruct-v0.3: the base repository has no license file (its card declares apache-2.0); `LICENSE` "
          f"is the Apache License 2.0 text from {lic['mistral']['source']} (retrieved {lic['mistral']['retrieved']}, "
          f"sha256 `{lic['mistral']['sha256']}`), added as the request asked (\"add Mistral's license\").",
          "- NVIDIA-Nemotron-Nano-9B-v2: `LICENSE` (the NVIDIA Open Model License Agreement, verbatim from the page the "
          "base card links, retrieved 2026-10-07) and `NOTICE`, as staged for the FlipQuant repository.",
          f"- phi-4: `LICENSE`, the base repository's own file ({lic['phi4']['source']}, retrieved "
          f"{lic['phi4']['retrieved']} without a token; MIT).", "",
          "## The cards", "",
          "Each `README.md` (copied under `cards/`) has front matter with the base model's license, `base_model`, and the "
          "tags quantization / fp4 / nvfp4 / focus, and no draft banner. It states: our reproduction of FOCUS "
          "(arXiv:2608.01847) as a baseline for the FlipQuant paper, not an official release by the FOCUS authors; the "
          "pinned base revision; the paper's NVFP4 hyperparameters (lr 5e-3 / 1e-3, 2 sub-blocks of 8, KL-Top k = 1000, "
          "AdamW with a constant LR, 1 epoch, global batch 32, init q 6, seed 42); the different calibration data (the "
          "FlipQuant release set, 256 x 512 math/code windows, 8 steps; the paper: 1,248 x 2,048 WikiText-2, 39 steps); "
          "per-token FourOverSix activations in calibration and evaluation; what the file holds (FP32 m per 16-element "
          "block, q per 8-element sub-block, s2 per tensor, per linear layer) and how it deploys as standard NVFP4 codes; "
          "that it is not a quantized checkpoint (it needs the BF16 base at the pinned revision and the FOCUS "
          "implementation; the code will be released with the paper, no link); native WikiText-2 / C4 PPL on the RTX PRO "
          "6000 with BF16 / NVFP4 / FourOverSix and the paired ΔNLL vs FourOverSix (from part K's records and final.json's "
          "sources, recomputed and checked equal to final.json); training time and peak GPU memory (report.json). "
          "`ppl_summary.json` holds the same numbers and the settings, without local paths.", "",
          "## Checks after the upload", "",
          "| repository | private (after upload / after all five) | anonymous API | card metadata (Hub) | card render "
          "(cards.json) | problems |", "|---|---|---:|---|---|---|"]
    for repo_id, d in up["repos"].items():
        cm, c = d["card_metadata"], cj[repo_id]
        L.append(f"| {repo_id.split('/', 1)[1]} | {d['private']} / {c['private_now']} | HTTP {d['anonymous_http']} / "
                 f"{c['anonymous_http_now']} | license {cm.get('license')}, "
                 f"base_model {cm.get('base_model')}, tags {cm.get('tags')} | Hub YAML validation "
                 f"{'passed' if c['hub_yaml_valid'] is True else c['hub_yaml_valid']}; {c['tables_html']}/{c['tables_markdown']} "
                 f"tables, {c['code_blocks_html']}/{c['code_fences']} code blocks rendered; README = staged: "
                 f"{c['readme_equal_to_staged']} | {d['problems'] or 'none'} |")
    L += ["", "\"The card renders\": the web page of a private repository cannot be fetched with a token (HTTP 401 for "
          "every repository, recorded as `page` in hf_upload.json), so the check is the Hub's own YAML validation of the "
          "card it serves (`ModelCard.validate`), the metadata the Hub parsed from it, the README downloaded from the Hub "
          "byte-identical to the staged one, and a local CommonMark + tables render (markdown-it-py) in which every table "
          "and code fence renders. The user can open the repositories while logged in."]
    L += ["", "Not uploaded: report.json, ppl.json and every other record (they hold local paths).", "",
          "Files here: `stage.json` (staging: files, sha256, scans, license sources, the state's metadata), "
          "`hf_upload.json` (the upload: commits, per-file checks), `cards/<repo>/{README.md, ppl_summary.json}`; "
          "scripts `experiments/paper_eval/focus_hf.py`.", ""]
    (OUT / "HF_UPLOAD.md").write_text("\n".join(L))
    print("\n".join(L[:20]))


if __name__ == "__main__":
    {"licenses": licenses, "stage": stage, "check": check, "upload": upload, "cards": cards,
     "record": record}[sys.argv[1]]()

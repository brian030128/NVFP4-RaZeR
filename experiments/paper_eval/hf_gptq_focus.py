"""The user's request of 2026-10-15 (relayed by the coordinator): the Llama-3.2-1B-Instruct FOCUS state and the main-table
GPTQ-double-dagger weight codes of the eight paper models, as nine PRIVATE repositories under edgeai-lab, in the formats
of the earlier uploads (focus_hf.py, llama_hf.py).

- edgeai-lab/Llama-3.2-1B-Instruct-FOCUS-NVFP4: part P's state (focus.pt as is), card, ppl_summary.json, Llama 3.2
  license files; the layout of edgeai-lab/Llama-3.1-8B-Instruct-FOCUS-NVFP4.
- edgeai-lab/<Base>-GPTQ-NVFP4 (eight): the GPTQ codes file behind tab:main-ppl's GPTQ-double-dagger row as is
  (codes_nvfp4_act-fo6.pt: NVFP4 weights, GPTQ with BF16-activation calibration on the release 256 x 512 windows, block
  128, damp 0.01), card, ppl_summary.json, the base model's license files.
Every source is checked against its record before staging (file sha256, the GPTQ code digest and settings key, the main-
table variant); staging uses reflinks; every staged file is scanned with the earlier uploads' patterns.

    python hf_gptq_focus.py licenses   # Llama 3.2's LICENSE.txt / USE_POLICY.md, read-only with the active token
    python hf_gptq_focus.py stage      # verify, stage, cards, scans: RECS/stage.json
    python hf_gptq_focus.py check      # read-only: the active account, no target repository exists, scans accepted
    python hf_gptq_focus.py upload     # create PRIVATE repositories, upload, verify: RECS/hf_upload.json
    python hf_gptq_focus.py cards      # the Hub's YAML validation, README equal to staged: RECS/cards.json
    python hf_gptq_focus.py record     # results/paper_eval/hf_gptq_focus/HF_UPLOAD.md (+ records and cards)

Later the same day (the user's grant, relayed by the coordinator): edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant, PRIVATE,
holding part P's three Llama-3.2-1B-Instruct maps. It takes the layout, file names and card of
edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant (llama_hf.py's staging and card, bound to this model), with the 1B FOCUS
repository's Llama 3.2 license files.

    python hf_gptq_focus.py stage_fq1b # verify against part P's records, stage, scan: added to stage.json
"""
import contextlib
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
import llama_hf as LH  # noqa: E402
import run_gpu as G  # noqa: E402
import run_zeroshot6 as Z  # noqa: E402
import tables_cpu as T  # noqa: E402

NS = "edgeai-lab"
GPTQ_STAGE = Path("/home/dev/gptq_hf")
RECS = GPTQ_STAGE / "records"
OUT = T.ROOT / "results" / "paper_eval" / "hf_gptq_focus"
LIC32 = Path("/home/dev/flipquant_hf_licenses") / "llama32"
OLD = Path("/home/dev/flipquant_hf")
FLIC = Path("/home/dev/focus_hf_licenses")
CODES_NAME = "codes_nvfp4_act-fo6.pt"
LFS_ONLY_ABOVE = 1 << 30                          # bytes: larger files are verified by the Hub's LFS sha256, not downloaded
TAGS = ["quantization", "fp4", "nvfp4", "gptq"]
CODE = "The FlipQuant code (GPTQ, the native kernels and the evaluation) will be released with the paper."
# models: title (the existing repositories' <Base>), base model id, pinned revision, the code revision that wrote the file
MODELS = {
    "qwen3-1.7b": ("Qwen3-1.7B", "Qwen/Qwen3-1.7B", Z.REV["qwen3-1.7b"], "c67a5bd"),
    "qwen3-8b": ("Qwen3-8B", "Qwen/Qwen3-8B", Z.REV["qwen3-8b"], "c67a5bd"),
    "mistral-7b": ("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", Z.REV["mistral-7b"], "c67a5bd"),
    "nemotron-nano-9b-v2": ("NVIDIA-Nemotron-Nano-9B-v2", "nvidia/NVIDIA-Nemotron-Nano-9B-v2", Z.REV["nemotron-nano-9b-v2"],
                            "1a2094e"),
    "phi4-14b": ("phi-4", "microsoft/phi-4", Z.REV["phi4-14b"], "c67a5bd"),
    "qwen3.8-27b": ("Qwen3.8-27B", "Qwen/Qwen3.8-27B", Z.REV["qwen3.8-27b"], "c67a5bd"),
    "llama3.2-1b-instruct": ("Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", Z.REV["llama3.2-1b-instruct"],
                             "3db0072"),
    "llama3.1-8b-instruct": ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", Z.REV["llama3.1-8b-instruct"],
                             "3db0072"),
}
APACHE_NOTE = ("The base model card declares Apache-2.0, and the base repository has no license file; `LICENSE` is the "
               f"Apache License 2.0 text from {FH.APACHE_URL}.")
L31_NOTICE = LH.LLAMA_NOTICE
L32_NOTICE = ("Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights "
              "Reserved.")
LICENSES = {
    "qwen3-1.7b": dict(front={"license": "apache-2.0"}, name="Apache License 2.0", files=dict(LICENSE=OLD / "Qwen3-1.7B-FlipQuant" / "LICENSE"),
                       note="`LICENSE` is a copy of the base model's license file (Qwen/Qwen3-1.7B)."),
    "qwen3-8b": dict(front={"license": "apache-2.0"}, name="Apache License 2.0", files=dict(LICENSE=OLD / "Qwen3-8B-FlipQuant" / "LICENSE"),
                     note="`LICENSE` is a copy of the base model's license file (Qwen/Qwen3-8B)."),
    "mistral-7b": dict(front={"license": "apache-2.0"}, name="Apache License 2.0", files=dict(LICENSE=FLIC / "mistral" / "LICENSE"),
                       note=APACHE_NOTE),
    "nemotron-nano-9b-v2": dict(front=FH.LICENSE["nemotron-nano-9b-v2"]["front"], name="NVIDIA Open Model License Agreement",
                                files=FH.LICENSE["nemotron-nano-9b-v2"]["files"],
                                note=(f"Governing terms: the [NVIDIA Open Model License Agreement]({FH.NVIDIA_URL}). "
                                      "`LICENSE` is the agreement (Last Modified: October 24, 2025), verbatim from the page "
                                      f"the base model card links (the base repository has no license file; retrieved "
                                      f"{FH.NVIDIA_RETRIEVED}); `NOTICE` is the attribution notice “Licensed by NVIDIA "
                                      "Corporation under the NVIDIA Open Model License” (section 3.1).")),
    "phi4-14b": dict(front={"license": "mit"}, name="MIT License", files=dict(LICENSE=FLIC / "phi4" / "LICENSE"),
                     note="`LICENSE` is a copy of the base repository's license file (microsoft/phi-4)."),
    "qwen3.8-27b": dict(front={"license": "apache-2.0"}, name="Apache License 2.0", files=dict(LICENSE=FLIC / "mistral" / "LICENSE"),
                        note=APACHE_NOTE),
    "llama3.2-1b-instruct": dict(front={"license": "llama3.2"}, name="Llama 3.2 Community License",
                                 files=dict(LICENSE=LIC32 / "LICENSE", USE_POLICY=LIC32 / "USE_POLICY.md",
                                            NOTICE=LIC32 / "NOTICE"), notice=L32_NOTICE, version="3.2"),
    "llama3.1-8b-instruct": dict(front={"license": "llama3.1"}, name="Llama 3.1 Community License",
                                 files=dict(LICENSE=LH.LIC / "LICENSE", USE_POLICY=LH.LIC / "USE_POLICY.md",
                                            NOTICE=LH.BASE_NOTICE), notice=L31_NOTICE, version="3.1"),
}
FILE_NAME = {"LICENSE": "LICENSE", "USE_POLICY": "USE_POLICY.md", "NOTICE": "NOTICE"}
FOCUS_MODEL = "llama3.2-1b-instruct"
FOCUS_REPO = f"{NS}/Llama-3.2-1B-Instruct-FOCUS-NVFP4"
FOCUS_STAGE = Path("/home/dev/focus_hf") / FOCUS_REPO.split("/")[1]
FOCUS_STATE = G.RUN / "llama_instruct" / "focus_states" / FOCUS_MODEL / "state"
KEY_FIELDS = ("model", "weight", "act", "unit", "mode", "map_sha256", "rotate", "damp", "block", "windows", "calib", "rules",
              "hessian_input", "data_sha256", "propagate")
MAIN_TABLE_KEY = dict(weight="nvfp4", act="none", unit=None, mode="fake", map_sha256=None, rotate="none", damp=0.01, block=128,
                      windows=256, calib="release", rules="modelopt-0.47", hessian_input="bf16", propagate="bf16")
ACCEPTED = [lambda h: h["file"] == "USE_POLICY.md" and h["match"] == "LlamaUseReport@meta.com",
            # llama_hf's second rule, for the 1B FlipQuant card: the words "commit" / "branch" in the card's own sentences
            # (the FlipQuant commit is kept; the provenance paragraph)
            lambda h: h["file"] == "README.md" and h["match"] in ("commit", "branch") and (
                h["text"].startswith("`python -m calibration.train_map --model llama3.2-1b-instruct") or
                h["text"].startswith("**Provenance of the map files.**"))]


# the user's naming decision (2026-10-15): <Base>-GPTQ-BF16Calib-NVFP4, because edgeai-lab/<Base>-GPTQ-NVFP4 already holds
# another upload's GPTQ (FourOverSix calibration activations) for these two models
COAUTHOR_GPTQ = {"nemotron-nano-9b-v2", "qwen3.8-27b"}


def gptq_repo(model):
    return f"{NS}/{MODELS[model][0]}-GPTQ-BF16Calib-NVFP4"


def old_gptq_repo(model):
    return f"{NS}/{MODELS[model][0]}-GPTQ-NVFP4"


def unaccepted(hits):
    return [h for h in hits if not any(rule(h) for rule in ACCEPTED)]


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------------------------------------- licenses
def licenses():
    """Llama 3.2's LICENSE.txt (staged as LICENSE) and USE_POLICY.md at the pinned revision, read-only with the active
    token; NOTICE carries the notice sentence the agreement prescribes (checked to appear in it verbatim)."""
    from huggingface_hub import hf_hub_download
    hf_id, rev = MODELS[FOCUS_MODEL][1], MODELS[FOCUS_MODEL][2]
    LIC32.mkdir(parents=True, exist_ok=True)
    rec = dict(source=f"{hf_id}@{rev}", retrieved=now(), files={})
    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
        for f, dst in (("LICENSE.txt", "LICENSE"), ("USE_POLICY.md", "USE_POLICY.md")):
            p = hf_hub_download(hf_id, f, revision=rev, cache_dir=tmp)       # gated: the active token, read-only
            (LIC32 / dst).write_bytes(Path(p).read_bytes())
    lic = (LIC32 / "LICENSE").read_text()
    text = re.sub(r"\s+", " ", lic)
    assert lic.lstrip().startswith("LLAMA 3.2 COMMUNITY LICENSE AGREEMENT"), lic[:80]
    for s in ("Llama 3.2 Version Release Date", "1. License Rights and Redistribution", "Built with Llama",
              L32_NOTICE, "2. Additional Commercial Terms", "3. Disclaimer of Warranty", "4. Limitation of Liability",
              "5. Intellectual Property", "6. Term and Termination", "7. Governing Law and Jurisdiction"):
        assert s in text, s
    assert "Acceptable Use Policy" in (LIC32 / "USE_POLICY.md").read_text()
    (LIC32 / "NOTICE").write_text(L32_NOTICE + "\n")
    for f in ("LICENSE", "USE_POLICY.md", "NOTICE"):
        rec["files"][f] = dict(sha256=FH.sha(LIC32 / f), bytes=(LIC32 / f).stat().st_size)
    rec["files"]["LICENSE"]["source_file"] = "LICENSE.txt"
    rec["files"]["NOTICE"]["source_file"] = "written here: the agreement's notice sentence (section 1.b.iii), verbatim"
    (LIC32 / "licenses.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


def license_files(model, dst):
    out = {}
    for k, src in LICENSES[model]["files"].items():
        name = FILE_NAME[k]
        shutil.copyfile(src, dst / name)
        out[name] = dict(sha256=FH.sha(dst / name), source=str(src))
    if "notice" in LICENSES[model]:
        assert (dst / "NOTICE").read_text().strip() == LICENSES[model]["notice"]
    return out


def llama_license_text(model, what):
    """The Llama-specific card text: what = 'head' (under the title) or 'terms' (the license section)."""
    lic = LICENSES[model]
    title, hf_id, rev = MODELS[model][:3]
    v = lic["version"]
    if what == "head":
        return (f"**Built with Llama.**\n\n{lic['notice']} The agreement is in `LICENSE`, the Acceptable Use Policy it "
                "incorporates in `USE_POLICY.md`, and the notice in `NOTICE`.")
    return (f"The Llama {v} Community License (section 1.b) applies to derivative works: provide a copy of the agreement, "
            "prominently display “Built with Llama”, include “Llama” at the beginning of the name of an AI model created "
            f"or improved with Llama materials or their outputs, and keep the notice “{lic['notice']}” in a NOTICE file.\n\n"
            f"- `LICENSE`: the Llama {v} Community License Agreement, the base repository's own license file ({hf_id} at "
            f"revision `{rev}`).\n"
            f"- `USE_POLICY.md`: the Llama {v} Acceptable Use Policy, which the agreement incorporates; the base "
            "repository's own `USE_POLICY.md`.\n"
            f"- `NOTICE`: “{lic['notice']}”")


# ------------------------------------------------------------------------------------------------------- GPTQ sources
def verify_codes(model):
    """The codes file behind the main table's GPTQ-double-dagger cell: its code digest and settings key equal the record's,
    the key is the main-table variant (BF16-activation calibration), and its file sha256 equals part X's record."""
    import torch
    path, record = Z.codes_file(model), Z.gptq_record(model)
    ptq = json.loads(record.read_text())["ptq"]
    obj = torch.load(path, map_location="cpu", weights_only=True)
    key = obj["key"]
    digest = LH_entries_sha256(obj["codes"])
    file_sha = FH.sha(path)
    x_codes = json.loads((Z.ZS / "step0" / "codes.json").read_text())["files"]
    x_gate = json.loads((Z.ZS / "step0" / "step0_ok.json").read_text())["codes_file_sha256"]
    recorded_file_sha = x_codes.get(model, {}).get("file_sha256") or x_gate.get(model)
    first = sorted(obj["codes"])[0]
    c = obj["codes"][first]
    out = dict(file=str(path), bytes=path.stat().st_size, file_sha256=file_sha, recorded_file_sha256=recorded_file_sha,
               record=str(record), format=obj.get("format"), modules=len(obj["codes"]), codes_sha256=digest,
               record_codes_sha256=ptq["codes_sha256"], key=key,
               key_differences={f: (key.get(f), ptq.get(f)) for f in KEY_FIELDS if key.get(f) != ptq.get(f)},
               main_table_variant={f: (key.get(f), v) for f, v in MAIN_TABLE_KEY.items() if key.get(f) != v},
               record_seconds=ptq.get("seconds"), record_cache=ptq.get("cache"),
               first_module=first, first_shapes=dict(q=list(c["q"].shape), s=list(c["s"].shape), g=list(c["g"].shape)),
               fmt_values=sorted({str(int(v["fmt"])) if not torch.is_tensor(v["fmt"]) else "tensor"
                                  for v in obj["codes"].values()}))
    out["decode_snippet_bitwise"] = decode_check(c)
    out["ok"] = (out["format"] == "flipquant-gptq/1" and digest == ptq["codes_sha256"] and not out["key_differences"]
                 and not out["main_table_variant"] and key.get("model") == model and out["fmt_values"] == ["1"]
                 and (recorded_file_sha is None or recorded_file_sha == file_sha) and out["decode_snippet_bitwise"])
    del obj
    return out


def LH_entries_sha256(entries):
    """flipquant's codes_sha256 formula over the stored entries (paper-sm120-runs gptq.entries_sha256; zs6_codes.py)."""
    import hashlib
    import torch
    h = hashlib.sha256()
    for n in sorted(entries):
        c = entries[n]
        h.update(n.encode())
        for t in (c["q"], c["s"], c["g"].float()):
            h.update(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        h.update(c["fmt"].numpy().tobytes() if torch.is_tensor(c["fmt"]) else str(int(c["fmt"])).encode())
    return h.hexdigest()


DECODE_SNIPPET = """import torch

obj = torch.load("codes_nvfp4_act-fo6.pt", map_location="cpu", weights_only=True)
obj["key"]                                    # the GPTQ settings
c = obj["codes"][NAME]                        # one linear layer, by its module name
nib = torch.stack((c["q"] & 15, c["q"] >> 4), -1).flatten(-2).long()    # [out, in] 4-bit codes
e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
values = torch.cat([e2m1, -e2m1])[nib]       # code = sign << 3 | magnitude index
scales = c["s"].view(torch.float8_e4m3fn).float()                       # [out, in / 16] E4M3 block scales
w = (values.unflatten(-1, (-1, 16)) * scales.unsqueeze(-1)).flatten(-2) * c["g"]   # the weight, FP32
"""


def decode_check(c):
    """The card's PyTorch snippet reproduces flipquant's own dequantization (FP4Codes.dequantize) bit for bit."""
    import torch
    sys.path.insert(0, str(Z.FQ))
    from flipquant import formats as F
    nib = torch.stack((c["q"] & 15, c["q"] >> 4), -1).flatten(-2).long()
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = torch.cat([e2m1, -e2m1])[nib]
    scales = c["s"].view(torch.float8_e4m3fn).float()
    w = (values.unflatten(-1, (-1, 16)) * scales.unsqueeze(-1)).flatten(-2) * c["g"]
    ref = F.FP4Codes(F.unpack_nibbles(c["q"]), c["s"].view(torch.float8_e4m3fn).float(), c["g"],
                     int(c["fmt"])).dequantize(torch.float32)
    return bool(torch.equal(w, ref))


def scan_codes(path):
    """The codes file's settings strings, its module names, the archive's record names and its raw bytes."""
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    strings = list(FH.walk(obj["key"], "key")) + [("codes (key)", k) for k in obj["codes"]] + [("format", obj["format"])]
    with zipfile.ZipFile(path) as z:
        strings += [("archive record", n) for n in z.namelist()]
    hits = []
    for where, s in strings:
        for rx in (FH.SCAN, FH.SCAN_REF):
            hits += [dict(file=path.name, where=where, match=m.group(0), text=s[:200]) for m in rx.finditer(s)]
    hits += [dict(file=path.name, where=f"raw bytes @{off}", match=m, text="") for off, m in sorted(FH.raw_hits(path).items())]
    return hits


# --------------------------------------------------------------------------------------------------------- the cards
def gptq_evaluated(model):
    """Main-table PPL (final.json) of BF16, NVFP4, FourOverSix and GPTQ-double-dagger, paired ΔNLL vs FourOverSix, and the
    GPTQ record's evaluation settings."""
    final = json.loads((T.ROOT / "results" / "paper_eval" / "final" / "final.json").read_text())
    rec = json.loads(Z.gptq_record(model).read_text())
    assert final["sources"][model]["gptq"] == str(Z.gptq_record(model)), (model, final["sources"][model]["gptq"])
    assert rec["quant"]["weight"] == "nvfp4" and rec["quant"]["act"] == "fourover6" and rec["native_backend"] == "sm120"
    return dict(ppl={r: final["ppl"][model][r] for r in ("bf16", "nvfp4", "fo6", "gptq")},
                paired={r: final["vs_fo6"][model][r] for r in ("bf16", "nvfp4", "gptq")}, record=rec,
                windows={c: rec["results"][c]["windows"] for c, _ in T.CORPORA})


def gptq_summary(model, v, ev):
    rec = ev["record"]
    title, hf_id, rev, code_rev = MODELS[model]
    return dict(model=hf_id, revision=rev, flipquant_model_key=model,
                method="GPTQ on NVFP4's fixed round-to-nearest grid, BF16-activation calibration (the FlipQuant paper's "
                       "GPTQ-double-dagger baseline)",
                evaluation=dict(mode="native sm_120 kernels (NVFP4 weights from these codes, per-token FourOverSix "
                                     "activations)", convention=rec["convention"], quantized_modules=rec["quantized_modules"],
                                transformers=rec["source"]["transformers"], torch=rec["source"]["torch"],
                                gpu=rec["resources"]["gpus"][0],
                                corpora={c: {"dataset": FH.DATASETS[c]} | {k: x for k, x in rec["results"][c]["data"].items()
                                                                            if k != "documents"} for c, _ in T.CORPORA},
                                c4_documents="the documents are drawn with the recorded seed from the listed shard"),
                ppl=ev["ppl"],
                paired_vs_fourover6={r: {c: dict(delta_nll=p["delta"], two_se=p["two_se"], n=p["n"],
                                                 significant=p["significant"], verdict=FH.verdict(p))
                                         for c, p in ev["paired"][r].items()} for r in ("bf16", "nvfp4", "gptq")},
                codes={CODES_NAME: dict(sha256=v["file_sha256"], bytes=v["bytes"], format=v["format"], modules=v["modules"],
                                        codes_sha256=v["codes_sha256"], key=v["key"])},
                gptq=dict(v["key"], flipquant_revision=code_rev, seconds=v["record_seconds"],
                          calibration_windows="256 windows of 512 tokens: the FlipQuant release maps' fit set (the "
                                              "paper's 128 math/code windows, OpenWebMath and CodeParrot-clean, plus 128 "
                                              "from the extension rule)"))


def gptq_card(model, v, ev, files):
    title, hf_id, rev, code_rev = MODELS[model]
    lic = LICENSES[model]
    key = v["key"]
    rec = ev["record"]
    wiki, c4 = rec["results"]["wiki"]["data"], rec["results"]["c4"]["data"]
    front = {**lic["front"], "base_model": hf_id, "tags": TAGS}
    llama = "notice" in lic
    regenerated = code_rev == "3db0072"
    L = ["---", yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=1000).strip(), "---", "",
         f"# GPTQ NVFP4 weights for {title}, calibrated with BF16 activations", ""]
    if llama:
        L += [llama_license_text(model, "head"), ""]
    L += ["These are the weights of the **GPTQ‡** baseline of the FlipQuant paper's main perplexity table: GPTQ-quantized "
          "NVFP4 weights, calibrated with BF16 activations and evaluated with FourOverSix per-token activations. They were "
          "made as a baseline for the FlipQuant paper.", ""]
    if model in COAUTHOR_GPTQ:
        L += [f"Not the same artifact as `{old_gptq_repo(model)}`: this repository is GPTQ calibrated with BF16 activations "
              "(the paper's main-table GPTQ‡); that one is calibrated with FourOverSix activations.", ""]
    L += [
          "## What this is", "",
          f"`{CODES_NAME}` holds the NVFP4 codes of every quantized linear layer of [{hf_id}](https://huggingface.co/{hf_id}) "
          f"at revision `{rev}` ({v['modules']} layers: all text-model linear layers except the LM head"
          + ("; the vision tower stays BF16" if model == "qwen3.8-27b" else "") + "). NVFP4 stores each weight as a 4-bit "
          "E2M1 code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.", "",
          "**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else "
          "(embeddings, norms, the LM head"
          + (", the Mamba / convolution parameters" if model == "nemotron-nano-9b-v2" else "")
          + (", the recurrent and convolution parameters" if model == "qwen3.8-27b" else "")
          + ") comes from the BF16 base model at the pinned revision.", "",
          "## How it was made", "",
          "GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: "
          "each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses "
          "the codes. The settings:", "",
          f"- Calibration data: {key['windows']} windows of 512 tokens, the FlipQuant release maps' fit set: the paper's "
          "128 math/code windows (OpenWebMath and CodeParrot-clean) plus 128 more from the extension rule. sha256 of their "
          f"token ids, in order: `{key['data_sha256']}`.",
          "- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already "
          "quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from "
          "the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.",
          f"- Lazy-update block {key['block']} columns, dampening {key['damp']} × mean(diag H), no act-order, the "
          "ModelOpt 0.47 handling of dead columns.",
          f"- Calibration batch {1 if model == 'qwen3.8-27b' else 8}; GPTQ took {v['record_seconds'] / 60:.1f} min on one "
          f"{rec['resources']['gpus'][0]}.",
          (f"- Code: FlipQuant revision `{code_rev}`. This file was regenerated with it on 2026-10-14; its codes equal, "
           "digest for digest, the codes the main table was evaluated with." if regenerated else
           f"- Code: FlipQuant revision `{code_rev}`, the run that produced the main table's numbers."),
          f"- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): "
          f"`{v['codes_sha256']}`.", "",
          CODE, "",
          "## Perplexity", "",
          f"The FlipQuant paper's main table. Native sm_120 kernels on the {rec['resources']['gpus'][0]}; per-token "
          f"FourOverSix activations for every quantized row except NVFP4 (per-token NVFP4 activations). WikiText-2 test "
          f"({ev['windows']['wiki']} windows of {wiki['window_tokens']} tokens) and C4 validation ({ev['windows']['c4']} "
          f"windows of {c4['window_tokens']} tokens); every row on the same windows. ΔNLL is the mean paired per-window "
          "difference in nats per token against FourOverSix; ± 2 SE over windows; `*` marks |Δ| > 2 SE. The verdict is "
          "“better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.", "",
          "| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict "
          "(WikiText-2 / C4) |", "|---|---:|---:|---:|---:|---|"]
    for r, name in (("bf16", "BF16"), ("nvfp4", "NVFP4 (round to nearest)"), ("fo6", "FourOverSix (reference)"),
                    ("gptq", "GPTQ‡ (these weights)")):
        p = ev["ppl"][r]
        if r == "fo6":
            d, vd = ["—", "—"], "reference"
        else:
            q = ev["paired"][r]
            d = [f"{q[c]['delta']:+.4f} ± {q[c]['two_se']:.4f}{' *' if q[c]['significant'] else ''}" for c, _ in T.CORPORA]
            vd = " / ".join(FH.verdict(q[c]) for c, _ in T.CORPORA)
        L.append(f"| {name} | {p['wiki']:.4f} | {p['c4']:.4f} | {d[0]} | {d[1]} | {vd} |")
    L += ["", "## How to use", "",
          "With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the "
          "command that produced the table:", "", "```bash",
          f"python -m evaluation.ppl --model {model} --revision {rev} --mode native \\",
          "    --weight nvfp4 --act fourover6 --any-act --paper-convention \\",
          "    --ptq gptq --gptq-propagate bf16 --gptq-hessian-input bf16 --gptq-calib release \\",
          "    --gptq-calib-record <calibration record> --gptq-calib-extension <fit-extension record> \\",
          f"    --gptq-block 128 --gptq-damp 0.01 --gptq-batch {1 if model == 'qwen3.8-27b' else 8} --gptq-cache {CODES_NAME}",
          "```", "",
          "FlipQuant reads the file instead of running GPTQ only if the file's stored settings, including the sha256 of the "
          "calibration windows, equal the run's; it rebuilds the windows from the calibration records to check this.", "",
          "With PyTorch alone, one layer's weight (this reproduces FlipQuant's own dequantization bit for bit):", "",
          "```python", DECODE_SNIPPET.replace("NAME", json.dumps(v["first_module"])).rstrip(), "```", "",
          f"In the file: `q` is uint8 `{v['first_shapes']['q']}`-shaped (two 4-bit codes per byte, element 2j in the low "
          f"nibble), `s` uint8 (the raw E4M3 bytes of the block scales), `g` the FP32 tensor scale, `fmt` 1 (every block "
          "E2M1).", "",
          "## License", "",
          f"These weights are derived from the base model's weights and are released under the base model's license, the "
          f"{lic['name']}.", ""]
    L += [llama_license_text(model, "terms") if llama else lic["note"], "",
          "## Files", "", "| file | bytes | sha256 |", "|---|---:|---|"]
    L += [f"| `{n}` | {d['bytes']:,} | `{d['sha256']}` |" for n, d in files.items()]
    L += ["", f"- `{CODES_NAME}`: the GPTQ codes (`flipquant-gptq/1`: `{{format, key, codes}}`; above), the file behind "
          "the main table's GPTQ‡ cell, unchanged.",
          "- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and "
          "the GPTQ settings."]
    L += [f"- `{FILE_NAME[k]}`: see License." for k in LICENSES[model]["files"]]
    L += [""]
    return "\n".join(L)


def focus_cost():
    r = json.loads((FOCUS_STATE / "report.json").read_text())
    assert r["status"] == "complete" and r["steps"] == 8 and len(r["log"]) == 8
    return dict(steps=r["steps"], micro_batch=int(r["args"]["micro"]), training_seconds=r["training_seconds"],
                end_to_end_seconds=r["resources"]["seconds"], gpu=r["resources"]["gpus"][0],
                gpu_peak_allocated_gib=r["resources"]["gpu_peak_gib"][0], parameters=r["parameters"],
                checkpointed_layers=r["checkpointed_layers"], data_windows=r["data"]["windows"],
                data_tokens=r["data"]["tokens"], kl_top_first_last=(r["log"][0]["kl_top"], r["log"][-1]["kl_top"]))


def focus_card(info, ev, cs, files):
    """focus_hf.card with the Llama 3.2 license items (llama_hf.focus_card for Llama 3.2)."""
    title, hf_id, rev = MODELS[FOCUS_MODEL][:3]
    FH.TITLES[FOCUS_MODEL] = title
    FH.LICENSE[FOCUS_MODEL] = dict(front={"license": "llama3.2"}, name="Llama 3.2 Community License",
                                   files=dict(LICENSE=LIC32 / "LICENSE", NOTICE=LIC32 / "NOTICE"))
    text = FH.card(FOCUS_MODEL, info, ev, cs, files)
    head = f"# FOCUS NVFP4 scales for {title} (our reproduction)\n"
    assert text.count(head) == 1
    text = text.replace(head, head + "\n" + llama_license_text(FOCUS_MODEL, "head") + "\n")
    old = f"`LICENSE` is a copy of the base model's license file ({hf_id} at revision `{rev}`)."
    assert text.count(old) == 1
    text = text.replace(old, llama_license_text(FOCUS_MODEL, "terms").replace(
        "\n\n- `LICENSE`", f" This state is trained against {title}'s outputs (KL to its BF16 logits).\n\n- `LICENSE`", 1))
    old2 = "- `NOTICE`: see License.\n"
    assert text.count(old2) == 1
    return text.replace(old2, old2 + "- `USE_POLICY.md`: see License.\n")


# ------------------------------------------------------------------------------------------------------------- stage
def stage_focus(record):
    FOCUS_STAGE.mkdir(parents=True, exist_ok=True)
    src, out = FOCUS_STATE / "focus.pt", FOCUS_STAGE / "focus.pt"
    want = json.loads((Z.ZS / "step0" / "step0_ok.json").read_text())["focus_state_sha256"][FOCUS_MODEL]
    if not out.exists():
        subprocess.run(["cp", "--reflink=always", str(src), str(out)], check=True)
    sha_src, sha_out = FH.sha(src), FH.sha(out)
    assert sha_src == sha_out == want, (sha_src, sha_out, want)
    hits, info = FH.scan_state(out)
    title, hf_id, rev = MODELS[FOCUS_MODEL][:3]
    assert info["source"]["hf_id"] == hf_id and info["source"]["revision"] == rev and info["config"]["model"] == FOCUS_MODEL
    ev, cs = FH.evaluated(FOCUS_MODEL), focus_cost()
    assert info["modules"] == ev["record"]["quantized_modules"] == cs["parameters"]["modules"]
    assert ev["record"]["focus"]["state"] == str(src)
    lic = license_files(FOCUS_MODEL, FOCUS_STAGE)
    fsum = FH.ppl_summary(FOCUS_MODEL, info, ev, cs, sha_out, out.stat().st_size)
    (FOCUS_STAGE / "ppl_summary.json").write_text(json.dumps(fsum, indent=1, ensure_ascii=False) + "\n")
    pre = {f.name: dict(bytes=f.stat().st_size, sha256=sha_out if f.name == "focus.pt" else FH.sha(f))
           for f in sorted(FOCUS_STAGE.iterdir()) if f.is_file()}
    cfiles = {n: pre[n] for n in ("focus.pt", "ppl_summary.json", "LICENSE", "NOTICE", "USE_POLICY.md")}
    (FOCUS_STAGE / "README.md").write_text(focus_card(info, ev, cs, cfiles))
    for f in sorted(FOCUS_STAGE.iterdir()):
        if f.is_file() and f.name != "focus.pt":
            hits += FH.scan_text(f.name, f.read_text(), ref=f.name in ("README.md", "ppl_summary.json"))
    files = {f.name: dict(bytes=f.stat().st_size, sha256=sha_out if f.name == "focus.pt" else FH.sha(f))
             for f in sorted(FOCUS_STAGE.iterdir()) if f.is_file()}
    record["repos"][FOCUS_REPO] = dict(
        kind="focus", model=FOCUS_MODEL, base_model=hf_id, revision=rev, folder=str(FOCUS_STAGE), license=FH.LICENSE[FOCUS_MODEL]["front"],
        state=dict(source=str(src), sha256_source=sha_src, sha256_staged=sha_out, recorded_sha256=want,
                   recorded_by="paper-eval part X step 0 (RUN/zeroshot6/step0/step0_ok.json; part P recorded no sha256, "
                               "part X matched its PPL record bit for bit)", identical=True,
                   **{k: x for k, x in info.items() if k not in ("config", "source")}, config=info["config"],
                   source_meta=info["source"]),
        license_files=lic, files=files, scan=hits, ppl=fsum["ppl"], paired_vs_fourover6=fsum["paired_vs_fourover6"],
        calibration_cost=cs)
    print(f"{FOCUS_REPO}: {list(files)} scan hits {len(hits)}", flush=True)


def stage_gptq(model, record):
    repo = gptq_repo(model)
    dst = GPTQ_STAGE / repo.split("/")[1]
    dst.mkdir(parents=True, exist_ok=True)
    v = verify_codes(model)
    if not v["ok"]:
        sys.exit(f"{model}: the codes file fails its checks: {json.dumps({k: v[k] for k in ('key_differences', 'main_table_variant', 'fmt_values', 'decode_snippet_bitwise')})}")
    out = dst / CODES_NAME
    if not out.exists():
        subprocess.run(["cp", "--reflink=always", v["file"], str(out)], check=True)
    sha_out = FH.sha(out)
    assert sha_out == v["file_sha256"], f"{model}: staged codes differ from the source"
    hits = scan_codes(out)
    ev = gptq_evaluated(model)
    assert ev["record"]["ptq"]["codes_sha256"] == v["codes_sha256"]
    lic = license_files(model, dst)
    summ = gptq_summary(model, v, ev)
    (dst / "ppl_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n")
    pre = {f.name: dict(bytes=f.stat().st_size, sha256=sha_out if f.name == CODES_NAME else FH.sha(f))
           for f in sorted(dst.iterdir()) if f.is_file()}
    cfiles = {CODES_NAME: pre[CODES_NAME], "ppl_summary.json": pre["ppl_summary.json"],
              **{FILE_NAME[k]: pre[FILE_NAME[k]] for k in LICENSES[model]["files"]}}
    (dst / "README.md").write_text(gptq_card(model, v, ev, cfiles))
    for f in sorted(dst.iterdir()):
        if f.is_file() and f.name != CODES_NAME:
            hits += FH.scan_text(f.name, f.read_text(), ref=f.name in ("README.md", "ppl_summary.json"))
    files = {f.name: dict(bytes=f.stat().st_size, sha256=sha_out if f.name == CODES_NAME else FH.sha(f))
             for f in sorted(dst.iterdir()) if f.is_file()}
    record["repos"][repo] = dict(kind="gptq", model=model, base_model=MODELS[model][1], revision=MODELS[model][2],
                                 folder=str(dst), license=LICENSES[model]["front"], codes={k: x for k, x in v.items() if k != "key"},
                                 key=v["key"], license_files=lic, files=files, scan=hits, ppl=summ["ppl"],
                                 paired_vs_fourover6=summ["paired_vs_fourover6"])
    print(f"{repo}: {list(files)} scan hits {len(hits)} ({len(unaccepted(hits))} not accepted)", flush=True)


def stage():
    RECS.mkdir(parents=True, exist_ok=True)
    path = RECS / "stage.json"
    record = json.loads(path.read_text()) if path.exists() else dict(namespace=NS, repos={})
    record.update(staged_utc=now(), licenses_fetched=json.loads((LIC32 / "licenses.json").read_text()))
    if FOCUS_REPO not in record["repos"]:
        stage_focus(record)
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n")
    for model in MODELS:
        if gptq_repo(model) in record["repos"]:
            continue
        stage_gptq(model, record)
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n")


def rename():
    """The user's naming decision: re-key the staged GPTQ repositories to <Base>-GPTQ-BF16Calib-NVFP4 in the staging
    records, keeping the verified staged folders (no new copies), regenerate each GPTQ card, rescan the text files and
    refresh the file records. The codes files are unchanged (same folders, same bytes)."""
    path = RECS / "stage.json"
    st = json.loads(path.read_text())
    hits = json.loads((RECS / "scan_hits.json").read_text())
    nonhf = json.loads((RECS / "nonhf_hits.json").read_text())
    renamed = {}
    for model in MODELS:
        old, new = old_gptq_repo(model), gptq_repo(model)
        if new in st["repos"]:
            continue
        rec = st["repos"].pop(old)
        dst = Path(rec["folder"])
        v = dict(rec["codes"], key=rec["key"])
        assert FH.sha(dst / CODES_NAME) == v["file_sha256"], f"{model}: staged codes changed"
        ev = gptq_evaluated(model)
        files = {n: dict(bytes=(dst / n).stat().st_size, sha256=v["file_sha256"] if n == CODES_NAME else FH.sha(dst / n))
                 for n in [CODES_NAME, "ppl_summary.json"] + [FILE_NAME[k] for k in LICENSES[model]["files"]]}
        (dst / "README.md").write_text(gptq_card(model, v, ev, files))
        rec["scan"] = [h for h in rec["scan"] if h["file"] != "README.md"] + \
            FH.scan_text("README.md", (dst / "README.md").read_text(), ref=True)
        rec["files"] = {f.name: dict(bytes=f.stat().st_size, sha256=v["file_sha256"] if f.name == CODES_NAME else FH.sha(f))
                        for f in sorted(dst.iterdir()) if f.is_file()}
        rec["renamed_from"] = old
        st["repos"][new] = rec
        if old in hits:
            hits[new] = hits.pop(old)
        for h in nonhf:
            if h["repository"] == old:
                h["repository"] = new
        renamed[old] = new
        print(f"{old} -> {new}: README regenerated, text scan hits {len([h for h in rec['scan'] if h['file'] == 'README.md'])}")
    st["renamed_utc"] = now()
    path.write_text(json.dumps(st, indent=1, ensure_ascii=False, default=str) + "\n")
    (RECS / "scan_hits.json").write_text(json.dumps(hits, indent=1) + "\n")
    (RECS / "nonhf_hits.json").write_text(json.dumps(nonhf, indent=1) + "\n")
    held = json.loads((RECS / "held.json").read_text())
    held.update(decided_utc=now(), decision="the user's naming: <Base>-GPTQ-BF16Calib-NVFP4 for all eight", renamed=renamed)
    (RECS / "held.json").write_text(json.dumps(held, indent=1) + "\n")


# ----------------------------------------------------------------------------------- Llama-3.2-1B-Instruct FlipQuant
# The user's grant (2026-10-15, relayed by the coordinator): part P's three Llama-3.2-1B-Instruct maps (4a64caf) as the
# PRIVATE repository edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant. It takes the layout, file names and card structure of
# edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant: llama_hf's stage_map / fq_summary / fq_card, bound to this model (they
# regenerate the 8B card and summary byte for byte). The license files are the 1B FOCUS repository's Llama 3.2 files.
FQ1B_MODEL = "llama3.2-1b-instruct"
FQ1B_REPO = f"{NS}/Llama-3.2-1B-Instruct-FlipQuant"
FQ1B_STAGE = OLD / FQ1B_REPO.split("/")[1]
FQ1B_RECORDS = T.ROOT / "results" / "paper_eval" / "llama_instruct" / "records" / FQ1B_MODEL   # <unit>/run.json
FQ1B_WINDOWS = ("Its first 128 windows are identical, token for token, to the fit windows of the Llama-3.1-8B release "
                "(Llama-3.1-8B-FlipQuant): the paper rule (`builder_seed0` with this model's tokenizer) gives the same 128 "
                "windows. Its 128 extension windows share none with that release's: this model's own development draws "
                "contain all 128 of that release's extension documents, which the extension rule skips.")


def fq1b_record_prep_seconds():
    """The 1B's one-time data preparation (part P amendment 1's record step), from queue.log's START / END lines."""
    lines = (G.RUN / "queue.log").read_text().splitlines()
    t = [l.split()[0] for l in lines
         if f"START llama_record_{FQ1B_MODEL} " in l or f"END llama_record_{FQ1B_MODEL} rc=0" in l]
    assert len(t) == 2, t
    a, b = (datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ") for x in t)
    return (b - a).total_seconds()


@contextlib.contextmanager
def llama_1b_flipquant():
    """llama_hf's FlipQuant functions bound to Llama-3.2-1B-Instruct; its globals are restored afterwards."""
    title, hf_id, rev = MODELS[FQ1B_MODEL][:3]
    d = LH.P.ROOT / FQ1B_MODEL
    bind = dict(MODEL=FQ1B_MODEL, TITLE=title, HF_ID=hf_id, REV=rev, FQ_REPO=FQ1B_REPO, FQ_STAGE=FQ1B_STAGE, D=d,
                RUN_PREFIX=str(d) + "/", LLAMA_NOTICE=L32_NOTICE,
                LLAMA_TERMS=llama_license_text(FQ1B_MODEL, "terms").split("\n\n")[0], SAME_WINDOWS=FQ1B_WINDOWS,
                record_prep_seconds=fq1b_record_prep_seconds)
    old = {k: getattr(LH, k) for k in bind}
    for k, v in bind.items():
        setattr(LH, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(LH, k, v)


def fq1b_card(staged, summary):
    """llama_hf.fq_card for this model, with its Llama 3.1 license strings replaced by Llama 3.2's (as in the 1B FOCUS
    card)."""
    title, hf_id, rev = MODELS[FQ1B_MODEL][:3]
    with llama_1b_flipquant():
        text = LH.fq_card(staged, summary)
    for a, b in (("license: llama3.1\n", "license: llama3.2\n"),
                 ("the same license as the base model, the Llama 3.1 Community License.",
                  "the same license as the base model, the Llama 3.2 Community License."),
                 (f"- `LICENSE`: the Llama 3.1 Community License Agreement, the base repository's own `LICENSE` file "
                  f"({hf_id} at revision `{rev}`).",
                  f"- `LICENSE`: the Llama 3.2 Community License Agreement, the base repository's own license file "
                  f"({hf_id} at revision `{rev}`)."),
                 ("- `USE_POLICY.md`: the Llama 3.1 Acceptable Use Policy,",
                  "- `USE_POLICY.md`: the Llama 3.2 Acceptable Use Policy,")):
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    assert "Llama 3.1" not in text and "llama3.1" not in text, "a Llama 3.1 string is left in the 1B card"
    return text


def stage_fq1b():
    """Verify part P's three maps against its records, then stage:
    - the maps (llama_hf.stage_map: meta cleaned, tiles and tensor records bit-identical);
    - the card, ppl_summary.json and the Llama 3.2 license files.
    Scan everything, and add the repository to stage.json; no other entry changes."""
    sys.path.insert(0, str(LH.P.S.FQ2))
    path = RECS / "stage.json"
    record = json.loads(path.read_text())
    if FQ1B_REPO in record["repos"]:
        sys.exit(f"{FQ1B_REPO} is already staged")
    want = {u: json.loads((FQ1B_RECORDS / u / "run.json").read_text())["map_sha256"] for u in LH.P.UNITS}
    final = json.loads((LH.OUT.parent / "final" / "final.json").read_text())
    FQ1B_STAGE.mkdir(parents=True, exist_ok=True)
    with llama_1b_flipquant():
        got = {u: FH.sha(LH.D / "maps" / f"flipquant_{u}.pt") for u in LH.P.UNITS}
        if got != want:
            sys.exit(f"{FQ1B_MODEL}: the maps differ from part P's records: {got} vs {want}")
        staged = {u: LH.stage_map(u) for u in LH.P.UNITS}
        summary = LH.fq_summary(staged, final)
    assert all(staged[u]["sha256_release"] == want[u] for u in want)
    lic = license_files(FQ1B_MODEL, FQ1B_STAGE)
    (FQ1B_STAGE / "ppl_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (FQ1B_STAGE / "README.md").write_text(fq1b_card(staged, summary))
    hits = []
    for f in sorted(FQ1B_STAGE.iterdir()):
        if f.is_file():
            hits += LH.scan_map(f) if f.suffix == ".pt" else \
                FH.scan_text(f.name, f.read_text(), ref=f.name in ("README.md", "ppl_summary.json"))
    files = {f.name: dict(sha256=FH.sha(f), bytes=f.stat().st_size) for f in sorted(FQ1B_STAGE.iterdir()) if f.is_file()}
    title, hf_id, rev = MODELS[FQ1B_MODEL][:3]
    record["repos"][FQ1B_REPO] = dict(
        kind="flipquant", model=FQ1B_MODEL, base_model=hf_id, revision=rev, folder=str(FQ1B_STAGE),
        license=LICENSES[FQ1B_MODEL]["front"], staged_utc=now(),
        maps={u: {k: v for k, v in s.items() if k != "meta"} for u, s in staged.items()},
        recorded_map_sha256=want, recorded_by=("part P amendment 1 (4a64caf): results/paper_eval/llama_instruct/records/"
                                               f"{FQ1B_MODEL}/<unit>/run.json, map_sha256"),
        license_files=lic, files=files, scan=hits, ppl=summary["ppl"], paired_vs_fourover6=summary["paired_vs_fourover6"])
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n")
    print(f"{FQ1B_REPO}: {list(files)} scan hits {len(hits)} ({len(unaccepted(hits))} not accepted)", flush=True)
    for h in unaccepted(hits):
        print("  not accepted:", h["file"], h["where"], pattern_class(h["match"]), flush=True)


# ----------------------------------------------------------------------------------------------------- check / upload
# The coordinator's rule (2026-10-15) for raw-byte hits in the GPTQ codes files: a hit is accepted iff (1) it matches
# only the hf_ token pattern, (2) it lies entirely inside a tensor-storage member that backs a codes tensor (q or s),
# (3) the file's pickle, metadata strings and record names have no hit, and (4) the file's sha256 equals its record (the
# file sha256 part X recorded; for the two files without one, the code digest the main table recorded). Text files
# (README, json, LICENSE, NOTICE, USE_POLICY) may have no hit but the accepted Llama reporting address, and every staged
# file must pass the exact-token check (hf_token_check.py) with zero occurrences.
ORDER_LAST = ("qwen3-1.7b", "qwen3-8b")         # no earlier file sha256: rule (4) by the code digest; uploaded last


PATTERN_CLASSES = [(r"^hf_[A-Za-z0-9]{20,}$", "token pattern"), (r"^/(home|tmp)/$", "path pattern"),
                   (r"^n16k64_campaign$", "path pattern"), (r"^(brian030128|chenjiaj|u4320956)$", "username pattern"),
                   (r"^razer$", "project-name pattern"), (r"^angelslim$", "reference-implementation pattern")]


def pattern_class(match):
    """The class of a raw-byte scan match (focus_hf.RAW's alternatives), so that records never quote it."""
    for rx, name in PATTERN_CLASSES:
        if re.match(rx, match, re.I):
            return name
    return "other pattern"


def sanitized(obj):
    """A JSON record with every raw-byte match replaced by its pattern class and length (records committed to git)."""
    if isinstance(obj, dict):
        out = {k: sanitized(v) for k, v in obj.items()}
        # a raw-byte hit: a scan record ("where": "raw bytes @...") or an example inside an archive member
        raw = str(obj.get("where", "")).startswith("raw bytes") or ("member" in obj and "offset" in obj)
        if isinstance(obj.get("match"), str) and raw:
            out["match"] = f"<{pattern_class(obj['match'])}, {len(obj['match'])} bytes>"
            if "text" in out:
                out["text"] = ""
        if "patterns" in out and isinstance(out["patterns"], list):
            out["patterns"] = sorted({p if p == "hf_ token pattern" else pattern_class(p) for p in obj["patterns"]})
        if "match_characters_after_prefix" in out:
            out["match_characters_after_prefix"] = "<omitted>"
        return out
    if isinstance(obj, list):
        return [sanitized(v) for v in obj]
    return obj


def rule_check():
    st = json.loads((RECS / "stage.json").read_text())
    hits_rec = json.loads((RECS / "scan_hits.json").read_text())
    tok = json.loads((RECS / "token_check.json").read_text())
    out = {}
    for repo, d in st["repos"].items():
        problems, binary = [], {}
        text_hits = [h for h in d["scan"] if not h["file"].endswith(".pt")]
        if unaccepted(text_hits):
            problems.append(f"text-file hits {unaccepted(text_hits)[:3]}")
        pt_hits = [h for h in d["scan"] if h["file"].endswith(".pt")]
        if d["kind"] == "focus":
            if pt_hits:
                problems.append(f"focus.pt hits {pt_hits[:3]}")
        elif d["kind"] == "flipquant":
            if pt_hits:
                problems.append(f"map-file hits {[(h['file'], h['where']) for h in pt_hits[:3]]}")
            binary = dict(maps_equal_part_p={u: m["sha256_release"] == d["recorded_map_sha256"][u]
                                             for u, m in d["maps"].items()})
            if not all(binary["maps_equal_part_p"].values()):
                problems.append("a map differs from part P's record")
        else:
            a = hits_rec.get(repo)
            c = d["codes"]
            raw = [h for h in pt_hits if h["where"].startswith("raw bytes @")]
            meta = [h for h in pt_hits if not h["where"].startswith("raw bytes @")]
            binary = dict(raw_hits=len(raw), by_field=(a or {}).get("by_field"),
                          r1_only_hf_pattern=all(re.fullmatch(r"hf_[A-Za-z0-9]{20,}", h["match"], re.I) for h in raw),
                          other_patterns=sorted({h["match"] for h in raw
                                                 if not re.fullmatch(r"hf_[A-Za-z0-9]{20,}", h["match"], re.I)}),
                          r2_inside_q_or_s=bool(a) and a["raw_hits"] == len(raw) and a["all_inside_tensor_storage"]
                          and set(a["by_field"]) <= {"q", "s"},
                          r3_no_metadata_hits=not meta and not [h for h in (a or {}).get("other_hits", [])
                                                                if h["file"].endswith(".pt")],
                          r4_sha256_record=(c["file_sha256"] == c["recorded_file_sha256"]) if c["recorded_file_sha256"]
                          else c["codes_sha256"] == c["record_codes_sha256"],
                          r4_basis="file sha256 (part X)" if c["recorded_file_sha256"] else "code digest (main table)")
            # the coordinator's extension (2026-10-15): (1) may cover any scan pattern when (2), (3) and the exact-token
            # check all hold
            binary["other_patterns"] = sorted({pattern_class(m) for m in binary["other_patterns"]})
            for k in ("r2_inside_q_or_s", "r3_no_metadata_hits", "r4_sha256_record"):
                if not binary[k]:
                    problems.append(f"rule {k} fails")
        tk = {n: v for n, v in tok["files"].items() if n.startswith(repo + "/")}
        if len(tk) != len(d["files"]) or any(v["exact_token_hits"] or v["prefix_hits"] for v in tk.values()):
            problems.append("exact-token check")
        out[repo] = dict(binary=binary, exact_token_check=f"{sum(v['exact_token_hits'] + v['prefix_hits'] for v in tk.values())} hits",
                         problems=problems)
    return out


def check():
    api = FH.account()
    st = json.loads((RECS / "stage.json").read_text())
    assert len(st["repos"]) == 10, sorted(st["repos"])
    ours = set(json.loads((RECS / "hf_upload.json").read_text())["repos"]) if (RECS / "hf_upload.json").exists() else set()
    exists = {r: api.repo_exists(r, repo_type="model") for r in st["repos"]}
    print("existing, created by this upload:", sorted(r for r, e in exists.items() if e and r in ours) or "none")
    foreign = sorted(r for r, e in exists.items() if e and r not in ours)
    print("existing, not created by this upload:", foreign or "none")
    if foreign:
        sys.exit("a target repository already exists; stopping (nothing is overwritten)")
    rc = rule_check()
    (RECS / "rule_check.json").write_text(json.dumps(rc, indent=1) + "\n")
    for r, v in rc.items():
        print(r, "ok" if not v["problems"] else v["problems"], v["exact_token_check"], v["binary"].get("by_field"),
              v["binary"].get("other_patterns") or "")
    print("within the rule:", [r for r, v in rc.items() if not v["problems"]])
    print("outside the rule (not uploaded):", [r for r, v in rc.items() if v["problems"]])


def upload(only=None):
    """Upload the staged repositories, or only those named in `only`. A repository outside the rule, or one that exists on
    the Hub without this upload having created it, is never uploaded (reported instead)."""
    from huggingface_hub import hf_hub_download
    api = FH.account()
    st = json.loads((RECS / "stage.json").read_text())
    rc = rule_check()
    path = RECS / "hf_upload.json"
    out = json.loads(path.read_text()) if path.exists() else dict(account=NS, repos={})
    foreign = {r for r in st["repos"] if r not in out["repos"] and api.repo_exists(r, repo_type="model")}
    # a repository an interrupted run of this upload created: private, and exactly the staged files with the staged LFS
    # sha256 on the Hub -> resumed (verified and recorded, not uploaded again)
    resumed = set()
    for r in sorted(foreign):
        rec = st["repos"][r]
        info = api.model_info(r, files_metadata=True)
        hub = {x.rfilename: x for x in info.siblings}
        def lfs_sha(x):
            lf = getattr(x, "lfs", None)
            return (lf.get("sha256") if isinstance(lf, dict) else getattr(lf, "sha256", None)) if lf else None
        lfs = {n: lfs_sha(hub.get(n)) for n in rec["files"]}
        if info.private is True and set(hub) - {".gitattributes"} == set(rec["files"]) and \
                all(lfs[n] in (None, rec["files"][n]["sha256"]) for n in rec["files"]) and \
                lfs.get(CODES_NAME, lfs.get("focus.pt")) == rec["files"].get(CODES_NAME, rec["files"].get("focus.pt", {})).get("sha256"):
            resumed.add(r)
    foreign -= resumed
    skip = {r for r, v in rc.items() if v["problems"]} | foreign
    if foreign:
        print(f"existing repositories this upload did not create (never touched): {sorted(foreign)}")
    order = sorted(st["repos"], key=lambda r: (st["repos"][r]["kind"] != "focus", st["repos"][r]["model"] in ORDER_LAST))
    if only:
        assert set(only) <= set(st["repos"]), only
        order = [r for r in order if r in only]
    for repo_id in order:
        rec = st["repos"][repo_id]
        if repo_id in skip:
            why = "exists on the Hub (not ours)" if repo_id in foreign else f"outside the rule: {rc[repo_id]['problems']}"
            print(f"{repo_id}: not uploaded, {why}")
            continue
        if out["repos"].get(repo_id, {}).get("verified"):
            print(f"{repo_id}: already uploaded and verified")
            continue
        folder = Path(rec["folder"])
        names = sorted(rec["files"])
        assert {f.name for f in folder.iterdir() if f.is_file()} == set(names), f"{repo_id}: staged files changed"
        for n in names:
            assert (folder / n).stat().st_size == rec["files"][n]["bytes"], f"{repo_id}: {n} changed size"
        if repo_id not in resumed:
            if not api.repo_exists(repo_id, repo_type="model"):
                api.create_repo(repo_id, repo_type="model", private=True, exist_ok=False)
            info = api.model_info(repo_id)
            anon = FH.anonymous_status(repo_id)
            if info.private is not True or anon == 200:
                sys.exit(f"{repo_id}: not private (private={info.private}, anonymous HTTP {anon}); stopping")
            api.upload_large_folder(repo_id=repo_id, folder_path=folder, repo_type="model", private=True,
                                    allow_patterns=names, print_report_every=300)
        else:
            print(f"{repo_id}: resumed (created and uploaded by an interrupted run; verifying)")
        info = api.model_info(repo_id, files_metadata=True)
        anon_after = FH.anonymous_status(repo_id)
        hub = {s.rfilename: s for s in info.siblings}
        problems = []
        if info.private is not True or anon_after == 200:
            problems.append(f"not private (private={info.private}, anonymous HTTP {anon_after})")
        if set(hub) - {".gitattributes"} != set(names):
            problems.append(f"file list {sorted(hub)}")
        card_data = info.card_data.to_dict() if info.card_data else {}
        want_front = dict(rec["license"], base_model=rec["base_model"])
        if any(card_data.get(k) != x for k, x in want_front.items()):
            problems.append(f"card metadata {card_data}")
        files = {}
        for n in names:
            want = rec["files"][n]["sha256"]
            lfs = getattr(hub.get(n), "lfs", None)
            lfs_sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            size = rec["files"][n]["bytes"]
            if lfs_sha is not None and size > LFS_ONLY_ABOVE:
                # the coordinator's rule: a file above a few GB is verified by the Hub's server-side LFS sha256 (file
                # metadata), not downloaded
                got, method = None, "hub LFS sha256"
                ok = lfs_sha == want and getattr(hub.get(n), "size", None) == size
            else:
                with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:     # small files only, one at a time, in RAM
                    p = hf_hub_download(repo_id, n, revision=info.sha, cache_dir=tmp, force_download=True)
                    got = FH.sha(p)
                method = "download sha256"
                ok = got == want and (lfs_sha is None or lfs_sha == want)
            files[n] = dict(sha256=want, bytes=size, hub_size=getattr(hub.get(n), "size", None), hub_lfs_sha256=lfs_sha,
                            download_sha256=got, method=method, equal=ok,
                            lfs_equal=None if lfs_sha is None else lfs_sha == want)
            if not ok:
                problems.append(f"{n}: sha256 differs ({method}: download {got}, lfs {lfs_sha})")
            print(f"  {repo_id} {n}: {method} {'==' if ok else '!='} staged", flush=True)
        commits = api.list_repo_commits(repo_id)
        out["repos"][repo_id] = dict(url=f"https://huggingface.co/{repo_id}", head=info.sha, private=info.private,
                                     resumed=repo_id in resumed,
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
                            renders=valid is True and card.content == staged and n_tables == html.count("<table>") and
                            n_fences == html.count("<pre><code") and html.count("<h1>") == 1)
        print(repo_id, {k: x for k, x in out[repo_id].items() if k != "card_data"}, flush=True)
    (RECS / "cards.json").write_text(json.dumps(out, indent=1) + "\n")


# ------------------------------------------------------------------------------------------------------------ record
# the user's visibility decision (2026-10-15, relayed by the coordinator), recorded and checked read-only: in this session
# the change itself was refused by the permission system ("Create Public Surface"), so the user makes it
VISIBILITY_DECISION = [FQ1B_REPO, FOCUS_REPO, *[gptq_repo(m) for m in MODELS], LH.FQ_REPO, LH.FOCUS_REPO]


def visibility_now():
    """Read-only, without a token: each VISIBILITY_DECISION repository's anonymous API status now (401: private, 200:
    public). For a public one, also check:
    - its model info;
    - its file list against the staged list;
    - its README (downloaded without a token) against the staged README;
    - its head against the verified upload's.
    The user makes the visibility change; this session only checks it."""
    from huggingface_hub import HfApi, hf_hub_download
    stages = [json.loads((r / "stage.json").read_text())["repos"] for r in (RECS, LH.RECS)]
    uploads = [json.loads((r / "hf_upload.json").read_text())["repos"] for r in (RECS, LH.RECS)]
    anon = HfApi(token=False)
    out = {}
    for r in VISIBILITY_DECISION:
        rec = next(s[r] for s in stages if r in s)
        ur = next((u[r] for u in uploads if r in u), {})
        code = FH.anonymous_status(r)
        row = dict(anonymous_http=code, visibility={401: "private", 200: "public"}.get(code, f"HTTP {code}"),
                   checked_utc=now(), upload_verified=ur.get("verified"), verified_head=ur.get("head"))
        if code == 200:
            info = anon.model_info(r)
            files = sorted(x.rfilename for x in info.siblings)
            with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
                p = hf_hub_download(r, "README.md", revision=info.sha, cache_dir=tmp, token=False, force_download=True)
                readme = FH.sha(p)
            row.update(private=info.private, model_info_readable=True, head=info.sha, files=files,
                       head_equal_verified=info.sha == ur.get("head"),
                       file_list_equal_staged=set(files) - {".gitattributes"} == set(rec["files"]),
                       readme_readable=True, readme_equal_staged=readme == rec["files"]["README.md"]["sha256"])
            row["public_checks_pass"] = row["private"] is False and row["file_list_equal_staged"]
        out[r] = row
        print(r, {k: v for k, v in row.items() if k not in ("files",)}, flush=True)
    (RECS / "visibility.json").write_text(json.dumps(dict(checked_utc=now(), repos=out), indent=1) + "\n")
    return out


ADDENDUM_8B = "## Addendum (2026-10-15): visibility"


def addendum_8b(vis, checked):
    """llama_instruct/HF_UPLOAD.md's addendum for part P's two repositories: the user's visibility decision and this
    session's read-only checks (replaced, not repeated, when written again)."""
    path = LH.OUT / "HF_UPLOAD.md"
    text = path.read_text()
    if ADDENDUM_8B in text:
        text = text[:text.index(ADDENDUM_8B)].rstrip("\n") + "\n"
    L = ["", ADDENDUM_8B, "",
         "The user's decision (2026-10-15, relayed by the coordinator) was to make both repositories public. The user "
         "made the change. The implementation session did not: its permission system refused to.", "",
         f"Read-only checks without a token, at {checked}; record in `../hf_gptq_focus/visibility.json`:", "",
         "| repository | visibility | model info readable | file list = staged | README = staged | head = verified upload |",
         "|---|---|---|---|---|---|"]
    for r in (LH.FQ_REPO, LH.FOCUS_REPO):
        v = vis[r]
        L.append(f"| `{r}` | {v['visibility']} | {v.get('model_info_readable', '–')} | {v.get('file_list_equal_staged', '–')} | "
                 f"{v.get('readme_equal_staged', '–')} | {v.get('head_equal_verified', '–')} |")
    path.write_text(text + "\n".join(L) + "\n")


def record():
    st = json.loads((RECS / "stage.json").read_text())
    up = json.loads((RECS / "hf_upload.json").read_text())
    cj = json.loads((RECS / "cards.json").read_text())
    rc = rule_check()                                     # from the current records (stage, scans, token check)
    (RECS / "rule_check.json").write_text(json.dumps(rc, indent=1) + "\n")
    tok = json.loads((RECS / "token_check.json").read_text())
    nonhf = json.loads((RECS / "nonhf_hits.json").read_text())
    hits = json.loads((RECS / "scan_hits.json").read_text())
    held = json.loads((RECS / "held.json").read_text()) if (RECS / "held.json").exists() else {}
    up8 = json.loads((LH.RECS / "hf_upload.json").read_text())
    vis = visibility_now()
    checked = json.loads((RECS / "visibility.json").read_text())["checked_utc"]
    addendum_8b(vis, checked)
    OUT.mkdir(parents=True, exist_ok=True)
    # the committed copies never quote a raw-byte match (pattern classes instead)
    for name, obj in (("stage.json", st), ("scan_hits.json", hits)):
        (OUT / name).write_text(json.dumps(sanitized(obj), indent=1, ensure_ascii=False, default=str) + "\n")
    for name in ("hf_upload.json", "cards.json", "rule_check.json", "token_check.json", "nonhf_hits.json", "visibility.json"):
        shutil.copyfile(RECS / name, OUT / name)
    if held:
        shutil.copyfile(RECS / "held.json", OUT / "held.json")
    shutil.copyfile(LIC32 / "licenses.json", OUT / "licenses_llama32.json")
    for repo_id, d in st["repos"].items():
        c = OUT / "cards" / repo_id.split("/", 1)[1]
        c.mkdir(parents=True, exist_ok=True)
        for f in ("README.md", "ppl_summary.json"):
            shutil.copyfile(Path(d["folder"]) / f, c / f)

    def method(f):
        return f.get("method", "download sha256")          # the first three entries predate the per-file method field

    def state(r):
        return vis[r]["visibility"]

    L = ["# Hugging Face upload: the Llama-3.2-1B-Instruct FOCUS state and FlipQuant maps, and the main-table GPTQ‡ weights "
         "(2026-10-15)", "",
         "The user's requests (relayed by the coordinator, 2026-10-15): nine **private** repositories under `edgeai-lab`, "
         "and later the same day a tenth, `edgeai-lab/Llama-3.2-1B-Instruct-FlipQuant`.",
         "- **Account:** the active account was checked first (`edgeai-lab`); no token was printed or changed.",
         "- **Order of operations:** each repository was created private and checked private (anonymous HTTP 401) before "
         "any file went up. It was then uploaded and verified.",
         "- **Verification:** every file's sha256 was compared with the staged file's. Each file's method is listed "
         "below.",
         "  - The first four repositories (the FOCUS state, Mistral, Nemotron, Phi-4): every file was downloaded again "
         "from the Hub at the head, the codes files included.",
         "  - From Qwen3.8-27B on: files up to 1 GiB were downloaded again (and checked against the Hub's LFS sha256 "
         "where it has one). Larger files were checked against the Hub's server-side LFS sha256 and size, without a "
         "download (the coordinator's rule).",
         "  - The first process downloaded into /dev/shm (8 GiB), which cannot hold Qwen3.8-27B's 13.7 GB codes file, and "
         "it stopped there after the upload. The rerun found that repository private, with exactly the staged files and "
         "LFS sha256, and verified it as above.",
         "- **Existing repositories:** none was touched.", "",
         "## Uploaded", "",
         f"Visibility: at {checked}, without a token (see Visibility).", "",
         "| repository | kind | files | sha256 = staged (method) | visibility now | anonymous at upload | head | card YAML "
         "(Hub) | README = staged |", "|---|---|---:|---|---|---:|---|---|---|"]
    for repo_id, d in up["repos"].items():
        ok = all(f["equal"] for f in d["files"].values())
        n_dl = sum(method(f) == "download sha256" for f in d["files"].values())
        n_lfs = len(d["files"]) - n_dl
        how = f"{n_dl} downloaded" + (f", {n_lfs} by Hub LFS sha256" if n_lfs else "")
        L.append(f"| `{repo_id}` | {st['repos'][repo_id]['kind']} | {len(d['files'])} | "
                 f"{'equal' if ok else 'DIFFERS'} ({how}) | {state(repo_id)} | {d['anonymous_http']} | "
                 f"`{d['head'][:12]}` | {cj[repo_id]['hub_yaml_valid']} | {cj[repo_id]['readme_equal_to_staged']} |")
    not_up = [r for r in st["repos"] if r not in up["repos"]]
    if not_up:
        L += ["", "## Staged, not uploaded", ""]
        L += [f"- `{r}`" for r in not_up]
    L += ["", "## Verification per file", "",
          "| repository | file | bytes | method | sha256 = staged |", "|---|---|---:|---|---|"]
    for repo_id, d in up["repos"].items():
        for n, f in sorted(d["files"].items()):
            L.append(f"| `{repo_id.split('/')[1]}` | `{n}` | {f['bytes']:,} | {method(f)} | {f['equal']} |")
    if held:
        L += ["", "## Naming of the GPTQ repositories", "",
              "The eight GPTQ repositories were first staged under the requested names `<Base>-GPTQ-NVFP4`. Two of those "
              "names already existed under `edgeai-lab`, created by another upload. That upload was not touched; these "
              "facts were read without changing anything:", ""]
        L += [f"- `{e['repo']}`: created {e['created']}, {e['visibility']}; {e['what']}." for e in held.get("existing", [])]
        L += ["", f"The uploads were held, and no GPTQ repository was created, until {held['decision']} "
              f"({held['decided_utc']}). The cards of the two models with an existing repository say in one sentence that "
              "this one is not the same artifact: it is calibrated with BF16 activations (the main table's GPTQ‡), that "
              "one with FourOverSix activations. The staged files were kept, only the records were renamed, and each card "
              "was regenerated and rescanned."]
    fo = st["repos"][FOCUS_REPO]
    L += ["", "## The FOCUS state", "",
          f"**`{FOCUS_REPO}`** holds `focus.pt` as is: part P's state behind tab:main-ppl and parts X / Y.",
          f"- sha256 `{fo['state']['sha256_staged']}`, equal to the value part X's step 0 recorded. Part P recorded no "
          "sha256; part X matched its PPL record bit for bit.",
          "- The layout and card of `edgeai-lab/Llama-3.1-8B-Instruct-FOCUS-NVFP4`: `focus.pt`; the card (our reproduction "
          "of FOCUS, arXiv:2608.01847; not an official release); `ppl_summary.json`, `LICENSE`, `USE_POLICY.md`, "
          "`NOTICE`."]
    if FQ1B_REPO in st["repos"]:
        fq = st["repos"][FQ1B_REPO]
        L += ["", "## The FlipQuant maps (Llama-3.2-1B-Instruct)", "",
              f"**`{FQ1B_REPO}`** holds part P's three Llama-3.2-1B-Instruct maps (amendment 1, 4a64caf).",
              "- **Before staging:** each map's sha256 was checked against part P's record (`map_sha256` in "
              "`records/llama3.2-1b-instruct/<unit>/run.json`); all three are equal.",
              "- **Meta:** cleaned as for `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant`. Local paths are made relative, "
              "the GitHub owner is removed from the extension reference, and the reference implementation's branch and "
              "commit are removed. The tiles and every tensor record are bit-identical.",
              "- **Layout, file names and card:** those of `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant`. They come from "
              "`llama_hf.py`'s staging, summary and card functions, bound to this model; those functions regenerate the "
              "8B card and `ppl_summary.json` byte for byte. Only the model's own facts differ:",
              "  - its records, windows, costs and perplexities;",
              "  - the Llama 3.2 license strings and files, as in the 1B FOCUS card.",
              "- **The windows sentence:** the first 128 fit windows equal the Llama-3.1-8B release's. The 128 extension "
              "windows share none with it: the model's own development draws contain all 128 of that release's extension "
              "documents (part P's window record).", "",
              "| map | part P's record = file before staging | uploaded file (meta cleaned) | E0M3 tiles |",
              "|---|---|---|---:|"]
        for u, m in fq["maps"].items():
            L.append(f"| `{m['file']}` | `{m['sha256_release'][:16]}…` ({m['sha256_release'] == fq['recorded_map_sha256'][u]}) | "
                     f"`{m['sha256_uploaded'][:16]}…` | {m['e0m3_tiles']:,} of {m['tiles']:,} |")
    L += ["", "## The GPTQ‡ weights", "",
          "One repository per model, holding `codes_nvfp4_act-fo6.pt` as is: the file behind the main table's GPTQ‡ cell.",
          "- **Checked before staging, against the main-table record:**",
          "  - each file's code digest (FlipQuant's `codes_sha256` formula over its stored entries);",
          "  - its settings key.",
          "- **The key against the main-table variant:**",
          "  - BF16-activation calibration: `propagate` = `hessian_input` = `bf16` (round 3's activation-quantized codes "
          "have `quantized`);",
          "  - the release windows (256), block 128, damp 0.01;",
          "  - NVFP4 weights, no map, no rotation.",
          "- **Decoding:** the card's PyTorch snippet reproduces FlipQuant's dequantization bit for bit on each file's first "
          "module.", "",
          "| repository | source file | bytes | file sha256 | rule (4) basis | code digest = record | key = record | "
          "main-table variant | written by FlipQuant |", "|---|---|---:|---|---|---|---|---|---|"]
    for repo_id, d in st["repos"].items():
        if d["kind"] != "gptq":
            continue
        c = d["codes"]
        basis = ("the file sha256 part X recorded" if c["recorded_file_sha256"] else
                 "the code digest equal to the main-table record; the file sha256 is recorded here for the first time")
        L.append(f"| `{repo_id}` | `{c['file'].replace(str(G.RUN), 'RUN')}` | {c['bytes']:,} | `{c['file_sha256'][:16]}…` | "
                 f"{basis} | {c['codes_sha256'] == c['record_codes_sha256']} (`{c['codes_sha256'][:12]}`) | "
                 f"{not c['key_differences']} | {not c['main_table_variant']} | `{MODELS[d['model']][3]}` |")
    L += ["", "**Where the files come from:**",
          "- The two Llama files were regenerated by FlipQuant main `3db0072` in part X's step 0. Part P computed these "
          "codes in-process and kept no file. Their code digests equal part P's records.",
          "- The other six are the main table's own files (parts H / J).", "",
          "## Scans", "",
          "The earlier uploads' patterns cover local paths, the host, user names, private repository names, tokens, e-mail "
          "addresses and pointers to a reference implementation. They were run over every staged file:",
          "- the codes files' settings strings, module names, archive record names and raw bytes;",
          "- the state's metadata, record names and raw bytes;",
          "- the maps' metadata strings, tile names and raw bytes;",
          "- every line of the text files.", "",
          "**The rule for binaries** (the coordinator's, 2026-10-15, with its extension). A raw-byte hit in a GPTQ codes "
          "file is accepted only if all of these hold:", "",
          "1. it matches a scan pattern (the token pattern `hf_…`, or another pattern under the extension below);",
          "2. it lies entirely inside a tensor-storage member (`<archive>/data/<n>`) that backs a codes tensor (`q`, the "
          "packed 4-bit codes, or `s`, the E4M3 block-scale bytes);",
          "3. the file's pickle, metadata strings and record names have zero hits;",
          "4. the file's sha256 equals its record (for Qwen3-1.7B and Qwen3-8B, which have no earlier file sha256: the code "
          "digest equals the main-table record, and the file sha256 is recorded now).", "",
          "**Extension:** (1) may cover any scan pattern, but only when (2), (3) and the exact-token check all hold. Why "
          "the hits occur: the E4M3 scale bytes of these weights fall mostly in 0x58–0x7A, the ASCII letters, so runs of "
          "block scales read as lowercase text, and the packed codes occasionally do too. The hits are such runs.", "",
          "**Other files:** the FOCUS state and the maps may have no hit at all.",
          "- Text files (README, json, LICENSE, NOTICE, USE_POLICY) may have no hit except two accepted kinds, both "
          "accepted before in the Llama uploads:",
          "  - Meta's reporting address `LlamaUseReport@meta.com`, inside the official `USE_POLICY.md` files;",
          "  - in the FlipQuant card, the words “commit” / “branch” in its own sentences (the FlipQuant commit is kept; the "
          "provenance paragraph), as in `edgeai-lab/Llama-3.1-8B-Instruct-FlipQuant`.", "",
          f"**Exact-token check** (the coordinator's condition): every staged file's bytes were searched in-process for "
          f"the {tok['tokens_checked']} Hugging Face tokens present in this machine's token files, and for each one's first "
          f"15 characters. The tokens were never printed, logged or written. Exact-token check: "
          f"{'0 hits on every file' if tok['all_zero'] else 'HITS'} ({len(tok['files'])} files).", "",
          "| repository | raw-byte hits | inside `s` | inside `q` | elsewhere | metadata hits | exact-token check | "
          "within the rule |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for repo_id, d in st["repos"].items():
        if d["kind"] in ("focus", "flipquant"):
            what = "focus.pt" if d["kind"] == "focus" else "map files"
            pt = [h for h in d["scan"] if h["file"].endswith(".pt")]
            raw = [h for h in pt if h["where"].startswith("raw bytes")]
            L.append(f"| `{repo_id}` ({what}) | {len(raw)} | – | – | {len(raw)} | {len(pt) - len(raw)} | "
                     f"{rc[repo_id]['exact_token_check']} | {'yes' if not rc[repo_id]['problems'] else 'NO'} |")
            continue
        a = hits[repo_id]
        bf = a["by_field"]
        L.append(f"| `{repo_id}` | {a['raw_hits']:,} | {bf.get('s', 0):,} | {bf.get('q', 0):,} | "
                 f"{a['raw_hits'] - bf.get('s', 0) - bf.get('q', 0)} | "
                 f"{len([h for h in a['other_hits'] if h['file'].endswith('.pt')])} | {rc[repo_id]['exact_token_check']} | "
                 f"{'yes' if not rc[repo_id]['problems'] else 'NO'} |")
    L += ["", "Hits of a pattern other than the token pattern (accepted under the extension; the matched bytes are not "
          "quoted):", ""]
    L += [f"- `{h['repository'].split('/')[1]}`: a {h['pattern_class']} ({h['length']} bytes) in `{h['tensor']}` "
          f"`{h['field']}`" for h in nonhf]
    L += ["", "## Licenses", "",
          "- **Llama-3.2-1B-Instruct** (FOCUS, FlipQuant and GPTQ):",
          "  - `LICENSE` (the base repository's `LICENSE.txt`) and `USE_POLICY.md`, downloaded read-only with the active "
          "token;",
          "  - `NOTICE`: the agreement's notice sentence verbatim (section 1.b.iii), checked to appear in `LICENSE`;",
          "  - “Built with Llama” is in all three cards, and all three names start with “Llama”.",
          "- **Llama-3.1-8B-Instruct:** the files of part P's upload (the base repository's `LICENSE` and `USE_POLICY.md`; "
          "the base-model release repository's `NOTICE`).",
          "- **Nemotron-Nano-9B-v2:** the NVIDIA Open Model License Agreement and its attribution `NOTICE`, as in the "
          "FlipQuant / FOCUS repositories.",
          "- **Qwen3-1.7B, Qwen3-8B:** the base repositories' Apache-2.0 `LICENSE`. **Phi-4:** its MIT `LICENSE`.",
          "- **Mistral-7B-Instruct-v0.3 and Qwen3.8-27B:** the base repositories have no license file, and their cards "
          "declare Apache-2.0. `LICENSE` is the Apache License 2.0 text from apache.org (as in the Mistral FOCUS "
          "repository).", "",
          "## Visibility", "",
          "**The user's decision** (2026-10-15, relayed by the coordinator): make these twelve repositories public, each "
          "only after its upload is verified, and change no other repository.",
          "- **Who made the change:** the user. This session did not; its permission system refused the change (“Create "
          "Public Surface”), and the coordinator's message said that in that case the user makes it.",
          f"- **Visibility now:** read-only checks without a token at {checked}; record in `visibility.json`.",
          "- **For a public repository,** the checks are:",
          "  - the model info is readable;",
          "  - the file list equals the staged list;",
          "  - the README downloads and equals the staged one;",
          "  - the head equals the verified upload's.",
          "- **The time of each change** is not recorded by the Hub's API. Each repository was public at the time of "
          "the check.", "",
          "| repository | upload verified | visibility | model info readable | file list = staged | README = staged | "
          "head = verified upload |", "|---|---|---|---|---|---|---|"]
    for r in VISIBILITY_DECISION:
        v = vis[r]
        src = "this upload" if r in up["repos"] else ("part P's upload" if r in up8["repos"] else "not uploaded")
        L.append(f"| `{r}` | {v['upload_verified']} ({src}) | {v['visibility']} | {v.get('model_info_readable', '–')} | "
                 f"{v.get('file_list_equal_staged', '–')} | {v.get('readme_equal_staged', '–')} | "
                 f"{v.get('head_equal_verified', '–')} |")
    L += ["", "## Disk", "",
          "Staging uses reflinks (`cp --reflink=always`): the staged and source files map the same physical extents. "
          "However, /home's XFS project quota (`prjquota`) charges each staged copy its full size, so the staging costs "
          "about 37 GiB of quota.",
          "- Nothing was deleted.",
          "- The uploads used an xet cache on /dev/shm.",
          "- The maps' staging is about 3 MB.", ""]
    (OUT / "HF_UPLOAD.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "upload":
        upload(sys.argv[2:] or None)
    else:
        {"licenses": licenses, "stage": stage, "rename": rename, "stage_fq1b": stage_fq1b, "check": check, "cards": cards,
         "record": record}.get(what, lambda: sys.exit(__doc__))()

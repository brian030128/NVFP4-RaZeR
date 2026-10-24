"""Part AB (PROTOCOL.md; the user's decision of 2026-10-23, relayed by the coordinator): the part AA 1,024-window
artifacts of the eight models as 24 PRIVATE repositories under edgeai-lab, in the formats of the earlier uploads (the
release / llama_hf.py FlipQuant cards, focus_hf.py's FOCUS card, hf_gptq_focus.py's GPTQ-BF16Calib card):

- <Base>-FlipQuant-Calib1k: the FQ16-1k map (16x64), its meta cleaned as for the earlier FlipQuant repositories (the tiles
  and every tensor record bit-identical; both sha256 recorded);
- <Base>-FOCUS-NVFP4-Calib1k: the FOCUS-1k state, as is;
- <Base>-GPTQ-BF16Calib-NVFP4-Calib1k: the GPTQ-double-dagger-1k codes, as is.

Every source is verified against its part AA job record (a fresh sha256 of the whole file) before staging. Staging
(/home/dev/calib1k_hf/<repo>/): the card, ppl_summary.json, the license files and the cleaned map are written there; the
FOCUS state and the codes file are symbolic links to their verified sources (/home's 10 GB floor allows no copy), so
every scan, hash and upload reads the source. The cards' numbers are part AA's: results/paper_eval/calib_size/
calib_size.json at paper-eval CS_COMMIT; GPTQ-1k vs FourOverSix, which that record lacks, comes from the same report
code's compare() on the same records.

    python hf_calib1k.py stage [--models=a,b]   # verify, stage, scan; locate the codes files' raw hits: RECS/stage.json
    python hf_calib1k.py tokens                 # the exact-token check of every staged file: RECS/token_check.json
    python hf_calib1k.py check                  # read-only: the account, none of the 24 names exists, every gate
    python hf_calib1k.py upload [repo ...]      # create PRIVATE, check private, upload, verify: RECS/hf_upload.json
    python hf_calib1k.py cards                  # the Hub's YAML validation, README = staged: RECS/cards.json
    python hf_calib1k.py status                 # read-only, without a token: each repository's anonymous status
    python hf_calib1k.py record                 # results/paper_eval/hf_calib1k/ (after the user's visibility change)
    python hf_calib1k.py a1_stage | a1_tokens | a1_check | a1_upload | a1_cards | a1_record   # amendment 1 (PUBLIC)

CPU and network only. The Hugging Face token is never printed, logged or written.
"""
import collections
import copy
import datetime
import functools
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import focus_hf as FH  # noqa: E402
import hf_gptq_focus as H  # noqa: E402
import hf_scan_hits as HS  # noqa: E402
import hf_token_check as TC  # noqa: E402
import llama_hf as LH  # noqa: E402
import report_calib_size as RC  # noqa: E402
import run_calib_size as CS  # noqa: E402
import run_zeroshot6 as Z  # noqa: E402
import tables_cpu as T  # noqa: E402

NS = "edgeai-lab"
STAGE = Path("/home/dev/calib1k_hf")
RECS = STAGE / "records"
OUT = T.ROOT / "results" / "paper_eval" / "hf_calib1k"
CS_COMMIT = "28a0ea2"                                 # paper-eval: part AA's record, amendments 1-5
ORDER = list(CS.ORDER)                                # Llama-3.2-1B-Instruct first
KINDS = ("flipquant", "focus", "gptq")
SUFFIX = {"flipquant": "FlipQuant", "focus": "FOCUS-NVFP4", "gptq": "GPTQ-BF16Calib-NVFP4"}
MAP_NAME = "flipquant_16x64.pt"
CODES_NAME = H.CODES_NAME
LFS_ONLY_ABOVE = 1 << 30           # bytes: larger files are verified by the Hub's LFS sha256 and size, not downloaded
XET_CACHE = Path("/dev/shm/calib1k_xet")
ROW = {"flipquant": "fq16_1024", "focus": "focus_1024", "gptq": "gptq_1024"}
REF256 = {"flipquant": "fq16_256", "focus": "focus256", "gptq": "gptq"}
VS256 = {"flipquant": "FQ16-1k vs FQ16-256", "focus": "FOCUS-1k vs FOCUS-256", "gptq": "GPTQ‡-1k vs GPTQ‡-256"}
VSFO6 = {"flipquant": "FQ16-1k vs FourOverSix", "focus": "FOCUS-1k vs FourOverSix", "gptq": None}
Q27 = "qwen3.8-27b"
TASKS = [("piqa", "PIQA"), ("hellaswag", "HellaSwag"), ("winogrande", "WinoGrande"), ("arc_easy", "ARC-e"),
         ("arc_challenge", "ARC-c"), ("lambada_openai", "LAMBADA")]
# the kinds of text-file hit the earlier uploads accepted: Meta's reporting address in the official USE_POLICY.md, and
# the words "commit" / "branch" in the FlipQuant card's own sentences (the FlipQuant commit is kept; the provenance
# paragraph)
ACCEPTED = [lambda h: h["file"] == "USE_POLICY.md" and h["match"] == "LlamaUseReport@meta.com",
            lambda h: h["file"] == "README.md" and h["match"].lower() in ("commit", "branch") and (
                h["text"].startswith("`python -m calibration.train_map --model ") or
                h["text"].startswith(("**Provenance of the map file.**", "**Provenance of the map files.**")))]


def unaccepted(hits):
    return [h for h in hits if not any(rule(h) for rule in ACCEPTED)]


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def repo_id(model, kind):
    return f"{NS}/{H.MODELS[model][0]}-{SUFFIX[kind]}-Calib1k"


def repo256(model, kind):
    """The existing 256-window repository of the same kind (cited in the card; never touched)."""
    return f"{NS}/{H.MODELS[model][0]}-{SUFFIX[kind]}"


def title(model):
    return H.MODELS[model][0]


def is_llama(model):
    return "notice" in H.LICENSES[model]


# ------------------------------------------------------------------------------------------------- part AA's numbers
@functools.lru_cache(maxsize=None)
def record():
    return T.git_json(CS_COMMIT, "results/paper_eval/calib_size/calib_size.json")


@functools.lru_cache(maxsize=None)
def gptq_vs_fo6(model):
    """GPTQ-1k vs FourOverSix, which part AA's record lacks: its report's compare() on the same records. As a check, the
    same function must reproduce the record's GPTQ-1k vs GPTQ-256 pair exactly."""
    c = RC.compare(model, "gptq_1024", "gptq")
    want = record()["models"][model]["paired"][VS256["gptq"]]
    assert json.loads(json.dumps(c, default=float)) == json.loads(json.dumps(want, default=float)), model
    return RC.compare(model, "gptq_1024", "fo6")


def values(model, name):
    return record()["models"][model]["values"][name]


def paired(model, kind):
    """[(label, pair)]: this repository's row against FourOverSix and against its 256-window version (for Qwen3.8-27B's
    FOCUS: the cluster state and the local control)."""
    p = record()["models"][model]["paired"]
    fo6 = p[VSFO6[kind]] if VSFO6[kind] else gptq_vs_fo6(model)
    out = [("vs FourOverSix", fo6)]
    if kind == "focus" and model == Q27:
        out += [("vs the 256-window state (2 x H200 cluster run; a different environment)",
                 p["FOCUS-1k vs FOCUS-256 (cluster: a different environment)"]),
                ("vs a 256-window state retrained locally as this one (the environment-matched control)",
                 p["FOCUS-1k vs FOCUS-256 local (same environment)"])]
    else:
        out += [(f"vs its 256-window version ({repo256(model, kind).split('/')[1]})", p[VS256[kind]])]
    return out


def rows(model, kind):
    """The results table's rows: (label, values name)."""
    name = {"flipquant": "FlipQuant 16x64", "focus": "FOCUS", "gptq": "GPTQ‡"}[kind]
    out = [("BF16", "bf16"), ("FourOverSix (reference)", "fo6"),
           (f"{name}, 256 windows (`{repo256(model, kind).split('/')[1]}`)", REF256[kind])]
    if kind == "focus" and model == Q27:
        out[-1] = (f"{name}, 256 windows, 2 x H200 (`{repo256(model, kind).split('/')[1]}`)", "focus256")
        out.append((f"{name}, 256 windows, retrained locally as this one (control; not released)", "focus256_local"))
    out.append((f"{name}, 1,024 windows (**this repository**)", ROW[kind]))
    return out


def fmt_pair(x):
    return f"{x['diff']:+.2f}{'*' if x['p'] < 0.05 else ''} ({x['row_only']}/{x['ref_only']}, p={x['p']:.2g})"


def fmt_nll(x):
    return f"{x['delta']:+.4f} ± {x['two_se']:.4f}{'*' if x['significant'] else ''}"


def ppl_record(model, kind):
    """The row's PPL record: evaluation settings and windows (no path is copied from it)."""
    return json.loads((CS.ROOT / model / "eval" / "ppl" / f"{ROW[kind]}.json").read_text())


def windows_facts(model):
    rec = record()
    nest = rec["windows"]["nesting"][model]
    hits = rec["windows"]["audit_hits"].get(model)
    assert nest["ok"] and nest["1024"]["ok"] and hits == [], (model, nest["ok"], hits)
    w = json.loads(CS.windows_file(model, 1024).read_text())
    return dict(token_list_sha256=nest["1024"]["token_list_sha256"], skipped=nest["1024"]["skipped"],
                extension_windows=w["fit_extension"]["windows"], audit_hits=0)


def windows_text(model):
    f = windows_facts(model)
    sk = f["skipped"]
    return ("1,024 windows of 512 tokens (524,288 tokens). The first 128 are the calibration record's: the FlipQuant "
            "paper's math/code windows (OpenWebMath @ fde8ef8, CodeParrot-clean @ 35a59fb), built by the paper's rule "
            f"with this model's tokenizer. The other {f['extension_windows']} come from the release extension rule "
            "(seed 20260930; one window per document, half from each source). It skips the record's fit documents, the "
            f"development records and the published C4 evaluation documents ({sk['fit']} / {sk['development']} / "
            f"{sk['c4_evaluation']} skipped). The set contains the release's 256 windows, window for window, so the "
            "256-window version's fit set is a subset of this one. Every window is recorded by document sha256, offset "
            f"and token sha256 (sha256 of the token-hash list: `{f['token_list_sha256']}`). A word 13-gram audit of the "
            "study's 2,048-window set, which contains these windows, against every evaluation text found no shared "
            "13-gram. The evaluation texts are WikiText-2 test, the C4 evaluation documents, the six zero-shot tasks, "
            "GSM8K test and its few-shot exemplars. The audit's positive control detects evaluation items spliced into "
            "windows, 30 of 30 for each item.")


def eval_text(model):
    gsm = ("lm-eval's `gsm8k_llama`: 8-shot chain of thought with the chat template, greedy decoding, at most 1,024 new "
           "tokens, strict-match accuracy ± 2 SE")
    if model in CS.QWEN3:
        gsm += ", with Qwen3's thinking mode off (`enable_thinking=False`)"
    if model == CS.LLAMA1B:
        gsm += (". The chat template writes the current date into the system prompt; every row of this model renders "
                "15 Oct 2026, so all rows see identical prompts")
    return ["Native sm_120 kernels on one NVIDIA RTX PRO 6000 Blackwell. Every quantized row uses per-token FourOverSix "
            "activations, and every row is evaluated on the same data:", "",
            "- PPL: WikiText-2 test (141 windows of 2,048 tokens) and C4 validation (256 windows of 2,048 tokens), "
            "FlipQuant's `--paper-convention`;",
            "- zero-shot: PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge and LAMBADA (lm-eval 0.4.11, full "
            "evaluation splits; acc_norm for PIQA, HellaSwag and ARC, acc for WinoGrande and LAMBADA). Avg. is their "
            "mean;",
            f"- GSM8K: {gsm}."]


def results_section(model, kind):
    L = ["## Results", "",
         "From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.", ""]
    L += eval_text(model)
    L += ["", "| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |", "|---|---:|---:|---:|---:|"]
    bc = (record()["models"][model].get("baseline_check") or {}).get("bf16", {})
    batch_mark = bool((bc.get("gsm8k") or {}).get("batch_mark"))
    for label, name in rows(model, kind):
        v = values(model, name)
        g = f"{v['gsm8k'][0]:.2f} ± {v['gsm8k'][1]:.2f}" + (" ◊" if name == "bf16" and batch_mark else "")
        L.append(f"| {label} | {v['zeroshot']['avg']:.2f} | {g} | {v['ppl']['wiki']:.3f} | {v['ppl']['c4']:.3f} |")
    if batch_mark:
        L += ["", f"◊ BF16's GSM8K ran at batch {bc['gsm8k']['batch']}, the quantized rows at {bc['gsm8k']['batch_next']}."]
    L += ["", "Zero-shot accuracy (%) per task:", "",
          "| method | " + " | ".join(n for _, n in TASKS) + " |", "|---|" + "---:|" * len(TASKS)]
    for label, name in rows(model, kind):
        z = values(model, name)["zeroshot"]
        L.append(f"| {label} | " + " | ".join(f"{z[t]:.2f}" for t, _ in TASKS) + " |")
    L += ["", "**This repository against its references, paired.**", "",
          "- Zero-shot: pooled over all 21,848 questions of the six tasks.",
          "- GSM8K: strict-match over its 1,319 problems.",
          "- Format: difference in points (questions only this row gets right / only the reference gets right, exact "
          "McNemar p).",
          "- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.",
          "- `*`: p < 0.05, or |Δ| > 2 SE.", "",
          "| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |", "|---|---|---|---|---|"]
    for label, c in paired(model, kind):
        L.append(f"| {label} | {fmt_pair(c['zeroshot'])} | {fmt_pair(c['gsm8k'])} | {fmt_nll(c['nll']['wiki'])} | "
                 f"{fmt_nll(c['nll']['c4'])} |")
    return L


def results_summary(model, kind):
    """The ppl_summary.json numbers: values of every row of the table and the paired comparisons."""
    return dict(rows={name: dict(label=label, zeroshot=values(model, name)["zeroshot"],
                                 gsm8k_strict=dict(accuracy=values(model, name)["gsm8k"][0],
                                                   two_se=values(model, name)["gsm8k"][1]),
                                 gsm8k_flexible=dict(accuracy=values(model, name)["gsm8k_flex"][0],
                                                     two_se=values(model, name)["gsm8k_flex"][1]),
                                 ppl=values(model, name)["ppl"])
                      for label, name in rows(model, kind)},
                paired={label: dict(zeroshot=c["zeroshot"], gsm8k_strict=c["gsm8k"], gsm8k_flexible=c["gsm8k_flex"],
                                    nll={k: dict(delta_nll=x["delta"], two_se=x["two_se"], n=x["n"],
                                                 significant=x["significant"]) for k, x in c["nll"].items()})
                        for label, c in paired(model, kind)},
                source="the FlipQuant paper's calibration-size study")


def evaluation_summary(model, kind):
    rec = ppl_record(model, kind)
    return dict(mode="native sm_120 kernels", convention=rec["convention"], quantized_modules=rec["quantized_modules"],
                transformers=rec["source"]["transformers"], torch=rec["source"]["torch"], gpu=rec["resources"]["gpus"][0],
                corpora={c: {"dataset": FH.DATASETS[c]} | {k: v for k, v in rec["results"][c]["data"].items()
                                                            if k != "documents"} for c, _ in T.CORPORA},
                zeroshot="PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, LAMBADA; lm-eval 0.4.11; full splits; "
                         "acc_norm for PIQA, HellaSwag, ARC; acc for WinoGrande, LAMBADA",
                gsm8k="gsm8k_llama, 8-shot CoT, chat template, greedy, 1,024 new tokens"
                      + (", Qwen3 thinking off" if model in CS.QWEN3 else "")
                      + (", chat-template date pinned to 15 Oct 2026" if model == CS.LLAMA1B else ""))


# ------------------------------------------------------------------------------------------------------------ license
def license_files(model, dst):
    return H.license_files(model, dst)


def license_section(model, kind, plural=False):
    lic = H.LICENSES[model]
    what = "These maps are" if plural else {"flipquant": "This map is", "focus": "This state is", "gptq": "These weights are"}[kind]
    L = ["## License", ""]
    if kind == "gptq":
        L.append(f"These weights are derived from the base model's weights and are released under the base model's "
                 f"license, the {lic['name']}.")
    else:
        L.append(f"{what} released under the same license as the base model, the {lic['name']}.")
    L.append("")
    if is_llama(model):
        terms = H.llama_license_text(model, "terms")
        if kind != "gptq":
            terms = terms.replace("\n\n- `LICENSE`", f" {what} trained against {title(model)}'s outputs (KL to its BF16 "
                                  "logits).\n\n- `LICENSE`", 1)
        L.append(terms)
    else:
        L.append(lic["note"])
    return L


def files_section(model, kind, files, main_lines):
    L = ["## Files", "", "| file | bytes | sha256 |", "|---|---:|---|"]
    L += [f"| `{n}` | {d['bytes']:,} | `{d['sha256']}` |" for n, d in files.items()]
    L += [""] + main_lines + ["- `ppl_summary.json`: the numbers above, the evaluation settings, the calibration "
                              "settings and cost."]
    L += [f"- `{H.FILE_NAME[k]}`: see License." for k in H.LICENSES[model]["files"]]
    return L


def front(model, tags):
    f = {**H.LICENSES[model]["front"], "base_model": H.MODELS[model][1], "tags": tags}
    return ["---", yaml.safe_dump(f, sort_keys=False, allow_unicode=True, width=1000).strip(), "---", ""]


def head_lines(model):
    return [H.llama_license_text(model, "head"), ""] if is_llama(model) else []


# --------------------------------------------------------------------------------------------------------- FlipQuant
def neutral_meta(model, meta, unit="16x64"):
    """llama_hf.neutral_meta for a part AA map: the run directory relative to the model's part AA directory, the
    calibration record relative to the release work directory, the extension reference without its GitHub owner, and
    the reference implementation's branch and commit removed (the FlipQuant commit kept)."""
    m, changes = copy.deepcopy(meta), []
    for k, prefix in (("run", str(CS.ROOT / model) + "/"), ("calibration_record", LH.FQREL_PREFIX)):
        v = m["source"].get(k)
        assert isinstance(v, str) and v.startswith(prefix), (k, v)
        m["source"][k] = v[len(prefix):]
        changes.append((f"source.{k}", v, m["source"][k]))
    v = m["fit_extension"]["flipquant_reference"]
    assert v.startswith(LH.OWNER_PREFIX), v
    m["fit_extension"]["flipquant_reference"] = "flipquant " + v[len(LH.OWNER_PREFIX):]
    changes.append(("fit_extension.flipquant_reference", v, m["fit_extension"]["flipquant_reference"]))
    for k in ("branch", "commit"):
        if k in m:
            changes.append((k, m.pop(k), None))
    assert m["flipquant"]["commit"] and m["repo"] == "the reference implementation"
    assert m["model_id"] == H.MODELS[model][1] and m["revision"] == H.MODELS[model][2] and m["unit"] == unit
    assert m["settings"]["fit_windows"] == 1024 and m["settings"]["epochs"] == 5 and m["settings"]["teacher_topk"] == 1000
    return m, changes


def stage_map(model, src, dst, unit="16x64"):
    import torch
    sys.path.insert(0, str(CS.FQ3))
    from flipquant import maps as M
    tiles, u, meta = M.load(src)
    new_meta, changes = neutral_meta(model, meta, unit)
    out = dst / f"flipquant_{unit}.pt"
    M.save(out, tiles, u, new_meta)
    t2, u2, m2 = M.load(out)
    assert u2 == u and set(t2) == set(tiles)
    assert all(t2[n].dtype == tiles[n].dtype and t2[n].shape == tiles[n].shape and torch.equal(t2[n], tiles[n])
               for n in tiles), "tiles differ"
    a, b = LH.tensor_records(src), LH.tensor_records(out)
    assert set(a) == set(b) and all(a[k] == b[k] for k in a), "tensor records differ"
    assert m2 == new_meta
    return dict(file=out.name, unit=u, sha256_source=FH.sha(src), sha256_uploaded=FH.sha(out), meta_changes=changes,
                modules=len(tiles), tiles=sum(t.numel() for t in tiles.values()),
                e0m3_tiles=int(sum(int(t.sum()) for t in tiles.values())), tiles_bit_identical=True,
                tensor_records_identical=True, flipquant_commit=new_meta["flipquant"]["commit"],
                settings=new_meta["settings"], teacher=new_meta["teacher"], teacher_host=new_meta["teacher_host"],
                calibration_time=new_meta["calibration_time"])


def fq_cost(model):
    j = json.loads((CS.ROOT / model / "fq16_1024.job.json").read_text())
    t = j["trainer"]
    return dict(steps=5 * 1024 // 8, job_seconds=j["job_seconds"], training_seconds=t["training_seconds"],
                setup_seconds=t["setup_seconds"], gpu_peak_allocated_gib=t["gpu_peak_allocated_gib"][0],
                gpu_peak_reserved_gib=t["gpu_peak_reserved_gib"][0], cpu_peak_rss_gib=t["cpu_peak_rss_gib"],
                device_peak_mib=j["device_peak_mib"], teacher_bytes=(j.get("teacher_storage") or {}).get("bytes"),
                teacher_devices=(j.get("teacher_storage") or {}).get("devices"), e0m3_share=j["e0m3_share"])


def fq_card(model, m, cost, files):
    key, s = model, m["settings"]
    share = m["e0m3_tiles"] / m["tiles"]
    host = model in CS.HOST_TEACHER
    L = front(model, LH.FQ_TAGS) + [f"# {title(model)} FlipQuant MixFP4 map, 1,024 calibration windows", ""]
    L += head_lines(model)
    L += ["## What this is", "",
          f"A format map for [{H.MODELS[model][1]}](https://huggingface.co/{H.MODELS[model][1]}) at revision "
          f"`{H.MODELS[model][2]}`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times the "
          f"release's 256 (`{repo256(model, 'flipquant')}`). A map holds one bit per weight tile of every quantized linear "
          "layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 (uniform "
          "levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an E4M3 scale "
          "and a per-tensor FP32 scale. The map holds no weights: it is applied to the base model's BF16 checkpoint when "
          "it is loaded.", "",
          "| file | tile (output rows × input columns) | E0M3 tiles | sha256 |", "|---|---|---:|---|",
          f"| `{MAP_NAME}` | 16 × 64 | {m['e0m3_tiles']:,} of {m['tiles']:,} ({100 * share:.2f} %) | "
          f"`{m['sha256_uploaded']}` |", "",
          f"The E0M3 share is {100 * share:.2f} % (the 256-window map: "
          f"{100 * RC_e0m3(model, 256):.2f} %). Only the 16x64 map is released here.", "", LH.FQ_CODE, "",
          "## Calibration", "",
          f"`python -m calibration.train_map --model {key} --unit 16x64 --epochs 5 --teacher-topk 1000 --fit-windows 1024"
          + (" --teacher-host" if host else "") + f"` (FlipQuant commit `{m['flipquant_commit']}`):", "",
          "- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per "
          "position plus one tail bucket, per-token FourOverSix activations.",
          f"- Adam, learning rate {s['lr']}, initial logit {format(s['init_logit'], 'g').replace('-', '−')} (every tile starts as E2M1), seed {s['seed']}, "
          f"{s['batch'] * s['accum']} sequences per step"
          + (f" ({s['batch']} × {s['accum']} accumulated)" if s["accum"] > 1 else "")
          + f", 5 epochs: **{cost['steps']} steps** (the release: 160). Deterministic.",
          f"- Fit set: {windows_text(model)}",
          "- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The map's "
          "`meta` records them as `non_default_settings`.", ""]
    if host:
        L += ["The BF16 teacher's top-1000 targets for 1,024 windows do not fit next to this model on the GPU, so they "
              "are kept in host memory (`--teacher-host`). On Llama-3.2-1B-Instruct, the same option gives the GPU-teacher "
              "map bit for bit.", ""]
    L += ["Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120):", "",
          "| setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | "
          "teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
          f"| {LH.fmt(cost['setup_seconds'])} | {LH.fmt(cost['training_seconds'])} | {LH.fmt(cost['job_seconds'])} | "
          f"{cost['gpu_peak_allocated_gib']:.1f} GiB | {cost['gpu_peak_reserved_gib']:.1f} GiB | "
          f"{cost['cpu_peak_rss_gib']:.1f} GiB | {cost['teacher_bytes'] / 2 ** 30:.2f} GiB "
          f"({', '.join(cost['teacher_devices'] or [])}) | {cost['device_peak_mib'] / 1024:.1f} GiB |", ""]
    L += results_section(model, "flipquant")
    L += ["", "## How to use", "",
          LH.FQ_CODE + " With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then "
          "evaluate with the native kernels:", "", "```bash", "bash kernels/sm120/build_deploy.sh",
          f"python -m evaluation.ppl --model {key} --mode native --weight mixfp4 --map {MAP_NAME} --paper-convention",
          "```", "", "On other GPUs, the same map with simulated (fake) quantization:", "", "```bash",
          f"python -m evaluation.ppl --model {key} --mode fake --weight mixfp4 --map {MAP_NAME} --paper-convention",
          "```", "", "The native command produced the PPL above.", ""]
    L += license_section(model, "flipquant") + [""]
    L += files_section(model, "flipquant", files, [
        f"- `{MAP_NAME}`: the map (`flipquant-map/1`: `{{format, unit, tiles, meta}}`; `flipquant.maps.load` reads it)."])
    L += ["", "**Provenance of the map file.** It is the calibration output with only its `meta` changed: the run "
          "directory and calibration-record paths are relative instead of local absolute paths, the data-extension code "
          "reference names the file without its repository owner, and the internal branch and commit of the reference "
          "implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the "
          f"calibration output → this repository's file: `{m['sha256_source']}` → `{m['sha256_uploaded']}`.", ""]
    return "\n".join(L)


def RC_e0m3(model, n):
    c = [x for x in record()["models"][model]["cost"] if x["method"] == "FlipQuant 16x64" and x["windows"] == n]
    assert len(c) == 1, (model, n)
    return c[0]["e0m3_share"]


def fq_summary(model, m, cost):
    return dict(model=H.MODELS[model][1], revision=H.MODELS[model][2], flipquant_model_key=model,
                method="FlipQuant (TM-OPT+TC), 16x64 tiles, 1,024 calibration windows",
                map={MAP_NAME: dict(sha256=m["sha256_uploaded"], calibration_output_sha256=m["sha256_source"],
                                    e0m3_tiles=m["e0m3_tiles"], tiles=m["tiles"],
                                    e0m3_share=m["e0m3_tiles"] / m["tiles"], modules=m["modules"])},
                evaluation=evaluation_summary(model, "flipquant"), results=results_summary(model, "flipquant"),
                calibration=dict(settings=m["settings"], steps=cost["steps"], teacher_host=m["teacher_host"],
                                 windows=windows_facts(model), flipquant_commit=m["flipquant_commit"],
                                 cost={k: v for k, v in cost.items() if k != "e0m3_share"}))


# ------------------------------------------------------------------------------------------------------------- FOCUS
def focus_report(model):
    d = CS.ROOT / model / "focus_1024"
    p = d / "report.json" if (d / "report.json").exists() else d / "state" / "report.json"
    r = json.loads(p.read_text())
    assert r["status"] == "complete" and r["steps"] == 32 and r["data"]["windows"] == 1024, model
    return r


def focus_cost(model):
    r = focus_report(model)
    j = json.loads((CS.ROOT / model / "focus_1024" / "state.job.json").read_text())
    return dict(steps=r["steps"], micro_batch=int(r["args"]["micro"]), training_seconds=r["training_seconds"],
                end_to_end_seconds=r["resources"]["seconds"], gpu=r["resources"]["gpus"][0],
                gpu_peak_allocated_gib=r["resources"]["gpu_peak_gib"][0], cpu_peak_rss_gib=j.get("cpu_peak_rss_gib"),
                offload=bool(r.get("offload")), parameters=r["parameters"], data_windows=r["data"]["windows"],
                data_tokens=r["data"]["tokens"], device_peak_mib=j["device_peak_mib"])


def focus_card(model, info, cs, files):
    hf_id, rev = info["source"]["hf_id"], info["source"]["revision"]
    t = title(model)
    L = front(model, FH.TAGS) + [f"# FOCUS NVFP4 scales for {t} (our reproduction), 1,024 calibration windows", ""]
    L += head_lines(model)
    L += ["**This is our reproduction of FOCUS** ([“FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity "
          "Scaling”](https://arxiv.org/abs/2608.01847), arXiv:2608.01847), **made as a baseline for the FlipQuant paper. "
          "It is not an official release by the FOCUS authors.** It is calibrated on 1,024 windows, four times the "
          f"256 of `{repo256(model, 'focus')}`.", ""]
    if model == Q27:
        st = record()["models"][model]["gsm8k_stops"]
        L += ["**Note on GSM8K.** This state stops its GSM8K answers early: GSM8K strict-match "
              f"{values(model, 'focus_1024')['gsm8k'][0]:.2f} %, with {st['focus_1024']['strict_no_answer']} of 1,319 "
              f"strict-match answers missing (the 256-window state: {st['focus256']['strict_no_answer']}). The behaviour "
              "is specific to this run. A 256-window state retrained locally exactly as this one "
              f"({st['focus256_local']['strict_no_answer']} missing) and the study's 2,048-window state "
              f"({st['focus_2048']['strict_no_answer']} missing) do not show it. FOCUS training is not reproducible run "
              "to run. The state is released as evaluated.", ""]
    L += ["## What this is", "",
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
          "`in_features`. Each layer also has `act_gscale = None`: activations use one scale per token, so there is no "
          "static activation scale. The file also stores the calibration settings (`config`) and the base model's id and "
          "revision (`source`).", "",
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
          "runs on standard NVFP4 kernels with no extra metadata or cost.", "", FH.CODE, "",
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
          f"- **Calibration data:** {windows_text(model)} That is {cs['data_tokens']:,} tokens and **{cs['steps']} "
          "optimizer steps** (the 256-window state: 8). The paper uses 1,248 WikiText-2 training samples of 2,048 tokens "
          "(39 steps).",
          "- **Activations**, during calibration and evaluation: FourOverSix with one scale per token, as in the FlipQuant "
          "paper's protocol. The paper quantizes activations to NVFP4 with a static per-tensor scale."]
    if model == Q27:
        L += ["- **Training setup (a deviation):** trained locally on one RTX PRO 6000 with host-offloaded AdamW "
              "(`--offload`, micro-batch 1), not on 2 x H200 like its 256-window state. AdamW runs on the CPU in FP32 on "
              "gradients accumulated in the GPU path's order; only the summation order differs."]
    L += ["", f"Cost on one {cs['gpu']} (sm_120), micro-batch {cs['micro_batch']} ({32 // cs['micro_batch']} "
          "forward/backward passes per step), activation checkpointing per decoder block"
          + (", AdamW offloaded to the host" if cs["offload"] else "") + ":", "",
          f"| training ({cs['steps']} steps) | end to end (model load, data, training, save) | peak GPU memory |",
          "|---:|---:|---:|",
          f"| {FH.fmt_s(cs['training_seconds'])} | {FH.fmt_s(cs['end_to_end_seconds'])} | "
          f"{cs['gpu_peak_allocated_gib']:.1f} GiB |", "",
          "The peak is PyTorch's peak allocated memory on the GPU (`torch.cuda.max_memory_allocated`). The training "
          "time includes the BF16 teacher's forward passes.", ""]
    L += results_section(model, "focus")
    L += ["", "## How to read the file", "", "With PyTorch alone:", "", "```python", "import torch", "",
          'state = torch.load("focus.pt", map_location="cpu", weights_only=True)',
          'state["config"]   # the calibration settings',
          'state["source"]   # the base model id and revision',
          f'layer = state["layers"]["{info["first_layer"]}"]',
          'layer["m"], layer["q"], layer["s2"]   # FP32 tensors (see above)', "```", ""]
    L += license_section(model, "focus") + [""]
    L += files_section(model, "focus", files, [
        "- `focus.pt`: the FOCUS state (`{config, layers, source}`; above). It is the calibration output, unchanged."])
    L.append("")
    return "\n".join(L)


def focus_summary(model, info, cs, sha):
    return dict(model=info["source"]["hf_id"], revision=info["source"]["revision"], model_key=info["config"]["model"],
                method="FOCUS (arXiv:2608.01847), our reproduction; not an official release by the FOCUS authors; "
                       "1,024 calibration windows",
                state={"focus.pt": dict(sha256=sha, modules=info["modules"], block_factors_m=info["block_factors_m"],
                                        subblock_logits_q=info["subblock_logits_q"],
                                        tensor_scales_s2=info["tensor_scales_s2"], dtype="float32")},
                evaluation=evaluation_summary(model, "focus"), results=results_summary(model, "focus"),
                calibration=dict(info["config"], steps=cs["steps"], micro_batch=cs["micro_batch"],
                                 windows=windows_facts(model), data_tokens=cs["data_tokens"],
                                 optimizer="AdamW (0.9, 0.999), weight decay 0, constant LR"
                                           + ("; offloaded to the host (one GPU)" if cs["offload"] else ""),
                                 teacher="the unquantized BF16 model", training_seconds=cs["training_seconds"],
                                 end_to_end_seconds=cs["end_to_end_seconds"],
                                 gpu_peak_allocated_gib=cs["gpu_peak_allocated_gib"], gpu=cs["gpu"]))


# -------------------------------------------------------------------------------------------------------------- GPTQ
def verify_codes(model, path, job):
    """The codes file: its code digest equals the calibration's, its key is the main-table variant at 1,024 windows, and
    the card's decoding snippet reproduces FlipQuant's dequantization bit for bit."""
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True)
    key = obj["key"]
    digest = H.LH_entries_sha256(obj["codes"])
    first = sorted(obj["codes"])[0]
    c = obj["codes"][first]
    want_key = dict(H.MAIN_TABLE_KEY, windows=1024, model=model, data_sha256=job["key"]["data_sha256"])
    out = dict(format=obj.get("format"), modules=len(obj["codes"]), codes_sha256=digest,
               record_codes_sha256=job["codes_sha256"], key=key,
               key_differences={f: (key.get(f), v) for f, v in want_key.items() if key.get(f) != v},
               first_module=first, first_shapes=dict(q=list(c["q"].shape), s=list(c["s"].shape), g=list(c["g"].shape)),
               fmt_values=sorted({str(int(v["fmt"])) if not torch.is_tensor(v["fmt"]) else "tensor"
                                  for v in obj["codes"].values()}),
               decode_snippet_bitwise=H.decode_check(c))
    out["ok"] = (out["format"] == "flipquant-gptq/1" and digest == job["codes_sha256"] and not out["key_differences"]
                 and out["fmt_values"] == ["1"] and out["decode_snippet_bitwise"])
    del obj
    return out


def gptq_card(model, v, job, files):
    t, hf_id, rev = H.MODELS[model][:3]
    key = v["key"]
    batch = 1 if model == Q27 else 8
    L = front(model, H.TAGS) + [f"# GPTQ NVFP4 weights for {t}, calibrated with BF16 activations on 1,024 windows", ""]
    L += head_lines(model)
    L += ["These are **GPTQ‡** weights as in the FlipQuant paper's main perplexity table (GPTQ-quantized NVFP4 weights, "
          "calibrated with BF16 activations, evaluated with FourOverSix per-token activations), calibrated on **1,024 "
          f"windows** instead of the main table's 256 (`{repo256(model, 'gptq')}`). They were made for the FlipQuant "
          "paper's calibration-size study.", ""]
    if model in H.COAUTHOR_GPTQ:
        L += [f"Not the same artifact as `{H.old_gptq_repo(model)}`: this repository is GPTQ calibrated with BF16 "
              "activations; that one is calibrated with FourOverSix activations.", ""]
    L += ["## What this is", "",
          f"`{CODES_NAME}` holds the NVFP4 codes of every quantized linear layer of [{hf_id}](https://huggingface.co/{hf_id}) "
          f"at revision `{rev}` ({v['modules']} layers: all text-model linear layers except the LM head"
          + ("; the vision tower stays BF16" if model == Q27 else "") + "). NVFP4 stores each weight as a 4-bit E2M1 "
          "code, with one E4M3 scale per 16 consecutive elements of a row and one FP32 scale per tensor.", "",
          "**It is not a full checkpoint**: only the quantized linear layers' weights are here. Everything else "
          "(embeddings, norms, the LM head"
          + (", the Mamba / convolution parameters" if model == "nemotron-nano-9b-v2" else "")
          + (", the recurrent and convolution parameters" if model == Q27 else "")
          + ") comes from the BF16 base model at the pinned revision.", "",
          "## How it was made", "",
          "GPTQ (FlipQuant's `flipquant.gptq`), layer by layer in model order, on NVFP4's fixed round-to-nearest grid: "
          "each 16-element block keeps its round-to-nearest E4M3 scale and each tensor its FP32 scale, and GPTQ chooses "
          "the codes. The settings are the main table's; only the windows differ:", "",
          f"- Calibration data: {windows_text(model)} sha256 of the windows' token ids, in order: `{key['data_sha256']}`.",
          "- BF16-activation calibration: during calibration every activation quantizer is off, so the layers already "
          "quantized run with quantized weights and BF16 activations, as in weight-only GPTQ, and the Hessian comes from "
          "the BF16 layer inputs. The codes therefore do not depend on the activation format they are evaluated with.",
          f"- Lazy-update block {key['block']} columns, dampening {key['damp']} × mean(diag H), no act-order, the "
          "ModelOpt 0.47 handling of dead columns.",
          f"- Calibration batch {batch}; GPTQ took {job['gptq_seconds'] / 60:.1f} min on one "
          f"{ppl_record(model, 'gptq')['resources']['gpus'][0]} (the 256-window run: "
          f"{H_record_seconds(model) / 60:.1f} min), peak allocated GPU memory {job['gpu_peak_gib'][0]:.1f} GiB.",
          f"- Code: FlipQuant revision `{gptq_revision()}`, whose GPTQ code is the main table's, unchanged. On it, "
          "Llama-3.2-1B-Instruct's 256-window run reproduces the main table's codes and the published 256-window codes "
          "file bit for bit.",
          f"- Code digest (FlipQuant's `codes_sha256` over every module's packed codes, scales and formats): "
          f"`{v['codes_sha256']}`.", "", H.CODE, ""]
    L += results_section(model, "gptq")
    L += ["", "## How to use", "",
          "With FlipQuant (to be released with the paper), on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), the "
          "command that produced the PPL above:", "", "```bash",
          f"python -m evaluation.ppl --model {model} --revision {rev} --mode native \\",
          "    --weight nvfp4 --act fourover6 --any-act --paper-convention \\",
          "    --ptq gptq --gptq-propagate bf16 --gptq-hessian-input bf16 --gptq-calib release \\",
          "    --gptq-calib-record <calibration record> --gptq-calib-extension <the 1,024-window record> \\",
          f"    --gptq-block 128 --gptq-damp 0.01 --gptq-batch {batch} --gptq-cache {CODES_NAME}",
          "```", "",
          "FlipQuant reads the file instead of running GPTQ only if the file's stored settings, including the sha256 of the "
          "calibration windows, equal the run's; it rebuilds the windows from the calibration records to check this.", "",
          "With PyTorch alone, one layer's weight (this reproduces FlipQuant's own dequantization bit for bit):", "",
          "```python", H.DECODE_SNIPPET.replace("NAME", json.dumps(v["first_module"])).rstrip(), "```", "",
          f"In the file: `q` is uint8 `{v['first_shapes']['q']}`-shaped (two 4-bit codes per byte, element 2j in the low "
          f"nibble), `s` uint8 (the raw E4M3 bytes of the block scales), `g` the FP32 tensor scale, `fmt` 1 (every block "
          "E2M1).", ""]
    L += license_section(model, "gptq") + [""]
    L += files_section(model, "gptq", files, [
        f"- `{CODES_NAME}`: the GPTQ codes (`flipquant-gptq/1`: `{{format, key, codes}}`; above), the calibration "
        "output, unchanged."])
    L.append("")
    return "\n".join(L)


@functools.lru_cache(maxsize=None)
def gptq_revision():
    import subprocess
    return subprocess.run(["git", "-C", str(CS.FQ3), "rev-parse", CS.GPTQ_CODE], capture_output=True, text=True,
                          check=True).stdout.strip()


def H_record_seconds(model):
    return json.loads(Z.gptq_record(model).read_text())["ptq"]["seconds"]


def gptq_summary(model, v, job, sha, size):
    return dict(model=H.MODELS[model][1], revision=H.MODELS[model][2], flipquant_model_key=model,
                method="GPTQ on NVFP4's fixed round-to-nearest grid, BF16-activation calibration (the FlipQuant paper's "
                       "GPTQ-double-dagger baseline), 1,024 calibration windows",
                codes={CODES_NAME: dict(sha256=sha, bytes=size, format=v["format"], modules=v["modules"],
                                        codes_sha256=v["codes_sha256"], key=v["key"])},
                evaluation=evaluation_summary(model, "gptq"), results=results_summary(model, "gptq"),
                gptq=dict(v["key"], flipquant_revision=gptq_revision(), seconds=job["gptq_seconds"],
                          gpu_peak_allocated_gib=job["gpu_peak_gib"][0], windows_detail=windows_facts(model)))


# ------------------------------------------------------------------------------------------------------------- stage
def sources(model):
    d = CS.ROOT / model
    fq = json.loads((d / "fq16_1024.job.json").read_text())
    fo = json.loads((d / "focus_1024" / "state.job.json").read_text())
    gq = json.loads((d / "gptq_1024.job.json").read_text())
    return dict(flipquant=dict(path=d / "fq16_1024" / "map.pt", sha256=fq["map_file_sha256"], job=fq),
                focus=dict(path=Path(fo.get("state") or d / "focus_1024" / "state" / "focus.pt"),   # the pilot's: no field
                           sha256=fo["state_sha256"], job=fo),
                gptq=dict(path=Path(gq["cache"]), sha256=gq["cache_sha256"], job=gq))


def link(src, dst):
    if dst.is_symlink() or dst.exists():
        assert dst.is_symlink() and os.readlink(dst) == str(src), f"{dst} exists and is not the link to {src}"
    else:
        dst.symlink_to(src)


def text_hits(dst, skip):
    hits = []
    for f in sorted(dst.iterdir()):
        if f.is_file() and f.name not in skip:
            hits += FH.scan_text(f.name, f.read_text(), ref=f.name in ("README.md", "ppl_summary.json"))
    return hits


def staged_files(dst, big=None):
    """{name: bytes, sha256} of the staged files; `big` = (name, sha256) of the linked file, hashed once already."""
    out = {}
    for f in sorted(dst.iterdir()):
        if f.is_file():
            out[f.name] = dict(bytes=f.stat().st_size,
                               sha256=big[1] if big and f.name == big[0] else FH.sha(f),
                               link=os.readlink(f) if f.is_symlink() else None)
    return out


def raw_hit_location(path, hits):
    """hf_scan_hits for one codes file: each raw-byte hit's archive member and the tensor field that member backs."""
    mem, owner = HS.members(path), HS.storage_owner(path)
    where, outside = collections.Counter(), 0
    for h in hits:
        off = int(h["where"].split("@")[1])
        end = off + len(h["match"].encode())
        m = next((x for x in mem if x[1] <= off and end <= x[2]), None)
        k = m[0].rsplit("/", 1)[-1] if m and "/data/" in m[0] else None
        field = owner.get(k, (None, None, None))[1]
        where[field or (m[0] if m else "outside any member")] += 1
        outside += field is None
    return dict(raw_hits=len(hits), by_field=dict(where), all_inside_tensor_storage=outside == 0)


def stage_model(model, record_):
    src = sources(model)
    for kind in KINDS:
        rid = repo_id(model, kind)
        if rid in record_["repos"]:
            continue
        s = src[kind]
        got = FH.sha(s["path"])                                   # a fresh full read of the source
        if got != s["sha256"]:
            sys.exit(f"{rid}: the source's sha256 {got[:12]} != its job record's {s['sha256'][:12]} ({s['path']})")
        dst = STAGE / rid.split("/")[1]
        dst.mkdir(parents=True, exist_ok=True)
        lic = license_files(model, dst)
        rec = dict(kind=kind, model=model, base_model=H.MODELS[model][1], revision=H.MODELS[model][2],
                   folder=str(dst), license=H.LICENSES[model]["front"], repo_256=repo256(model, kind),
                   source=dict(path=str(s["path"]), sha256=got, recorded_sha256=s["sha256"], equal=True),
                   license_files=lic, staged_utc=now())
        if kind == "flipquant":
            m = stage_map(model, s["path"], dst)
            assert m["sha256_source"] == got
            cost = fq_cost(model)
            assert abs(cost["e0m3_share"] - m["e0m3_tiles"] / m["tiles"]) < 1e-12, model
            (dst / "ppl_summary.json").write_text(json.dumps(fq_summary(model, m, cost), indent=1, ensure_ascii=False) + "\n")
            files = staged_files(dst)
            (dst / "README.md").write_text(fq_card(model, m, cost, files))
            hits = LH.scan_map(dst / MAP_NAME) + text_hits(dst, {MAP_NAME})
            rec.update(map={k: v for k, v in m.items() if k not in ("settings", "teacher")}, cost=cost)
        elif kind == "focus":
            link(s["path"], dst / "focus.pt")
            hits, info = FH.scan_state(dst / "focus.pt")
            assert info["source"]["hf_id"] == H.MODELS[model][1] and info["source"]["revision"] == H.MODELS[model][2]
            assert info["config"]["model"] == model and info["config"]["windows"] == 1024
            cs = focus_cost(model)
            assert info["modules"] == cs["parameters"]["modules"]
            assert info["block_factors_m"] == cs["parameters"]["crs"] and info["subblock_logits_q"] == cs["parameters"]["dgs"]
            (dst / "ppl_summary.json").write_text(json.dumps(focus_summary(model, info, cs, got), indent=1,
                                                             ensure_ascii=False) + "\n")
            files = staged_files(dst, ("focus.pt", got))
            (dst / "README.md").write_text(focus_card(model, info, cs, files))
            hits += text_hits(dst, {"focus.pt"})
            rec.update(state={k: v for k, v in info.items() if k not in ("config", "source")}, config=info["config"],
                       source_meta=info["source"], cost=cs)
        else:
            link(s["path"], dst / CODES_NAME)
            v = verify_codes(model, dst / CODES_NAME, s["job"])
            if not v["ok"]:
                sys.exit(f"{rid}: the codes file fails its checks: "
                         f"{json.dumps({k: v[k] for k in ('key_differences', 'fmt_values', 'decode_snippet_bitwise')})}")
            hits = H.scan_codes(dst / CODES_NAME)
            raw = [h for h in hits if h["where"].startswith("raw bytes @")]
            loc = raw_hit_location(dst / CODES_NAME, raw)
            size = (dst / CODES_NAME).stat().st_size
            (dst / "ppl_summary.json").write_text(json.dumps(gptq_summary(model, v, s["job"], got, size), indent=1,
                                                             ensure_ascii=False) + "\n")
            files = staged_files(dst, (CODES_NAME, got))
            (dst / "README.md").write_text(gptq_card(model, v, s["job"], files))
            hits += text_hits(dst, {CODES_NAME})
            rec.update(codes={k: x for k, x in v.items() if k != "key"}, key=v["key"], raw_hits=loc)
        rec.update(files=staged_files(dst, {"flipquant": None, "focus": ("focus.pt", got),
                                            "gptq": (CODES_NAME, got)}[kind]), scan=hits)
        record_["repos"][rid] = rec
        print(f"{rid}: {list(rec['files'])} scan hits {len(hits)} ({len(unaccepted(hits))} not accepted)", flush=True)
        (RECS / "stage.json").write_text(json.dumps(record_, indent=1, ensure_ascii=False, default=str) + "\n")


def stage(models=None):
    RECS.mkdir(parents=True, exist_ok=True)
    path = RECS / "stage.json"
    rec = json.loads(path.read_text()) if path.exists() else dict(namespace=NS, part="AB", cs_commit=CS_COMMIT, repos={})
    for model in models or ORDER:
        stage_model(model, rec)
    rec["staged_utc"] = now()
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str) + "\n")


def tokens():
    """hf_token_check over every staged file (the linked sources included); the tokens stay in memory."""
    full, prefix = TC.needles()
    if not full:
        sys.exit("no token found in the token files; nothing to check against")
    st = json.loads((RECS / "stage.json").read_text())
    out = dict(tokens_checked=len(full), token_files=[str(p) for p in TC.SOURCES if p.is_file()], files={})
    for rid, d in st["repos"].items():
        for name in sorted(d["files"]):
            n_full, n_prefix = TC.count(Path(d["folder"]) / name, full, prefix)
            out["files"][f"{rid}/{name}"] = dict(exact_token_hits=n_full, prefix_hits=n_prefix)
    out["all_zero"] = all(v["exact_token_hits"] == 0 and v["prefix_hits"] == 0 for v in out["files"].values())
    del full, prefix
    out["checked_utc"] = now()
    (RECS / "token_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(dict(tokens_checked=out["tokens_checked"], files=len(out["files"]), all_zero=out["all_zero"])))


# ----------------------------------------------------------------------------------------------------- check / upload
def rule_check():
    """The gates per repository: text files (accepted kinds only), the map and the FOCUS state (no hit at all), the codes
    file (the GPTQ upload's binary rule), the exact-token check, and each staged file unchanged since staging."""
    st = json.loads((RECS / "stage.json").read_text())
    tok = json.loads((RECS / "token_check.json").read_text())
    out = {}
    for rid, d in st["repos"].items():
        problems, binary = [], {}
        text = [h for h in d["scan"] if not h["file"].endswith(".pt")]
        if unaccepted(text):
            problems.append(f"text-file hits {[(h['file'], h['where'], h['match']) for h in unaccepted(text)][:3]}")
        pt = [h for h in d["scan"] if h["file"].endswith(".pt")]
        if d["kind"] in ("flipquant", "focus"):
            if pt:
                problems.append(f"{d['kind']} file hits {[(h['file'], h['where']) for h in pt[:3]]}")
        else:
            raw = [h for h in pt if h["where"].startswith("raw bytes @")]
            meta = [h for h in pt if not h["where"].startswith("raw bytes @")]
            loc = d["raw_hits"]
            binary = dict(raw_hits=len(raw), by_field=loc["by_field"],
                          r2_inside_q_or_s=loc["raw_hits"] == len(raw) and loc["all_inside_tensor_storage"]
                          and set(loc["by_field"]) <= {"q", "s"},
                          r3_no_metadata_hits=not meta,
                          r4_sha256_record=d["source"]["sha256"] == d["source"]["recorded_sha256"]
                          and d["codes"]["codes_sha256"] == d["codes"]["record_codes_sha256"])
            for k in ("r2_inside_q_or_s", "r3_no_metadata_hits", "r4_sha256_record"):
                if not binary[k]:
                    problems.append(f"rule {k} fails")
        tk = {n: v for n, v in tok["files"].items() if n.startswith(rid + "/")}
        if len(tk) != len(d["files"]) or any(v["exact_token_hits"] or v["prefix_hits"] for v in tk.values()):
            problems.append("exact-token check")
        for n, f in d["files"].items():
            p = Path(d["folder"]) / n
            if p.stat().st_size != f["bytes"] or (f["link"] is None and FH.sha(p) != f["sha256"]):
                problems.append(f"{n} changed since staging")
        out[rid] = dict(binary=binary, exact_token_check=f"{sum(v['exact_token_hits'] + v['prefix_hits'] for v in tk.values())} hits",
                        problems=problems)
    return out


def check():
    api = FH.account()
    st = json.loads((RECS / "stage.json").read_text())
    want = {repo_id(m, k) for m in ORDER for k in KINDS}
    assert set(st["repos"]) == want, sorted(want ^ set(st["repos"]))
    ours = set(json.loads((RECS / "hf_upload.json").read_text())["repos"]) if (RECS / "hf_upload.json").exists() else set()
    exists = {r: api.repo_exists(r, repo_type="model") for r in sorted(want)}
    foreign = sorted(r for r, e in exists.items() if e and r not in ours)
    print("existing, created by this upload:", sorted(r for r, e in exists.items() if e and r in ours) or "none")
    print("existing, not created by this upload:", foreign or "none")
    cited = {d["repo_256"]: api.repo_exists(d["repo_256"], repo_type="model") for d in st["repos"].values()}
    print("256-window repositories the cards cite, all present:", all(cited.values()),
          [r for r, e in cited.items() if not e] or "")
    rc = rule_check()
    (RECS / "rule_check.json").write_text(json.dumps(rc, indent=1) + "\n")
    for r, v in rc.items():
        print(r, "ok" if not v["problems"] else v["problems"], v["exact_token_check"], v["binary"].get("by_field") or "")
    if foreign:
        sys.exit("a target repository already exists; stopping (nothing is overwritten)")
    if not all(cited.values()):
        sys.exit("a cited 256-window repository is missing; stopping")
    if any(v["problems"] for v in rc.values()):
        sys.exit("a gate fails; stopping")
    print("all gates pass")


def lfs_sha(entry):
    lf = getattr(entry, "lfs", None)
    return (lf.get("sha256") if isinstance(lf, dict) else getattr(lf, "sha256", None)) if lf else None


def upload(only=None):
    """Upload the staged repositories (or those named in `only`), in ORDER, each: create PRIVATE, check private
    (anonymous HTTP 401), upload exactly the staged files, verify every file's sha256 (download up to 1 GiB, else the
    Hub's LFS sha256 and size), the file list and the card metadata. A failure stops the run."""
    os.environ.setdefault("HF_XET_CACHE", str(XET_CACHE))
    api = FH.account()
    st = json.loads((RECS / "stage.json").read_text())
    rc = rule_check()
    bad = {r: v["problems"] for r, v in rc.items() if v["problems"]}
    if bad:
        sys.exit(f"gates fail; nothing uploaded: {bad}")
    path = RECS / "hf_upload.json"
    out = json.loads(path.read_text()) if path.exists() else dict(account=NS, repos={})
    order = [repo_id(m, k) for m in ORDER for k in KINDS]
    if only:
        assert set(only) <= set(order), only
        order = [r for r in order if r in only]
    for rid in order:
        rec = st["repos"][rid]
        if out["repos"].get(rid, {}).get("verified"):
            print(f"{rid}: already uploaded and verified")
            continue
        folder = Path(rec["folder"])
        names = sorted(rec["files"])
        assert {f.name for f in folder.iterdir() if f.is_file()} == set(names), f"{rid}: staged files changed"
        resumed = rid in out["repos"]
        if api.repo_exists(rid, repo_type="model") and not resumed:
            sys.exit(f"{rid} exists and was not created by this upload; stopping")
        if not api.repo_exists(rid, repo_type="model"):
            api.create_repo(rid, repo_type="model", private=True, exist_ok=False)
        out["repos"].setdefault(rid, dict(created_utc=now()))
        path.write_text(json.dumps(out, indent=1, default=str) + "\n")
        info = api.model_info(rid)
        anon = FH.anonymous_status(rid)
        if info.private is not True or anon == 200:
            sys.exit(f"{rid}: not private (private={info.private}, anonymous HTTP {anon}); stopping")
        api.upload_large_folder(repo_id=rid, folder_path=folder, repo_type="model", private=True,
                                allow_patterns=names, print_report_every=300)
        verify_upload(api, rid, rec, out, path, anon, resumed)


def verify_upload(api, rid, rec, out, path, anon, resumed):
    """After an upload: private (and anonymous HTTP 401), exactly the staged files, every file's sha256 (a download up to
    1 GiB, else the Hub's LFS sha256 and size), the card metadata; recorded in hf_upload.json. A failure stops the run."""
    from huggingface_hub import hf_hub_download
    names = sorted(rec["files"])
    info = api.model_info(rid, files_metadata=True)
    anon_after = FH.anonymous_status(rid)
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
        want, size = rec["files"][n]["sha256"], rec["files"][n]["bytes"]
        ls = lfs_sha(hub.get(n))
        if ls is not None and size > LFS_ONLY_ABOVE:
            got, method = None, "hub LFS sha256"
            ok = ls == want and getattr(hub.get(n), "size", None) == size
        else:
            with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:     # one file at a time, in RAM
                p = hf_hub_download(rid, n, revision=info.sha, cache_dir=tmp, force_download=True)
                got = FH.sha(p)
            method = "download sha256"
            ok = got == want and (ls is None or ls == want)
        files[n] = dict(sha256=want, bytes=size, hub_size=getattr(hub.get(n), "size", None), hub_lfs_sha256=ls,
                        download_sha256=got, method=method, equal=ok, lfs_equal=None if ls is None else ls == want)
        if not ok:
            problems.append(f"{n}: sha256 differs ({method}: download {got}, lfs {ls})")
        print(f"  {rid} {n}: {method} {'==' if ok else '!='} staged", flush=True)
    commits = api.list_repo_commits(rid)
    out["repos"][rid].update(url=f"https://huggingface.co/{rid}", head=info.sha, private=info.private,
                             resumed=resumed, anonymous_http_before_upload=anon, anonymous_http=anon_after,
                             hub_files=sorted(hub), commits=[dict(id=c.commit_id, title=c.title) for c in commits],
                             card_metadata=card_data, files=files, problems=problems, verified=not problems,
                             uploaded_utc=now())
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"{rid}: head {info.sha} private={info.private} anonymous={anon_after} problems={problems or 'none'}",
          flush=True)
    if problems:
        sys.exit(f"{rid}: verification failed; stopping")


def upload_gentle(rid, start_after, attempts=6, interval=600):
    """The coordinator's rule after the Hub's commit rate limit (2026-10-23): one commit per attempt
    (HfApi.upload_folder, whose commit POST has no automatic retry), at most `attempts` attempts `interval` seconds apart,
    the first not before `start_after` (epoch seconds). A 429 at the last attempt stops the run, to be reported; there is
    no further loop. The repository must be the private one this upload created."""
    from huggingface_hub.errors import HfHubHTTPError
    os.environ.setdefault("HF_XET_CACHE", str(XET_CACHE))
    api = FH.account()
    st = json.loads((RECS / "stage.json").read_text())
    rc = rule_check()
    if rc[rid]["problems"]:
        sys.exit(f"{rid}: gates fail: {rc[rid]['problems']}")
    path = RECS / "hf_upload.json"
    out = json.loads(path.read_text())
    if out["repos"].get(rid, {}).get("verified"):
        print(f"{rid}: already uploaded and verified")
        return
    rec = st["repos"][rid]
    folder = Path(rec["folder"])
    names = sorted(rec["files"])
    if rid not in out["repos"] or not api.repo_exists(rid, repo_type="model"):
        sys.exit(f"{rid}: not a repository this upload created; stopping")
    info = api.model_info(rid)
    anon = FH.anonymous_status(rid)
    if info.private is not True or anon == 200:
        sys.exit(f"{rid}: not private (private={info.private}, anonymous HTTP {anon}); stopping")
    wait = start_after - time.time()
    if wait > 0:
        print(f"{rid}: first commit attempt at {datetime.datetime.fromtimestamp(start_after, datetime.timezone.utc):%H:%M} "
              f"UTC ({wait / 60:.0f} min)", flush=True)
        time.sleep(wait)
    attempts_log = out["repos"][rid].setdefault("gentle_attempts", [])
    for i in range(attempts):
        try:
            api.upload_folder(repo_id=rid, folder_path=folder, repo_type="model", allow_patterns=names,
                              commit_message="Add files")
            attempts_log.append(dict(utc=now(), result="committed"))
            path.write_text(json.dumps(out, indent=1, default=str) + "\n")
            print(f"{rid}: committed at attempt {i + 1}", flush=True)
            break
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            attempts_log.append(dict(utc=now(), result=f"HTTP {code}"))
            path.write_text(json.dumps(out, indent=1, default=str) + "\n")
            print(f"{rid}: attempt {i + 1}: HTTP {code}", flush=True)
            if code != 429:
                raise
            if i == attempts - 1:
                sys.exit(f"{rid}: still HTTP 429 after {attempts} attempts; stopping (to be reported)")
            time.sleep(interval)
    verify_upload(api, rid, rec, out, path, anon, True)


def cards():
    """hf_gptq_focus.cards for these repositories: the Hub's YAML validation of the card it serves, the README equal to
    the staged one, and a local CommonMark + tables render."""
    from huggingface_hub import HfApi, ModelCard
    from markdown_it import MarkdownIt
    up = json.loads((RECS / "hf_upload.json").read_text())
    st = json.loads((RECS / "stage.json").read_text())
    md = MarkdownIt("commonmark").enable("table")
    out = {}
    for rid in [r for r, d in up["repos"].items() if d.get("verified")]:     # an unverified repository has no card yet
        card = ModelCard.load(rid, repo_type="model")
        staged = (Path(st["repos"][rid]["folder"]) / "README.md").read_text()
        try:
            card.validate(repo_type="model")
            valid = True
        except Exception as e:                                    # noqa: BLE001 -- recorded, not raised
            valid = f"{type(e).__name__}: {e}"
        html = md.render(card.text)
        lines = card.text.splitlines()
        n_tables = sum(1 for i, ln in enumerate(lines) if re.match(r"^\|[-:| ]+\|$", ln) and i and lines[i - 1].startswith("|"))
        n_fences = sum(1 for ln in lines if ln.startswith("```")) // 2
        info = HfApi().model_info(rid)
        out[rid] = dict(private_now=info.private, anonymous_http_now=FH.anonymous_status(rid), hub_yaml_valid=valid,
                        readme_equal_to_staged=card.content == staged, card_data=card.data.to_dict(),
                        tables_markdown=n_tables, tables_html=html.count("<table>"), code_fences=n_fences,
                        code_blocks_html=html.count("<pre><code"), h1=html.count("<h1>"),
                        renders=valid is True and card.content == staged and n_tables == html.count("<table>") and
                        n_fences == html.count("<pre><code") and html.count("<h1>") == 1)
        print(rid, {k: v for k, v in out[rid].items() if k != "card_data"}, flush=True)
    (RECS / "cards.json").write_text(json.dumps(out, indent=1) + "\n")


def status():
    """Read-only, without a token: each repository's anonymous API status (401: private, 200: public)."""
    out = {repo_id(m, k): FH.anonymous_status(repo_id(m, k)) for m in ORDER for k in KINDS}
    rec = dict(checked_utc=now(), repos={r: {401: "private", 200: "public"}.get(c, f"HTTP {c}") for r, c in out.items()})
    (RECS / "status.json").write_text(json.dumps(rec, indent=1) + "\n")
    for r, v in rec["repos"].items():
        print(f"{r}: {v}")
    return rec



def nonhf_hits():
    """The codes files' raw-byte hits of a pattern other than the hf_ token pattern (accepted under the earlier rule's
    extension): repository, pattern class, length, tensor and field; the matched bytes are never quoted."""
    st = json.loads((RECS / "stage.json").read_text())
    out = []
    for rid, d in st["repos"].items():
        if d["kind"] != "gptq":
            continue
        raw = [h for h in d["scan"] if h["where"].startswith("raw bytes @")
               and not re.fullmatch(r"hf_[A-Za-z0-9]{20,}", h["match"], re.I)]
        if not raw:
            continue
        path = Path(d["folder"]) / CODES_NAME
        mem, owner = HS.members(path), HS.storage_owner(path)
        for h in raw:
            off = int(h["where"].split("@")[1])
            end = off + len(h["match"].encode())
            m = next((x for x in mem if x[1] <= off and end <= x[2]), None)
            k = m[0].rsplit("/", 1)[-1] if m and "/data/" in m[0] else None
            mod, field, _ = owner.get(k, (None, None, None))
            out.append(dict(repository=rid, pattern_class=H.pattern_class(h["match"]), length=len(h["match"]),
                            tensor=mod, field=field))
    return out


def visibility_now():
    """Read-only, without a token: each repository's anonymous API status; for a public one also its file list, README
    and head against the verified upload."""
    from huggingface_hub import HfApi, hf_hub_download
    st = json.loads((RECS / "stage.json").read_text())
    up = json.loads((RECS / "hf_upload.json").read_text())
    anon = HfApi(token=False)
    out = {}
    for rid in [repo_id(m, k) for m in ORDER for k in KINDS]:
        code = FH.anonymous_status(rid)
        ur = up["repos"].get(rid, {})
        row = dict(anonymous_http=code, visibility={401: "private", 200: "public"}.get(code, f"HTTP {code}"),
                   upload_verified=ur.get("verified"), verified_head=ur.get("head"), checked_utc=now())
        if code == 200:
            info = anon.model_info(rid)
            files = sorted(x.rfilename for x in info.siblings)
            with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
                p = hf_hub_download(rid, "README.md", revision=info.sha, cache_dir=tmp, token=False, force_download=True)
                readme = FH.sha(p)
            row.update(head=info.sha, head_equal_verified=info.sha == ur.get("head"),
                       file_list_equal_staged=set(files) - {".gitattributes"} == set(st["repos"][rid]["files"]),
                       readme_equal_staged=readme == st["repos"][rid]["files"]["README.md"]["sha256"])
        out[rid] = row
        print(rid, {k: v for k, v in row.items() if k != "checked_utc"}, flush=True)
    (RECS / "visibility.json").write_text(json.dumps(dict(checked_utc=now(), repos=out), indent=1) + "\n")
    return out


def hf_record():
    """results/paper_eval/hf_calib1k/: HF_UPLOAD.md, the records (stage.json and the hit records sanitized: raw-byte
    matches by pattern class only) and each repository's card and summary."""
    st = json.loads((RECS / "stage.json").read_text())
    up = json.loads((RECS / "hf_upload.json").read_text())
    cj = json.loads((RECS / "cards.json").read_text())
    tok = json.loads((RECS / "token_check.json").read_text())
    rc = rule_check()
    (RECS / "rule_check.json").write_text(json.dumps(rc, indent=1) + "\n")
    nh = nonhf_hits()
    (RECS / "nonhf_hits.json").write_text(json.dumps(nh, indent=1) + "\n")
    vis = visibility_now()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage.json").write_text(json.dumps(H.sanitized(st), indent=1, ensure_ascii=False, default=str) + "\n")
    for name in ("hf_upload.json", "cards.json", "rule_check.json", "token_check.json", "nonhf_hits.json", "visibility.json"):
        shutil.copyfile(RECS / name, OUT / name)
    for rid, d in st["repos"].items():
        c = OUT / "cards" / rid.split("/", 1)[1]
        c.mkdir(parents=True, exist_ok=True)
        for f in ("README.md", "ppl_summary.json"):
            shutil.copyfile(Path(d["folder"]) / f, c / f)
    checked = json.loads((RECS / "visibility.json").read_text())["checked_utc"]
    L = ["# Hugging Face upload: the part AA 1,024-window artifacts (part AB, 2026-10-23)", "",
         "The user's decision (relayed by the coordinator, 2026-10-23): the eight part AA models' 1,024-window calibrations "
         "as 24 new **private** repositories under `edgeai-lab`: `<Base>-FlipQuant-Calib1k` (the FQ16-1k map), "
         "`<Base>-FOCUS-NVFP4-Calib1k` (the FOCUS-1k state) and `<Base>-GPTQ-BF16Calib-NVFP4-Calib1k` (the GPTQ‡-1k "
         "codes). PROTOCOL.md part AB; script `experiments/paper_eval/hf_calib1k.py`.",
         "- **Account:** checked first (`edgeai-lab`); no token was printed or changed. No existing repository was touched.",
         "- **Order of operations:** each repository was created private and checked private (anonymous HTTP 401) before "
         "any file went up, then uploaded and verified.",
         "- **Verification:** every file's sha256 against the staged file's: a download into /dev/shm for files up to "
         "1 GiB, the Hub's server-side LFS sha256 and size for larger ones.", "",
         "## Uploaded", "", f"Visibility: at {checked}, without a token.", "",
         "| repository | kind | files | sha256 = staged (method) | visibility now | anonymous at upload | head | card YAML "
         "(Hub) | README = staged |", "|---|---|---:|---|---|---:|---|---|---|"]
    for rid in [repo_id(m, k) for m in ORDER for k in KINDS]:
        d = up["repos"].get(rid, {})
        if not d.get("verified"):
            L.append(f"| `{rid}` | {st['repos'][rid]['kind']} | — | not committed | {vis[rid]['visibility']} | "
                     f"{d.get('anonymous_http_before_upload', '—')} | — | — | — |")
            continue
        n_dl = sum(f["method"] == "download sha256" for f in d["files"].values())
        n_lfs = len(d["files"]) - n_dl
        ok = all(f["equal"] for f in d["files"].values())
        c = cj.get(rid, {})
        L.append(f"| `{rid}` | {st['repos'][rid]['kind']} | {len(d['files'])} | {'equal' if ok else 'DIFFERS'} "
                 f"({n_dl} downloaded" + (f", {n_lfs} by Hub LFS sha256" if n_lfs else "") + f") | {vis[rid]['visibility']} | "
                 f"{d['anonymous_http']} | `{d['head'][:12]}` | {c.get('hub_yaml_valid')} | {c.get('readme_equal_to_staged')} |")
    L += ["", "## Sources", "",
          "Each source was verified against its part AA job record by a fresh sha256 of the whole file before staging.",
          "- **Maps:** `fq16_1024/map.pt`, against `map_file_sha256`. The meta was cleaned as for the earlier FlipQuant "
          "repositories: the run and calibration-record paths are relative, the GitHub owner is removed from the "
          "extension reference, and the reference implementation's branch and commit are removed. The tiles and every "
          "tensor record are bit-identical; both sha256 are in each card.",
          "- **FOCUS states:** as is, against `state_sha256` (the pilot's two in RUN, the others on /vault).",
          "- **GPTQ‡ codes:** as is, against `cache_sha256`. The code digest equals the calibration's, the key is the "
          "main-table variant at 1,024 windows, and the card's PyTorch snippet reproduces FlipQuant's dequantization bit "
          "for bit.",
          "- **Staging:** the cards, summaries, license files and cleaned maps were written to /home/dev/calib1k_hf. The "
          "FOCUS states and the codes files were symbolic links to their verified sources (/home's 10 GB floor), so "
          "every scan, hash and upload read the source.", "",
          "## Scans", "",
          "The earlier uploads' patterns (local paths, the host, user names, private repository names, tokens, e-mail "
          "addresses, pointers to a reference implementation) over every staged file: the maps' metadata strings, tile "
          "names and raw bytes; the states' metadata, record names and raw bytes; the codes files' settings strings, "
          "module names, record names and raw bytes; every line of the text files.",
          "- **Maps and FOCUS states:** no hit.",
          "- **Text files:** only the kinds accepted before (Meta's reporting address inside the official "
          "`USE_POLICY.md`; the words “commit” / “branch” in the FlipQuant card's own sentences).",
          "- **Codes files:** the GPTQ upload's binary rule (2026-10-15, with its extension). A raw-byte hit is accepted "
          "only if it lies inside a tensor-storage member backing a codes tensor (`q` or `s`), the file's pickle, "
          "metadata strings and record names have no hit, the file's sha256 equals its record, and the exact-token "
          "check is clean.", "",
          "| repository | raw-byte hits | inside `s` | inside `q` | metadata hits | within the rule |",
          "|---|---:|---:|---:|---:|---|"]
    for rid, d in st["repos"].items():
        if d["kind"] != "gptq":
            continue
        bf = d["raw_hits"]["by_field"]
        meta = len([h for h in d["scan"] if h["file"].endswith(".pt") and not h["where"].startswith("raw bytes")])
        L.append(f"| `{rid.split('/')[1]}` | {d['raw_hits']['raw_hits']:,} | {bf.get('s', 0):,} | {bf.get('q', 0):,} | "
                 f"{meta} | {'yes' if not rc[rid]['problems'] else 'NO'} |")
    L += ["", "Hits of a pattern other than the token pattern (accepted under the extension; the matched bytes are not "
          "quoted):", ""]
    L += [f"- `{h['repository'].split('/')[1]}`: a {h['pattern_class']} ({h['length']} bytes) in `{h['tensor']}` "
          f"`{h['field']}`" for h in nh]
    L += ["", f"**Exact-token check:** every staged file's bytes were searched in-process for the {tok['tokens_checked']} "
          "Hugging Face tokens in this machine's token files and their 15-character prefixes; the tokens were never "
          f"printed, logged or written: {'0 hits' if tok['all_zero'] else 'HITS'} on all {len(tok['files'])} files.", "",
          "## Events", "",
          "- **Commit rate limit.** The 24th repository's commit was refused with HTTP 429 (\"repository commit creation, "
          "128 per hour\") after 46 commits by this upload; the limit is treated as account-wide. upload_large_folder "
          "retried about twice a second. The coordinator authorized stopping the uploader (12:02 UTC), then one commit "
          "attempt at least 65 minutes later, at most one per 10 minutes and six in all.",
          "- **Private storage limit.** The first attempt (13:08 UTC) was refused with HTTP 400: \"Private storage limit "
          "reached for user edgeai-lab\" (the 23 verified repositories hold 86.7 GB). No further attempt was made; the "
          "decision went to the user. After the 23 were made public (below), the 24th was uploaded privately with the "
          "same throttled single-commit retry.",
          "- **Commit attempts of the 24th** (hf_upload.json `gentle_attempts`): "
          + "; ".join(f"{a['utc']}: {a['result']}" for a in up["repos"].get(repo_id("qwen3-8b", "gptq"), {})
                      .get("gentle_attempts", [])) + ".", ""]
    L += ["## Licenses", "",
          "As in the earlier repositories of each model: the Llama 3.2 / 3.1 Community License files (`LICENSE`, "
          "`USE_POLICY.md`, `NOTICE`; “Built with Llama”; every name starts with “Llama”), the NVIDIA Open Model License "
          "and its `NOTICE` (Nemotron), Apache-2.0 (Qwen3; Mistral-7B-Instruct-v0.3 and Qwen3.8-27B with the apache.org "
          "text, as their base repositories have no license file) and MIT (Phi-4).", "",
          "## Visibility", "",
          "**The user's request:** make the uploaded repositories public. **Who made the change:** the coordinator, in "
          "its own session, at the user's direct request (`update_repo_settings(private=False)`), first for the 23 "
          "verified repositories and later for the 24th once its upload was verified. This session made nothing "
          "public and changed no visibility.",
          f"- **Visibility now:** this session's read-only check without a token at {checked} (`visibility.json`); the "
          "table above lists each repository's state.",
          "- **For a public repository** the check also compares, without a token, its file list with the staged list, "
          "its README with the staged one and its head with the verified upload's.", "",
          "Files here: `stage.json` (sanitized), `hf_upload.json`, `cards.json`, `rule_check.json`, `token_check.json`, "
          "`nonhf_hits.json`, `visibility.json`, `cards/<repo>/{README.md, ppl_summary.json}`.", ""]
    (OUT / "HF_UPLOAD.md").write_text("\n".join(L))
    print("\n".join(L[:30]))


# ------------------------------------------------------------------------- amendment 1: the 8x64 / 256x64 maps (PUBLIC)
A1_COMMIT = "5dcb1ff"                    # paper-eval: part AA's record with amendment 6
A1_UNITS = ("8x64", "16x64", "256x64")
A1_NEW = ("8x64", "256x64")
A1_ROW = {"8x64": "fq8_1024", "16x64": "fq16_1024", "256x64": "fq256_1024"}
A1_REF = {"8x64": "fq-8x64", "16x64": "fq16_256", "256x64": "fq-256x64"}
A1_TAG = {"8x64": "FQ8", "16x64": "FQ16", "256x64": "FQ256"}
A1_MESSAGE = "Add the 8x64 and 256x64 maps (1,024 windows); update the card"


@functools.lru_cache(maxsize=None)
def record_a1():
    return T.git_json(A1_COMMIT, "results/paper_eval/calib_size/calib_size.json")


def a1_values(model, name):
    return record_a1()["models"][model]["values"][name]


def a1_pair(model, label):
    return record_a1()["models"][model]["paired"][label]


def a1_share(model, unit, n):
    method = f"FlipQuant {unit}"
    c = [x for x in record_a1()["models"][model]["cost"] if x["method"] == method and x["windows"] == n]
    assert len(c) == 1, (model, unit, n)
    return c[0]["e0m3_share"]


def a1_rows(model):
    out = [("BF16", "bf16"), ("FourOverSix (reference)", "fo6")]
    for u in A1_UNITS:
        out += [(f"FlipQuant {u}, 256 windows (`{repo256(model, 'flipquant').split('/')[1]}`)", A1_REF[u]),
                (f"FlipQuant {u}, 1,024 windows (**this repository**)", A1_ROW[u])]
    return out


def a1_paired(model):
    out = []
    for u in A1_UNITS:
        t = A1_TAG[u]
        out += [(f"{u} vs FourOverSix", a1_pair(model, f"{t}-1k vs FourOverSix")),
                (f"{u} vs its 256-window version", a1_pair(model, f"{t}-1k vs {t}-256"))]
        if u != "16x64":
            out.append((f"{u} vs the 16x64 map (this repository)", a1_pair(model, f"{t}-1k vs FQ16-1k")))
    return out


def a1_note_256(model):
    c = a1_pair(model, "FQ256-1k vs FQ16-1k")
    w, c4 = c["nll"]["wiki"], c["nll"]["c4"]
    worse = all(x["significant"] and x["delta"] > 0 for x in (w, c4))
    s256, s16 = 100 * a1_share(model, "256x64", 1024), 100 * a1_share(model, "16x64", 1024)
    nll = f"ΔNLL WikiText-2 {fmt_nll(w)}, C4 {fmt_nll(c4)}"
    sig = [n for n, x in (("WikiText-2", w), ("C4", c4)) if x["significant"]]
    phrase = (f"higher, beyond 2 SE on both corpora ({nll})" if worse else
              f"not significantly different ({nll}; neither beyond 2 SE)" if not sig else
              f"different ({nll}; beyond 2 SE on {' and '.join(sig)})")
    text = (f"**The 256x64 map** has the highest E0M3 share of the three ({s256:.2f} %; 16x64: {s16:.2f} %). Against the "
            f"16x64 map its PPL is {phrase}")
    P = record_a1()["pooled"]
    v16, v256 = P["FQ256-1k vs FQ16-1k"], P["FQ256-1k vs FQ256-256"]

    def pp(x):
        return f"{x['diff']:+.2f}".replace("-", "−") + ("*" if x["p"] < 0.05 else " (n.s.)")
    return (text + f"; GSM8K {fmt_pair(c['gsm8k'])}, zero-shot {fmt_pair(c['zeroshot'])}. Pooled over the study's eight "
            f"models, 256x64 at 1,024 windows is the weakest of the three tiles (GSM8K {pp(v16['gsm8k'])}, zero-shot "
            f"{pp(v16['zeroshot'])} against 16x64), and more windows did not improve it over its 256-window version (GSM8K "
            f"{pp(v256['gsm8k'])}).")


def a1_results_section(model):
    L = ["## Results", "",
         "From the FlipQuant paper's calibration-size study. The numbers are also in `ppl_summary.json`.", ""]
    L += eval_text(model)
    L += ["", "| method | zero-shot Avg. (%) | GSM8K strict (%) | WikiText-2 PPL | C4 PPL |", "|---|---:|---:|---:|---:|"]
    bc = (record_a1()["models"][model].get("baseline_check") or {}).get("bf16", {})
    mark = bool((bc.get("gsm8k") or {}).get("batch_mark"))
    for label, name in a1_rows(model):
        v = a1_values(model, name)
        g = f"{v['gsm8k'][0]:.2f} ± {v['gsm8k'][1]:.2f}" + (" ◊" if name == "bf16" and mark else "")
        L.append(f"| {label} | {v['zeroshot']['avg']:.2f} | {g} | {v['ppl']['wiki']:.3f} | {v['ppl']['c4']:.3f} |")
    if mark:
        L += ["", f"◊ BF16's GSM8K ran at batch {bc['gsm8k']['batch']}, the quantized rows at {bc['gsm8k']['batch_next']}."]
    L += ["", "Zero-shot accuracy (%) per task:", "",
          "| method | " + " | ".join(n for _, n in TASKS) + " |", "|---|" + "---:|" * len(TASKS)]
    for label, name in a1_rows(model):
        z = a1_values(model, name)["zeroshot"]
        L.append(f"| {label} | " + " | ".join(f"{z[t]:.2f}" for t, _ in TASKS) + " |")
    L += ["", "**The maps against their references, paired.**", "",
          "- Zero-shot: pooled over all 21,848 questions of the six tasks.",
          "- GSM8K: strict-match over its 1,319 problems.",
          "- Format: difference in points (questions only this map gets right / only the reference gets right, exact "
          "McNemar p).",
          "- ΔNLL: the mean paired per-window difference in nats per token, ± 2 SE; negative is lower NLL.",
          "- `*`: p < 0.05, or |Δ| > 2 SE.", "",
          "| comparison | zero-shot | GSM8K | ΔNLL WikiText-2 | ΔNLL C4 |", "|---|---|---|---|---|"]
    for label, c in a1_paired(model):
        L.append(f"| {label} | {fmt_pair(c['zeroshot'])} | {fmt_pair(c['gsm8k'])} | {fmt_nll(c['nll']['wiki'])} | "
                 f"{fmt_nll(c['nll']['c4'])} |")
    return L


def a1_cost(model, unit):
    tag = {"8x64": "fq8", "16x64": "fq16", "256x64": "fq256"}[unit]
    j = json.loads((CS.ROOT / model / f"{tag}_1024.job.json").read_text())
    t = j["trainer"]
    return dict(steps=5 * 1024 // 8, job_seconds=j["job_seconds"], training_seconds=t["training_seconds"],
                setup_seconds=t["setup_seconds"], gpu_peak_allocated_gib=t["gpu_peak_allocated_gib"][0],
                gpu_peak_reserved_gib=t["gpu_peak_reserved_gib"][0], cpu_peak_rss_gib=t["cpu_peak_rss_gib"],
                device_peak_mib=j["device_peak_mib"], teacher_bytes=(j.get("teacher_storage") or {}).get("bytes"),
                teacher_devices=(j.get("teacher_storage") or {}).get("devices"), e0m3_share=j["e0m3_share"])


def a1_card(model, maps, costs, files):
    key, host = model, model in CS.HOST_TEACHER
    s = maps["16x64"]["settings"]
    L = front(model, LH.FQ_TAGS) + [f"# {title(model)} FlipQuant MixFP4 maps, 1,024 calibration windows", ""]
    L += head_lines(model)
    L += ["## What this is", "",
          f"Format maps for [{H.MODELS[model][1]}](https://huggingface.co/{H.MODELS[model][1]}) at revision "
          f"`{H.MODELS[model][2]}`, calibrated with FlipQuant (TM-OPT+TC) on **1,024 calibration windows**, four times "
          f"the release's 256 (`{repo256(model, 'flipquant')}`). A map holds one bit per weight tile of every quantized "
          "linear layer (all text-model linears except the LM head): `True` stores the tile's 4-bit elements as E0M3 "
          "(uniform levels 0..7), `False` as E2M1 (FourOverSix). Everything else is NVFP4: 16-element blocks with an "
          "E4M3 scale and a per-tensor FP32 scale. The maps hold no weights: they are applied to the base model's BF16 "
          "checkpoint when it is loaded.", "",
          "| file | tile (output rows × input columns) | E0M3 tiles | sha256 |", "|---|---|---:|---|"]
    for u in A1_UNITS:
        m = maps[u]
        L.append(f"| `flipquant_{u}.pt` | {u.replace('x', ' × ')} | {m['e0m3_tiles']:,} of {m['tiles']:,} "
                 f"({100 * m['e0m3_tiles'] / m['tiles']:.2f} %) | `{m['sha256_uploaded']}` |")
    L += ["", "8x64 is the smallest tile one MMA operand covers on sm_100 (and with weights as the B operand on sm_120); "
          "16x64 and 256x64 are coarser.", "",
          "With more windows every map elects more E0M3 tiles. The E0M3 shares of the release's 256-window maps are "
          + ", ".join(f"{u} {100 * a1_share(model, u, 256):.2f} %" for u in A1_UNITS) + ".", "",
          a1_note_256(model), "", LH.FQ_CODE, "",
          "## Calibration", "",
          f"`python -m calibration.train_map --model {key} --unit <unit> --epochs 5 --teacher-topk 1000 --fit-windows 1024"
          + (" --teacher-host" if host else "") + "`, the same settings and the same 1,024 windows for the three tiles "
          "(FlipQuant revisions: " + ", ".join(f"{u} `{maps[u]['flipquant_commit']}`" for u in A1_UNITS) + "):", "",
          "- TM-OPT+TC: one logit per tile, a straight-through hard map, KL to the BF16 model's top-1000 tokens per "
          "position plus one tail bucket, per-token FourOverSix activations.",
          f"- Adam, learning rate {s['lr']}, initial logit {format(s['init_logit'], 'g').replace('-', '−')} (every tile "
          f"starts as E2M1), seed {s['seed']}, {s['batch'] * s['accum']} sequences per step"
          + (f" ({s['batch']} × {s['accum']} accumulated)" if s["accum"] > 1 else "")
          + f", 5 epochs: **{costs['16x64']['steps']} steps** for each tile (the release: 160). Deterministic.",
          f"- Fit set: {windows_text(model)}",
          "- **These are not the paper's settings** (20 epochs, 128 windows, the full-vocabulary teacher). The maps' "
          "`meta` records them as `non_default_settings`.", ""]
    if host:
        L += ["The BF16 teacher's top-1000 targets for 1,024 windows do not fit next to this model on the GPU, so they "
              "are kept in host memory (`--teacher-host`). On Llama-3.2-1B-Instruct, the same option gives the GPU-teacher "
              "map bit for bit.", ""]
    L += ["Cost on one NVIDIA RTX PRO 6000 Blackwell (sm_120), per unit:", "",
          "| unit | setup | training | train_map end to end | GPU peak allocated | GPU peak reserved | trainer peak RSS | "
          "teacher (top-1000) | device memory peak (nvidia-smi, 2 s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for u in A1_UNITS:
        c = costs[u]
        L.append(f"| {u} | {LH.fmt(c['setup_seconds'])} | {LH.fmt(c['training_seconds'])} | {LH.fmt(c['job_seconds'])} | "
                 f"{c['gpu_peak_allocated_gib']:.1f} GiB | {c['gpu_peak_reserved_gib']:.1f} GiB | {c['cpu_peak_rss_gib']:.1f} GiB "
                 f"| {c['teacher_bytes'] / 2 ** 30:.2f} GiB ({', '.join(c['teacher_devices'] or [])}) | "
                 f"{c['device_peak_mib'] / 1024:.1f} GiB |")
    L += [""] + a1_results_section(model)
    L += ["", "## How to use", "",
          LH.FQ_CODE + " With it, on an sm_120 GPU (RTX PRO 6000 / RTX 50-series class), build the kernels once, then "
          "evaluate with the native kernels (any of the three maps):", "", "```bash", "bash kernels/sm120/build_deploy.sh",
          f"python -m evaluation.ppl --model {key} --mode native --weight mixfp4 --map flipquant_16x64.pt --paper-convention",
          "```", "", "On other GPUs, the same map with simulated (fake) quantization:", "", "```bash",
          f"python -m evaluation.ppl --model {key} --mode fake --weight mixfp4 --map flipquant_16x64.pt --paper-convention",
          "```", "", "The native command produced the PPL above.", ""]
    L += license_section(model, "flipquant", plural=True) + [""]
    L += files_section(model, "flipquant", files, [
        "- `flipquant_<unit>.pt`: the maps (`flipquant-map/1`: `{format, unit, tiles, meta}`; `flipquant.maps.load` reads "
        "them)."])
    L += ["", "**Provenance of the map files.** They are the calibration outputs with only their `meta` changed: the run "
          "directory and calibration-record paths are relative instead of local absolute paths, the data-extension code "
          "reference names the file without its repository owner, and the internal branch and commit of the reference "
          "implementation are removed (the FlipQuant commit is kept). The tiles are bit-identical. sha256 of the "
          "calibration output → this repository's file:", ""]
    L += [f"- `flipquant_{u}.pt`: `{maps[u]['sha256_source']}` → `{maps[u]['sha256_uploaded']}`" for u in A1_UNITS]
    L.append("")
    return "\n".join(L)


def a1_summary(model, maps, costs):
    return dict(model=H.MODELS[model][1], revision=H.MODELS[model][2], flipquant_model_key=model,
                method="FlipQuant (TM-OPT+TC), 8x64 / 16x64 / 256x64 tiles, 1,024 calibration windows",
                maps={f"flipquant_{u}.pt": dict(sha256=maps[u]["sha256_uploaded"],
                                                calibration_output_sha256=maps[u]["sha256_source"],
                                                e0m3_tiles=maps[u]["e0m3_tiles"], tiles=maps[u]["tiles"],
                                                e0m3_share=maps[u]["e0m3_tiles"] / maps[u]["tiles"],
                                                e0m3_share_256_windows=a1_share(model, u, 256), modules=maps[u]["modules"])
                      for u in A1_UNITS},
                evaluation=evaluation_summary(model, "flipquant"),
                results=dict(rows={name: dict(label=label, zeroshot=a1_values(model, name)["zeroshot"],
                                              gsm8k_strict=dict(accuracy=a1_values(model, name)["gsm8k"][0],
                                                                two_se=a1_values(model, name)["gsm8k"][1]),
                                              gsm8k_flexible=dict(accuracy=a1_values(model, name)["gsm8k_flex"][0],
                                                                  two_se=a1_values(model, name)["gsm8k_flex"][1]),
                                              ppl=a1_values(model, name)["ppl"]) for label, name in a1_rows(model)},
                             paired={label: dict(zeroshot=c["zeroshot"], gsm8k_strict=c["gsm8k"], gsm8k_flexible=c["gsm8k_flex"],
                                                 nll={k: dict(delta_nll=x["delta"], two_se=x["two_se"], n=x["n"],
                                                              significant=x["significant"]) for k, x in c["nll"].items()})
                                     for label, c in a1_paired(model)},
                             source="the FlipQuant paper's calibration-size study"),
                calibration=dict(settings=maps["16x64"]["settings"], steps=costs["16x64"]["steps"],
                                 teacher_host=maps["16x64"]["teacher_host"], windows=windows_facts(model),
                                 flipquant_commits={u: maps[u]["flipquant_commit"] for u in A1_UNITS},
                                 cost={u: {k: v for k, v in costs[u].items() if k != "e0m3_share"} for u in A1_UNITS}))


def a1_stage(models=None):
    """The two new maps of each FlipQuant repository: sources against their job records, staged with the meta cleaned;
    the 16x64 map re-staged from its own source the same way (it must reproduce the uploaded file byte for byte); the card
    and summary regenerated; scans. RECS/stage_a1.json."""
    path = RECS / "stage_a1.json"
    st0 = json.loads((RECS / "stage.json").read_text())
    rec = json.loads(path.read_text()) if path.exists() else dict(namespace=NS, part="AB amendment 1", cs_commit=A1_COMMIT,
                                                                  repos={})
    for model in models or ORDER:
        rid = repo_id(model, "flipquant")
        if rid in rec["repos"]:
            continue
        dst = Path(st0["repos"][rid]["folder"])
        maps, sources = {}, {}
        for u in A1_UNITS:
            tag = {"8x64": "fq8", "16x64": "fq16", "256x64": "fq256"}[u]
            src = CS.ROOT / model / f"{tag}_1024" / "map.pt"
            want = json.loads((CS.ROOT / model / f"{tag}_1024.job.json").read_text())["map_file_sha256"]
            got = FH.sha(src)
            if got != want:
                sys.exit(f"{rid} {u}: the source's sha256 {got[:12]} != its job record's {want[:12]}")
            sources[u] = dict(path=str(src), sha256=got, recorded_sha256=want, equal=True)
            if u in A1_NEW:
                maps[u] = stage_map(model, src, dst, u)
            else:                     # the uploaded 16x64 map stays; its re-staging in a temporary directory must equal it
                with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
                    maps[u] = stage_map(model, src, Path(tmp), u)
        if maps["16x64"]["sha256_uploaded"] != st0["repos"][rid]["files"][MAP_NAME]["sha256"] or \
                FH.sha(dst / MAP_NAME) != st0["repos"][rid]["files"][MAP_NAME]["sha256"]:
            sys.exit(f"{rid}: the re-staged 16x64 map differs from the uploaded one")
        costs = {u: a1_cost(model, u) for u in A1_UNITS}
        (dst / "ppl_summary.json").write_text(json.dumps(a1_summary(model, maps, costs), indent=1, ensure_ascii=False) + "\n")
        files = {n: f for n, f in staged_files(dst).items() if n != "README.md"}
        (dst / "README.md").write_text(a1_card(model, maps, costs, files))
        hits = []
        for u in A1_UNITS:
            hits += LH.scan_map(dst / f"flipquant_{u}.pt")
        hits += text_hits(dst, {f"flipquant_{u}.pt" for u in A1_UNITS})
        rec["repos"][rid] = dict(kind="flipquant", model=model, folder=str(dst), base_model=H.MODELS[model][1],
                                 license=H.LICENSES[model]["front"], sources=sources,
                                 maps={u: {k: v for k, v in m.items() if k not in ("settings", "teacher")} for u, m in maps.items()},
                                 files=staged_files(dst), changed=["flipquant_8x64.pt", "flipquant_256x64.pt", "README.md",
                                                                   "ppl_summary.json"],
                                 scan=hits, staged_utc=now())
        print(f"{rid}: scan hits {len(hits)} ({len(unaccepted(hits))} not accepted)", flush=True)
        path.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str) + "\n")


def a1_tokens():
    full, prefix = TC.needles()
    if not full:
        sys.exit("no token found in the token files; nothing to check against")
    st = json.loads((RECS / "stage_a1.json").read_text())
    out = dict(tokens_checked=len(full), files={})
    for rid, d in st["repos"].items():
        for name in sorted(d["files"]):
            n_full, n_prefix = TC.count(Path(d["folder"]) / name, full, prefix)
            out["files"][f"{rid}/{name}"] = dict(exact_token_hits=n_full, prefix_hits=n_prefix)
    out["all_zero"] = all(v["exact_token_hits"] == 0 and v["prefix_hits"] == 0 for v in out["files"].values())
    del full, prefix
    out["checked_utc"] = now()
    (RECS / "token_check_a1.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(dict(tokens_checked=out["tokens_checked"], files=len(out["files"]), all_zero=out["all_zero"])))


def a1_gates():
    st = json.loads((RECS / "stage_a1.json").read_text())
    tok = json.loads((RECS / "token_check_a1.json").read_text())
    out = {}
    for rid, d in st["repos"].items():
        problems = []
        if [h for h in d["scan"] if h["file"].endswith(".pt")]:
            problems.append("map-file hits")
        if unaccepted([h for h in d["scan"] if not h["file"].endswith(".pt")]):
            problems.append(f"text hits {[(h['file'], h['match']) for h in unaccepted(d['scan'])][:3]}")
        tk = {n: v for n, v in tok["files"].items() if n.startswith(rid + "/")}
        if len(tk) != len(d["files"]) or any(v["exact_token_hits"] or v["prefix_hits"] for v in tk.values()):
            problems.append("exact-token check")
        for n, f in d["files"].items():
            if FH.sha(Path(d["folder"]) / n) != f["sha256"]:
                problems.append(f"{n} changed since staging")
        out[rid] = problems
    return out


def a1_check():
    """Read-only: the account, each repository this part created and now public, the gates."""
    api = FH.account()
    st = json.loads((RECS / "stage_a1.json").read_text())
    assert set(st["repos"]) == {repo_id(m, "flipquant") for m in ORDER}, sorted(st["repos"])
    up = json.loads((RECS / "hf_upload.json").read_text())
    bad = {}
    for rid in st["repos"]:
        info = api.model_info(rid)
        if not (up["repos"].get(rid, {}).get("verified") and info.private is False and FH.anonymous_status(rid) == 200):
            bad[rid] = f"verified {up['repos'].get(rid, {}).get('verified')}, private {info.private}"
    gates = a1_gates()
    for rid, pr in gates.items():
        print(rid, "ok" if not pr else pr)
    if bad or any(gates.values()):
        sys.exit(f"stopping: {bad or {r: p for r, p in gates.items() if p}}")
    print("all gates pass")


def a1_verify(rid, rec, head_before):
    """Without a token: the file list, every file's sha256 by download, the card metadata, the head and its commit."""
    from huggingface_hub import HfApi, hf_hub_download
    anon = HfApi(token=False)
    info = anon.model_info(rid, files_metadata=True)
    hub = {x.rfilename: x for x in info.siblings}
    names = sorted(rec["files"])
    problems = []
    if set(hub) - {".gitattributes"} != set(names):
        problems.append(f"file list {sorted(hub)}")
    files = {}
    for n in names:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            p = hf_hub_download(rid, n, revision=info.sha, cache_dir=tmp, token=False, force_download=True)
            got = FH.sha(p)
        ok = got == rec["files"][n]["sha256"]
        files[n] = dict(sha256=rec["files"][n]["sha256"], download_sha256=got, equal=ok, method="download sha256 (no token)")
        if not ok:
            problems.append(f"{n}: sha256 differs")
        print(f"  {rid} {n}: download sha256 (no token) {'==' if ok else '!='} staged", flush=True)
    card = info.card_data.to_dict() if info.card_data else {}
    if any(card.get(k) != v for k, v in dict(rec["license"], base_model=rec["base_model"]).items()):
        problems.append(f"card metadata {card}")
    commits = anon.list_repo_commits(rid)
    if info.sha == head_before or not commits or commits[0].title != A1_MESSAGE:
        problems.append(f"head {info.sha[:12]} / last commit {commits[0].title if commits else None}")
    return dict(head=info.sha, head_before=head_before, private=info.private, files=files, card_metadata=card,
                commits=[dict(id=c.commit_id, title=c.title) for c in commits], problems=problems, verified=not problems,
                verified_utc=now())


def a1_upload(only=None, attempts=6, interval=600):
    """One commit per repository (the two maps, README.md, ppl_summary.json), at most one attempt per `interval` seconds
    and `attempts` per repository on HTTP 429; any other error stops. Then a1_verify."""
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import HfHubHTTPError
    os.environ.setdefault("HF_XET_CACHE", str(XET_CACHE))
    api = FH.account()
    st = json.loads((RECS / "stage_a1.json").read_text())
    gates = a1_gates()
    if any(gates.values()):
        sys.exit(f"gates fail; nothing uploaded: {gates}")
    path = RECS / "upload_a1.json"
    out = json.loads(path.read_text()) if path.exists() else dict(account=NS, repos={})
    order = [repo_id(m, "flipquant") for m in ORDER]
    if only:
        order = [r for r in order if r in only]
    for rid in order:
        rec = st["repos"][rid]
        if out["repos"].get(rid, {}).get("verified"):
            print(f"{rid}: already updated and verified")
            continue
        info = api.model_info(rid)
        head_before = info.sha
        if info.private is not False:
            sys.exit(f"{rid}: not public as expected (private={info.private}); stopping (no visibility change here)")
        folder = Path(rec["folder"])
        ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(folder / n)) for n in rec["changed"]]
        row = out["repos"].setdefault(rid, dict(head_before=head_before, attempts=[]))
        for i in range(attempts):
            try:
                ci = api.create_commit(repo_id=rid, repo_type="model", operations=ops, commit_message=A1_MESSAGE)
                row["attempts"].append(dict(utc=now(), result="committed", oid=ci.oid))
                path.write_text(json.dumps(out, indent=1, default=str) + "\n")
                break
            except HfHubHTTPError as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                row["attempts"].append(dict(utc=now(), result=f"HTTP {code}"))
                path.write_text(json.dumps(out, indent=1, default=str) + "\n")
                print(f"{rid}: attempt {i + 1}: HTTP {code}", flush=True)
                if code != 429:
                    raise
                if i == attempts - 1:
                    sys.exit(f"{rid}: still HTTP 429 after {attempts} attempts; stopping (to be reported)")
                time.sleep(interval)
        row.update(a1_verify(rid, rec, head_before))
        path.write_text(json.dumps(out, indent=1, default=str) + "\n")
        print(f"{rid}: head {row['head'][:12]} problems={row['problems'] or 'none'}", flush=True)
        if row["problems"]:
            sys.exit(f"{rid}: verification failed; stopping")


def a1_cards():
    """Without a token: the Hub's YAML validation of each updated card, README = staged, a local render."""
    from huggingface_hub import ModelCard
    from markdown_it import MarkdownIt
    st = json.loads((RECS / "stage_a1.json").read_text())
    md = MarkdownIt("commonmark").enable("table")
    out = {}
    for rid, d in st["repos"].items():
        card = ModelCard.load(rid, repo_type="model", token=False)
        staged = (Path(d["folder"]) / "README.md").read_text()
        try:
            card.validate(repo_type="model")
            valid = True
        except Exception as e:                                    # noqa: BLE001 -- recorded, not raised
            valid = f"{type(e).__name__}: {e}"
        html = md.render(card.text)
        lines = card.text.splitlines()
        n_tables = sum(1 for i, ln in enumerate(lines) if re.match(r"^\|[-:| ]+\|$", ln) and i and lines[i - 1].startswith("|"))
        n_fences = sum(1 for ln in lines if ln.startswith("```")) // 2
        out[rid] = dict(hub_yaml_valid=valid, readme_equal_to_staged=card.content == staged, tables_markdown=n_tables,
                        tables_html=html.count("<table>"), code_fences=n_fences, code_blocks_html=html.count("<pre><code"),
                        h1=html.count("<h1>"),
                        renders=valid is True and card.content == staged and n_tables == html.count("<table>")
                        and n_fences == html.count("<pre><code") and html.count("<h1>") == 1)
        print(rid, out[rid], flush=True)
    (RECS / "cards_a1.json").write_text(json.dumps(out, indent=1) + "\n")


def a1_record():
    """HF_UPLOAD.md's addendum (replaced, not repeated, when written again), the amendment's records, the cards."""
    st = json.loads((RECS / "stage_a1.json").read_text())
    up = json.loads((RECS / "upload_a1.json").read_text())
    cj = json.loads((RECS / "cards_a1.json").read_text())
    tok = json.loads((RECS / "token_check_a1.json").read_text())
    vis = {rid: FH.anonymous_status(rid) for rid in st["repos"]}
    checked = now()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage_a1.json").write_text(json.dumps(H.sanitized(st), indent=1, ensure_ascii=False, default=str) + "\n")
    for name in ("upload_a1.json", "cards_a1.json", "token_check_a1.json"):
        shutil.copyfile(RECS / name, OUT / name)
    for rid, d in st["repos"].items():
        c = OUT / "cards" / rid.split("/", 1)[1]
        c.mkdir(parents=True, exist_ok=True)
        for f in ("README.md", "ppl_summary.json"):
            shutil.copyfile(Path(d["folder"]) / f, c / f)
    head = "## Addendum (2026-10-24): the 8x64 / 256x64 maps (part AB amendment 1)"
    path = OUT / "HF_UPLOAD.md"
    text = path.read_text()
    if head in text:
        text = text[:text.index(head)].rstrip("\n") + "\n"
    L = ["", head, "",
         "The user's request (relayed by the coordinator, 2026-10-24): part AA amendment 6's maps (FlipQuant 8x64 and "
         "256x64 at 1,024 windows) added to the eight public `<Base>-FlipQuant-Calib1k` repositories, in the layout of the "
         "256-window `<Base>-FlipQuant` repositories. The repositories were public before and after; every commit is "
         "public. No visibility was changed and no other repository was touched.",
         "- **Authorization:** this session's permission system refused the first commit (\"Create Public Surface\"); "
         "the part stopped and nothing was uploaded. The user then authorized it directly in this session (2026-10-24): "
         "only these eight repositories; per repository the two maps added and README.md and ppl_summary.json updated, "
         "exactly as staged; no other repository, file or visibility change. The uploads ran after that authorization.",
         "- **Sources:** each map's sha256 equals its amendment 6 job record. The meta is cleaned as for the 16x64 map; "
         "tiles and every tensor record are bit-identical. The 16x64 map, re-staged the same way, reproduces the uploaded "
         "file byte for byte.",
         "- **Cards:** README.md and ppl_summary.json regenerated for the three tiles, with numbers from part AA's record at "
         f"paper-eval {A1_COMMIT}: the three maps and their E0M3 shares (and the 256-window maps'), the settings, the "
         "results against each map's 256-window version, FourOverSix and the 16x64 map, and a note on 256x64.",
         f"- **Gates:** no hit in the maps' metadata, tile names or raw bytes; text files only the accepted kinds; the "
         f"exact-token check: {'0 hits' if tok['all_zero'] else 'HITS'} on all {len(tok['files'])} staged files.",
         "- **Upload:** one commit per repository (`" + A1_MESSAGE + "`), adding the two maps and updating README.md and "
         "ppl_summary.json.",
         "- **Verification, without a token:** the file list, every file's sha256 by download, the card metadata, the "
         "head and its commit; the Hub's YAML validation, README = staged and a local render.", "",
         f"Visibility at {checked}, without a token: "
         + ", ".join(sorted({{200: 'public', 401: 'private'}.get(c, f'HTTP {c}') for c in vis.values()})) + ".", "",
         "| repository | head before → after | files (downloaded, = staged) | card YAML (Hub) | README = staged | attempts |",
         "|---|---|---:|---|---|---|"]
    for rid, d in up["repos"].items():
        n_ok = sum(f["equal"] for f in d.get("files", {}).values())
        c = cj.get(rid, {})
        L.append(f"| `{rid}` | `{d['head_before'][:12]}` → `{d.get('head', '')[:12]}` | {n_ok} of {len(d.get('files', {}))} | "
                 f"{c.get('hub_yaml_valid')} | {c.get('readme_equal_to_staged')} | "
                 + "; ".join(f"{a['utc']} {a['result']}" for a in d["attempts"]) + " |")
    L += ["", "| repository | map | source = job record | E0M3 share | uploaded sha256 |", "|---|---|---|---:|---|"]
    for rid, d in st["repos"].items():
        for u in A1_NEW:
            m, s_ = d["maps"][u], d["sources"][u]
            L.append(f"| `{rid.split('/')[1]}` | `flipquant_{u}.pt` | {s_['equal']} | {100 * m['e0m3_tiles'] / m['tiles']:.2f} % | "
                     f"`{m['sha256_uploaded'][:16]}…` |")
    L += ["", "Files here: `stage_a1.json` (sanitized), `upload_a1.json`, `cards_a1.json`, `token_check_a1.json`; "
          "`cards/<repo>/` now holds the updated cards of the FlipQuant repositories.", ""]
    path.write_text(text + "\n".join(L) + "\n")
    print("\n".join(L[:20]))

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "stage":
        sel = [a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("--models=")]
        stage(sel[0] if sel else None)
    elif what == "upload":
        upload(sys.argv[2:] or None)
    elif what == "upload_gentle":
        upload_gentle(sys.argv[2], float(sys.argv[3]))
    elif what == "a1_stage":
        sel = [a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("--models=")]
        a1_stage(sel[0] if sel else None)
    elif what == "a1_upload":
        a1_upload(sys.argv[2:] or None)
    elif what in ("a1_tokens", "a1_check", "a1_cards", "a1_record"):
        {"a1_tokens": a1_tokens, "a1_check": a1_check, "a1_cards": a1_cards, "a1_record": a1_record}[what]()
    else:
        {"tokens": tokens, "check": check, "cards": cards, "status": status, "record": hf_record}.get(
            what, lambda: sys.exit(__doc__))()

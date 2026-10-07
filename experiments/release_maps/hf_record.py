"""The record of the Hugging Face upload, from hf_stage.py's stage.json and hf_upload.py's hf_upload.json:
- results/release_maps/HF_UPLOAD.md and results/release_maps/hf/{stage.json, hf_upload.json} in the release-maps tree;
- the "Hugging Face upload" section of the release directory's REPORT.md (appended once), mirrored to
  results/release_maps/REPORT.md as publish_records.sh does.

    python hf_record.py RELEASE_MAPS_WORKTREE
"""
import json
import shutil
import sys
from pathlib import Path

STAGE = Path("/home/dev/flipquant_hf")
REL = Path("/home/dev/flipquant_release")
LIC = Path("/home/dev/flipquant_hf_licenses")
MARK = "## Hugging Face upload (2026-10-07)"
WINDOWS_DRAFT = 146


def main(wt):
    wt = Path(wt)
    stage = json.loads((STAGE / "stage.json").read_text())
    up = json.loads((STAGE / "hf_upload.json").read_text())
    ppl = {repo: json.loads((STAGE / repo.split("/", 1)[1] / "ppl_summary.json").read_text())
           for repo in stage["repos"]}
    L = ["# Hugging Face upload of the release maps (2026-10-07)", "",
         "The 24 release maps (8 models × 8x64 / 16x64 / 256x64) are on the Hugging Face Hub, one **private** repository "
         "per model under `edgeai-lab`, uploaded with that account's active token on the user's decision. No repository is "
         "public, and no token or account setting was changed.", "",
         "| model | repository | HF commit | files |", "|---|---|---|---|"]
    for repo, d in up["repos"].items():
        L.append(f"| {stage['repos'][repo]['model']} | [{repo}]({d['url']}) | `{d['commit']}` | "
                 f"{', '.join(f'`{f}`' for f in d['hub_files'])} |")
    L += ["", "## What was uploaded", "",
          "- `flipquant_{8x64,16x64,256x64}.pt`: the release maps with only their `meta` changed (below); the tiles are "
          "bit-identical.",
          "- `README.md`: the final model card (from the release directory's draft; below).",
          "- `ppl_summary.json`: PPL per unit with BF16 and FourOverSix, the paired ΔNLL ± 2 SE against FourOverSix with "
          "verdicts, and the evaluation settings (native sm_120, the paper convention, datasets, revisions, windows); no "
          "local paths, host or user names.",
          "- License files: Qwen3-1.7B and Qwen3-8B `LICENSE`; Llama-3.1-8B `LICENSE`, `USE_POLICY.md`, `NOTICE`; "
          "Nemotron `LICENSE`, `NOTICE`. Mistral (both), Phi-4 and Qwen3.8-27B have no license file in their base "
          "repositories, so their cards' license field carries it.",
          "- **Not uploaded:** the release directory's `records/` (they hold local paths) and `ppl/` (the per-window "
          "reports).", "",
          "## The maps' meta changes (all 24; the user's decisions)", "",
          "| field | release file | uploaded file |", "|---|---|---|",
          "| `source.run` | the local run directory | relative, e.g. `runs/llama3.1-8b_16x64/run` |",
          "| `source.calibration_record` | the local record path | relative, e.g. `tmopt_data/llama8b/calibration/report.json` (its sha256 field kept) |",
          "| `fit_extension.flipquant_reference` | `brian030128/flipquant flipquant/data.py calibration_sets_extended @ b1c4123` | `flipquant flipquant/data.py calibration_sets_extended @ b1c4123` |",
          "| `branch`, `commit` | the reference implementation's branch and commit | removed (the FlipQuant commit `flipquant.commit` = 120173a and the neutral `repo` / `trainer` strings kept) |",
          "", "Each uploaded map was written with `flipquant.maps.save` and checked: torch.equal per module, every tensor "
          "record of the archive byte-identical to the release file's, and `flipquant.maps.load` reads it. sha256, release "
          "file → uploaded file (also in each card's provenance lines and in `hf/stage.json`):", "",
          "| model | file | release file | uploaded file |", "|---|---|---|---|"]
    for repo, d in stage["repos"].items():
        for u, st in d["maps"].items():
            L.append(f"| {d['model']} | `{st['file']}` | `{st['sha256_release']}` | `{st['sha256_uploaded']}` |")
    L += ["", "## License files", "",
          "- **Llama-3.1-8B:** `LICENSE` and `USE_POLICY.md` are meta-llama/Llama-3.1-8B's own files at the calibration "
          "revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` (sha256 "
          f"`{_sha(LIC / 'llama' / 'LICENSE')}` and `{_sha(LIC / 'llama' / 'USE_POLICY.md')}`). The local snapshot "
          "holds neither, so they were downloaded read-only with a stored read token chosen by name (the user's "
          "decision): passed to that download only, never printed, the active account unchanged.",
          "  - Against the text first extracted from the base card's gated prompt (after normalizing whitespace, quotes and "
          "headings): the official `LICENSE` has one more sentence, “By clicking \"I Accept\" below or by using or "
          "distributing any portion or element of the Llama Materials, you agree to be bound by this Agreement.”; the "
          "official `USE_POLICY.md` says “unlicensed uses of Llama 3.1” where the card says “unlicensed uses of Meta Llama "
          "3”. Nothing else differs. The official files are the ones uploaded.",
          "  - `NOTICE`: “Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. "
          "All Rights Reserved.” The card shows “Built with Llama” under its title, and the repository name starts with "
          "“Llama”.",
          "- **NVIDIA-Nemotron-Nano-9B-v2:** the base repository has no license file. `LICENSE` is the NVIDIA Open Model "
          "License Agreement (Last Modified: October 24, 2025) from the page the base card links, "
          "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/, fetched 2026-10-07 "
          f"05:26 UTC (page sha256 `{_sha(LIC / 'nemotron' / 'page.html')}`), rendered to text by "
          "`hf_licenses.py render`: verbatim paragraphs, the section numbers the page displays (CSS counters), bullets as "
          f"“•” (sha256 `{_sha(LIC / 'nemotron' / 'LICENSE')}`; re-rendering reproduces it byte for byte).",
          "  - Cross-check against NVIDIA's PDF of the same agreement (`nvidia-open-model-license-agreements-24-10-2025.pdf`, "
          f"sha256 `{_sha(LIC / 'nemotron' / 'agreement.pdf')}`; text decoded through its ToUnicode maps, some glyphs "
          "undecodable): the decoded text matches the page except formatting and punctuation. The PDF numbers "
          "subsections “1.1.” where the page shows “1.1”, ends definitions 1.4 and 1.5 with a period, and uses curly "
          "quotes where the page has straight ones in some definitions. The page, which the card links, is what was "
          "uploaded.",
          "  - `NOTICE`: “Licensed by NVIDIA Corporation under the NVIDIA Open Model License” (the agreement's section "
          "3.1). The card's license section records the source URL and retrieval time.",
          "- **Qwen3-1.7B, Qwen3-8B:** `LICENSE` is the base snapshot's own file (Apache 2.0).", "",
          "## Model cards", "",
          "From the release directory's drafts:",
          "- removed: the draft banner, the uploader placeholders, the flipquant link placeholder, and references to "
          "`records/`, `ppl/` and internal tool names;",
          "- added: Hub front matter (license; `license_name` / `license_link` for Nemotron; `base_model`; tags "
          "`quantization`, `fp4`, `nvfp4`, `mixfp4`, `flipquant`), “The FlipQuant code will be released with the "
          "paper.” (no link), “released under the same license as the base model”, the license files' sources, the "
          "uploaded files' sha256 and the provenance lines (release → uploaded sha256);",
          "- kept: the usage section, the settings (flagged as not the paper's 20-epoch setting), the PPL table against "
          "FourOverSix and BF16 with the paper-setting rows, the per-unit time and memory tables, and Qwen3.8-27B's note "
          "that the simulated path does not fit one 96 GB GPU;",
          f"- **corrected:** the WikiText-2 window count. Every draft said {WINDOWS_DRAFT}; the evaluation records give:", ""]
    L += ["| model | WikiText-2 windows (2048 tokens) | C4 windows |", "|---|---:|---:|"]
    for repo, p in ppl.items():
        c = p["evaluation"]["corpora"]
        L.append(f"| {stage['repos'][repo]['model']} | {c['wiki']['windows']} | {c['c4']['windows']} |")
    L += ["", "## Checks", "",
          "- **Before upload:** every staged file was scanned for local paths, the host name, user names, the private "
          "repositories and branch names, e-mail addresses and tokens (in the maps: every meta string, the module names "
          "and the raw bytes). The only hit is Meta's reporting address `LlamaUseReport@meta.com` inside the official "
          "`USE_POLICY.md`.",
          "- **Creation:** `create_repo(private=True, exist_ok=False)`; none of the names existed. Each repo reported "
          "`private` and answered an anonymous API request with HTTP 401 before any file went up.",
          "- **After upload** (`hf/hf_upload.json`):", ""]
    L += ["| repository | private | anonymous API | files as staged | card metadata parsed by the Hub | every file downloaded again, sha256 equal |",
          "|---|---|---:|---|---|---|"]
    for repo, d in up["repos"].items():
        cm = d["card_metadata"]
        meta = f"license `{cm.get('license')}`" + (f" ({cm.get('license_name')})" if cm.get("license_name") else "") + \
            f", base_model `{cm.get('base_model')}`"
        L.append(f"| {repo} | {d['private']} | {d['anonymous_http']} | "
                 f"{'yes' if set(d['hub_files']) - {'.gitattributes'} == set(d['files']) else 'NO'} | {meta} | "
                 f"{'yes' if all(f['download_sha256_equal'] for f in d['files'].values()) else 'NO'} |")
    L += ["", "## Files here", "",
          "- `hf/stage.json`: the staged files (sha256, bytes), the maps' meta changes (old → new) and sha256 (release → "
          "uploaded), the license sources and the scans.",
          "- `hf/hf_upload.json`: per repository the URL, commit, privacy checks, Hub file list, parsed card metadata and "
          "the download check.",
          "- Scripts (`experiments/release_maps/`): `hf_stage.py` (staging and scan), `hf_licenses.py` (the license "
          "texts), `hf_upload.py` (create, upload, verify), `hf_record.py` (this record).", ""]
    (wt / "results/release_maps/HF_UPLOAD.md").write_text("\n".join(L))
    (wt / "results/release_maps/hf").mkdir(exist_ok=True)
    for f in ("stage.json", "hf_upload.json"):
        shutil.copyfile(STAGE / f, wt / "results/release_maps/hf" / f)

    # the release REPORT's section (once), then the mirror
    rep = (REL / "REPORT.md").read_text()
    mirror = wt / "results/release_maps/REPORT.md"
    assert mirror.read_text() == rep or MARK in rep, "the mirrored REPORT differs from the release directory's"
    if MARK not in rep:
        S = ["", MARK, "",
             "The maps are on the Hub as private repositories under `edgeai-lab` (one per model; record and checks: "
             "NVFP4-RaZeR release-maps `results/release_maps/HF_UPLOAD.md`). The uploaded maps differ from this "
             "directory's files only in `meta`: local paths made relative, the data-extension reference without its "
             "repository owner, and the reference implementation's branch and commit removed; the tiles are "
             "bit-identical. Their cards are the finalized versions of the drafts here.", "",
             "| model | repository | HF commit |", "|---|---|---|"]
        for repo, d in up["repos"].items():
            S.append(f"| {stage['repos'][repo]['model']} | {d['url']} | `{d['commit']}` |")
        S += ["", "Map sha256, this directory's file → the uploaded file:", "", "| model | file | release | uploaded |",
              "|---|---|---|---|"]
        for repo, d in stage["repos"].items():
            for u, st in d["maps"].items():
                S.append(f"| {d['model']} | {st['file']} | `{st['sha256_release']}` | `{st['sha256_uploaded']}` |")
        S += ["", f"**Correction.** The draft cards in this directory give WikiText-2 as {WINDOWS_DRAFT} windows for every "
              "model. The evaluation records give " + ", ".join(
                  f"{stage['repos'][r]['model']} {p['evaluation']['corpora']['wiki']['windows']}" for r, p in ppl.items())
              + " (2048 tokens each; C4 is 256 for all). The uploaded cards state the correct counts; the PPL values were "
              "always computed on the actual windows.", ""]
        (REL / "REPORT.md").write_text(rep.rstrip("\n") + "\n" + "\n".join(S))
    shutil.copyfile(REL / "REPORT.md", mirror)


def _sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    main(sys.argv[1])

"""The calibration-size study (part AA) for the flipquant repository (branch calib-size, results/calib_size/): README.md
rendered from results/paper_eval/calib_size/calib_size.json at a paper-eval commit, and the report files that hold no
local paths, user or host names or tokens, copied unchanged from the same commit.

    python calib_size_flipquant_readme.py --commit <paper-eval commit> --out <flipquant worktree>/results/calib_size

Every number is rounded once, from calib_size.json. The copied files are scanned first with the HF uploads' patterns
(focus_hf.SCAN's path, user, host and token classes); a file with any match is not copied, and the README names it and
points to the paper-eval commit instead.
"""
import argparse
import json
import re
import socket
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = "results/paper_eval/calib_size"
CANDIDATES = ["REPORT.md", "WINDOWS.md", "STEP0.md", "PROBES.md", "calib_size.json"]
GENERATOR = "experiments/paper_eval/calib_size_flipquant_readme.py"
MODELS = [("llama3.2-1b-instruct", "Llama-3.2-1B-Instruct"), ("phi4-14b", "Phi-4"),
          ("mistral-7b", "Mistral-7B-Instruct-v0.3"), ("llama3.1-8b-instruct", "Llama-3.1-8B-Instruct"),
          ("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2"), ("qwen3.8-27b", "Qwen3.8-27B"), ("qwen3-1.7b", "Qwen3-1.7B"),
          ("qwen3-8b", "Qwen3-8B")]
SHORT = {"llama3.2-1b-instruct": "Llama-1B", "phi4-14b": "Phi-4", "mistral-7b": "Mistral", "llama3.1-8b-instruct": "Llama-8B",
         "nemotron-nano-9b-v2": "Nemotron", "qwen3.8-27b": "Qwen27B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B"}
# the HF uploads' scan (focus_hf.SCAN), its local-path, user / host-name and token classes
CLASSES = {"local path": r"/home|/tmp|/root|/mnt|/scratch|/vault|n16k64_campaign|fqrel|fqopt|~/",
           "user name": r"brian030128|chenjiaj|u4320956|nycu|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}",
           "host name": re.escape(socket.gethostname()) + r"|\bgputw\b",
           "token": r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"}


def show(commit, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, text=True,
                          check=True).stdout


def scan(text):
    return {k: len(re.findall(rx, text, re.I)) for k, rx in CLASSES.items() if re.findall(rx, text, re.I)}


def d(x, nd=2):
    """A paired difference with its significance mark and exact p."""
    if not x:
        return "—"
    return f"{x['diff']:+.{nd}f}{'*' if x['p'] < 0.05 else ''} ({x['row_only']}/{x['ref_only']}, p={x['p']:.2g})"


def dn(x):
    if not x:
        return "—"
    return f"{x['delta'] * 1e3:+.1f} ± {x['two_se'] * 1e3:.1f}{'*' if x['significant'] else ''}"


def v(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True, help="the paper-eval commit holding results/paper_eval/calib_size")
    ap.add_argument("--out", type=Path, required=True, help="results/calib_size in the flipquant worktree")
    args = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", args.commit], capture_output=True,
                            text=True, check=True).stdout.strip()
    rec = json.loads(show(commit, f"{SRC}/calib_size.json"))
    copied, left = [], {}
    for f in CANDIDATES:
        hits = scan(show(commit, f"{SRC}/{f}"))
        (left.__setitem__(f, hits) if hits else copied.append(f))
    M, P, P7 = rec["models"], rec["pooled"], rec["pooled_without_qwen27b"]

    def pv(label, src=P):
        x = src.get(label) or {}
        return [d(x.get("zeroshot")), d(x.get("gsm8k")), ", ".join(SHORT[m] for m in (x.get("gsm8k") or x.get(
            "zeroshot") or {}).get("models", []))]

    L = ["# Calibration size: FlipQuant 16x64 and FOCUS at 256, 1,024 and 2,048 windows", "",
         "**Source.**",
         f"- Every number here comes from NVFP4-RaZeR paper-eval {commit}: `{SRC}/calib_size.json`, rendered by "
         f"`{GENERATOR}`. The protocol is `experiments/paper_eval/PROTOCOL.md` there (part AA, amendments 1–4).",
         "- " + (", ".join(f"`{f}`" for f in copied) + " here are that commit's files, unchanged." if copied else
                 "No report file could be copied."),
         *([f"- Not copied, because they hold local paths or user names: "
            + "; ".join(f"`{f}` ({', '.join(f'{k} {n}' for k, n in h.items())})" for f, h in left.items())
            + f". They are in paper-eval {commit}, `{SRC}/`."] if left else []),
         "", "## The question and the settings", "",
         "- **Question:** do more calibration windows (1,024 or 2,048 of 512 tokens, instead of the release's 256) "
         "improve FlipQuant's downstream accuracy? FOCUS, trained on the same windows, is the comparison.",
         "- **Windows.** The release builder: the calibration record's 128 windows, then (N − 128) / 2 per source.",
         "  - The sources are OpenWebMath train-00000 @ fde8ef8 and CodeParrot-clean file-000000000001 @ 35a59fb, one "
         "window per document, seed 20260930.",
         "  - The record's fit documents, the three development records and the published C4 evaluation documents are "
         "skipped.",
         "  - The sets are nested, 256 ⊂ 1,024 ⊂ 2,048, record by record.",
         "  - FlipQuant and FOCUS read byte-identical windows (token sha256 checked).",
         "  - A word 13-gram audit of every 2,048 set against all evaluation texts finds 0 hits. Its positive control "
         "detects spliced evaluation items 30 / 30 per model.",
         "- **FlipQuant 16x64:** the release settings at 5 epochs, i.e. 160 / 640 / 1,280 optimizer steps.",
         "  - Adam lr 0.02, eps 1e-12; 8 sequences per step (Qwen3.8-27B and Nemotron 2 x 4); theta_u = −1; seed 0.",
         "  - The top-1000 + tail teacher; no development set; the last-step map.",
         "  - Qwen3.8-27B keeps the teacher in host memory (`--teacher-host`). It gives the GPU-teacher map bit for bit, "
         "as checked on Llama-3.2-1B-Instruct in step 0.",
         "- **FOCUS:** FOCUS_COMMON (1 epoch, global batch 32: 8 / 32 / 64 steps; lr 5e-3 / 1e-3; KL-Top 1,000; num_sub "
         "2; init_q 6; FourOverSix per-token activations; seed 42), micro-batch 8.",
         "  - Qwen3.8-27B is trained locally on one RTX PRO 6000 with host-offloaded AdamW, micro 1.",
         "  - Its 256 baseline was trained on 2 x H200 (part Q); a local 256 run is the environment-matched control.",
         "- **Evaluation**, exactly as the paper's zero-shot and GSM8K panels:",
         "  - zero-shot: six tasks (PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, LAMBADA), lm-eval 0.4.11;",
         "  - GSM8K: `gsm8k_llama`, 8-shot CoT, chat template, greedy, 1,024 tokens, strict-match primary;",
         "  - WikiText-2 / C4 PPL, paper convention.",
         "  - Native on the RTX PRO 6000 (build_V). IF4 and MixFP4 (Zou) are simulated, from the zero-shot / GSM8K panels.",
         "- **Code:** this branch: f912a30 (`--teacher-host`, `--probe-steps`, `calibration.fit_windows`), 898d00e "
         "(`train_focus --offload`), 9f19863 (`tmopt_launch --save-theta`).",
         "- **Statistics:**",
         "  - Paired differences in points (row-only / reference-only correct, exact McNemar p).",
         "  - Pooled rows sum the discordant counts over the models.",
         "  - ΔNLL is the mean per-window NLL difference x 1e-3 ± 2 SE; negative is better.",
         "  - \\* = p < 0.05 or |Δ| > 2 SE.", "",
         "## Pooled over the eight models", "",
         "| comparison | zero-shot | GSM8K | GSM8K models |", "|---|---|---|---|"]
    rows = [("FlipQuant 1,024 vs its 256", "FQ16-1k vs FQ16-256", P), ("FlipQuant 2,048 vs its 256", "FQ16-2k vs FQ16-256", P),
            ("FlipQuant 2,048 vs 1,024", "FQ16-2k vs FQ16-1k", P),
            ("FOCUS 1,024 vs its 256 (7 models; Qwen27B's 256 is another environment)", "FOCUS-1k vs FOCUS-256", P),
            ("FOCUS 2,048 vs its 256 (7 models)", "FOCUS-2k vs FOCUS-256", P),
            ("FlipQuant vs FOCUS, 256", "FQ16-256 vs FOCUS-256", P),
            ("FlipQuant vs FOCUS, 256, without Qwen27B", "FQ16-256 vs FOCUS-256", P7),
            ("FlipQuant vs FOCUS, 1,024 (incl. Qwen27B's early-stopping FOCUS-1k)", "FQ16-1k vs FOCUS-1k", P),
            ("FlipQuant vs FOCUS, 1,024, without Qwen27B", "FQ16-1k vs FOCUS-1k", P7),
            ("FlipQuant vs FOCUS, 2,048", "FQ16-2k vs FOCUS-2k", P),
            ("FlipQuant vs FOCUS, 2,048, without Qwen27B", "FQ16-2k vs FOCUS-2k", P7),
            ("FlipQuant 2,048 vs IF4† (GSM8K: 6 models)", "FQ16-2k vs IF4†", P),
            ("FlipQuant 2,048 vs MixFP4 (Zou)† (GSM8K: 6 models)", "FQ16-2k vs MixFP4 (Zou)†", P)]
    for label, key, src in rows:
        z, g, ms = pv(key, src)
        L.append(f"| {label} | {z} | {g} | {ms} |")
    L += ["", "IF4 and MixFP4 have no GSM8K rows for the two Qwen3 models (their simulated GSM8K runs were not part of "
          "the study).", "", "## Per model", ""]
    for m, title in MODELS:
        mm = M.get(m)
        if not mm:
            continue
        val, pr = mm["values"], mm["paired"]
        e0 = {c["windows"]: c.get("e0m3_share") for c in mm["cost"] if c["method"] == "FlipQuant 16x64"}
        L += [f"### {title}", "", "| method | windows | zero-shot Avg. | GSM8K strict | Wiki PPL | C4 PPL | E0M3 share |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        lines = [("FourOverSix", "—", "fo6", None)]
        lines += [("FOCUS", f"{n:,}", name, None) for n, name in ((256, "focus256"), (1024, "focus_1024"), (2048, "focus_2048"))]
        if m == "qwen3.8-27b":
            lines.insert(2, ("FOCUS, local control", "256", "focus256_local", None))
        lines += [("FlipQuant 16x64", f"{n:,}", name, n) for n, name in ((256, "fq16_256"), (1024, "fq16_1024"), (2048, "fq16_2048"))]
        for method, w, name, n in lines:
            x = val.get(name) or {}
            zs, g, pp = x.get("zeroshot"), x.get("gsm8k"), x.get("ppl")
            label = method + (" (cluster)" if m == "qwen3.8-27b" and name == "focus256" else "")
            L.append(f"| {label} | {w} | {v(zs and zs['avg'])} | {v(g and g[0])} | {v(pp and pp['wiki'], 3)} | "
                     f"{v(pp and pp['c4'], 3)} | {v(100 * e0[n], 1) + ' %' if n and e0.get(n) is not None else '—'} |")
        L += ["", "| comparison | zero-shot | GSM8K | ΔNLL Wiki | ΔNLL C4 |", "|---|---|---|---|---|"]
        comps = [("FlipQuant 1,024 vs 256", "FQ16-1k vs FQ16-256"), ("FlipQuant 2,048 vs 256", "FQ16-2k vs FQ16-256")]
        if m == "qwen3.8-27b":
            comps += [("FOCUS 256 local vs 256 cluster", "FOCUS-256 local vs FOCUS-256 cluster"),
                      ("FOCUS 1,024 vs 256 local (same environment)", "FOCUS-1k vs FOCUS-256 local (same environment)"),
                      ("FOCUS 2,048 vs 256 local (same environment)", "FOCUS-2k vs FOCUS-256 local (same environment)")]
        else:
            comps += [("FOCUS 1,024 vs 256", "FOCUS-1k vs FOCUS-256"), ("FOCUS 2,048 vs 256", "FOCUS-2k vs FOCUS-256")]
        comps += [("FlipQuant vs FOCUS, 256" + (" (cluster)" if m == "qwen3.8-27b" else ""), "FQ16-256 vs FOCUS-256")]
        if m == "qwen3.8-27b":
            comps += [("FlipQuant vs FOCUS, 256 local", "FQ16-256 vs FOCUS-256 local (FOCUS in this environment)")]
        comps += [("FlipQuant vs FOCUS, 1,024", "FQ16-1k vs FOCUS-1k"), ("FlipQuant vs FOCUS, 2,048", "FQ16-2k vs FOCUS-2k"),
                  ("FlipQuant 2,048 vs FourOverSix", "FQ16-2k vs FourOverSix")]
        for label, key in comps:
            c = pr.get(key) or {}
            nll = c.get("nll") or {}
            L.append(f"| {label} | {d(c.get('zeroshot'))} | {d(c.get('gsm8k'))} | {dn(nll.get('wiki'))} | "
                     f"{dn(nll.get('c4'))} |")
        L.append("")

    def z(m, key, ev="zeroshot"):
        return d((M[m]["paired"].get(key) or {}).get(ev))
    L += ["## The motivating losses at 256 / 1,024 / 2,048", "",
          "The paired losses that started the study, at each calibration size of FlipQuant 16x64.", "",
          "| loss | 256 | 1,024 | 2,048 |", "|---|---|---|---|",
          f"| Phi-4 zero-shot, FlipQuant vs FOCUS-256 | {z('phi4-14b', 'FQ16-256 vs FOCUS-256')} | "
          f"{z('phi4-14b', 'FQ16-1k vs FOCUS-256')} | {z('phi4-14b', 'FQ16-2k vs FOCUS-256')} |",
          f"| Phi-4 zero-shot, FlipQuant vs FOCUS at the same size | {z('phi4-14b', 'FQ16-256 vs FOCUS-256')} | "
          f"{z('phi4-14b', 'FQ16-1k vs FOCUS-1k')} | {z('phi4-14b', 'FQ16-2k vs FOCUS-2k')} |",
          f"| Nemotron zero-shot, FlipQuant vs FOCUS-256 | {z('nemotron-nano-9b-v2', 'FQ16-256 vs FOCUS-256')} | "
          f"{z('nemotron-nano-9b-v2', 'FQ16-1k vs FOCUS-256')} | {z('nemotron-nano-9b-v2', 'FQ16-2k vs FOCUS-256')} |",
          f"| Nemotron zero-shot, FlipQuant vs FOCUS at the same size | {z('nemotron-nano-9b-v2', 'FQ16-256 vs FOCUS-256')} | "
          f"{z('nemotron-nano-9b-v2', 'FQ16-1k vs FOCUS-1k')} | {z('nemotron-nano-9b-v2', 'FQ16-2k vs FOCUS-2k')} |",
          f"| Qwen3.8-27B zero-shot, FlipQuant vs IF4† | {z('qwen3.8-27b', 'FQ16-256 vs IF4†')} | "
          f"{z('qwen3.8-27b', 'FQ16-1k vs IF4†')} | {z('qwen3.8-27b', 'FQ16-2k vs IF4†')} |",
          f"| Phi-4 GSM8K, FlipQuant vs MixFP4 (Zou)† | {z('phi4-14b', 'FQ16-256 vs MixFP4 (Zou)†', 'gsm8k')} | "
          f"{z('phi4-14b', 'FQ16-1k vs MixFP4 (Zou)†', 'gsm8k')} | {z('phi4-14b', 'FQ16-2k vs MixFP4 (Zou)†', 'gsm8k')} |",
          "", "## Cost and E0M3 share", "",
          "Job wall time (s) of each calibration, its peak allocated GPU memory (GiB), and FlipQuant's E0M3 share; 256: the "
          "release / part P / part K runs (Qwen3.8-27B FOCUS-256: 2 x H200, training seconds).", "",
          "| model | FlipQuant 256 / 1,024 / 2,048 (s) | FlipQuant peak (GiB) | E0M3 share 256 / 1,024 / 2,048 | FOCUS training 256 / "
          "1,024 / 2,048 (s) | FOCUS peak (GiB) |", "|---|---|---|---|---|---|"]
    for m, title in MODELS:
        mm = M.get(m)
        if not mm:
            continue
        c = {(x["method"], x["windows"]): x for x in mm["cost"]}
        fq = [c.get(("FlipQuant 16x64", n), {}) for n in (256, 1024, 2048)]
        fo = [c.get(("FOCUS", n), {}) for n in (256, 1024, 2048)]
        L.append(f"| {SHORT[m]} | " + " / ".join(v(x.get("job_seconds"), 0) for x in fq) + " | "
                 + " / ".join(v(x.get("gpu_peak_allocated_gib"), 1) for x in fq) + " | "
                 + " / ".join(v(100 * x["e0m3_share"], 1) if x.get("e0m3_share") is not None else "—" for x in fq) + " % | "
                 + " / ".join(v(x.get("training_seconds"), 0) for x in fo) + " | "
                 + " / ".join(v(x.get("gpu_peak_allocated_gib"), 1) for x in fo)
                 + (" (256: GPU 0 of 2 x H200)" if m == "qwen3.8-27b" else "") + " |")
    gt = rec.get("gpu_time_seconds") or {}
    L += ["", "GPU time of the whole study: " + ", ".join(f"{k} {s / 3600:.1f} h" for k, s in gt.items())
          + f"; {sum(gt.values()) / 3600:.1f} h in all (one RTX PRO 6000).", ""]
    ll = M["llama3.2-1b-instruct"]
    th = ((ll.get("curves") or {}).get("2048") or {}).get("theta") or {}
    q27 = M["qwen3.8-27b"]
    st = q27.get("gsm8k_stops") or {}
    nem = M["nemotron-nano-9b-v2"]["paired"]["FQ16-2k vs FQ16-256"]
    q17 = M["qwen3-1.7b"]["paired"]
    L += ["## Caveats", "",
          "- **FOCUS is not reproducible run to run, and each FOCUS cell is one run.**",
          f"  - Llama-1B: a retrained FOCUS-256 against the original gives GSM8K "
          f"{d((ll['paired'].get('FOCUS-256 retrained vs FOCUS-256') or {}).get('gsm8k'))}.",
          f"  - Qwen3.8-27B: the two 256 runs (local and cluster) differ by GSM8K "
          f"{d((q27['paired'].get('FOCUS-256 local vs FOCUS-256 cluster') or {}).get('gsm8k'))}.",
          "  - FOCUS's 1,024-vs-256 differences of this size are within its own spread.",
          "- **Qwen3.8-27B's FOCUS-1k stops its answers early on GSM8K.**",
          "  - Strict no-answer counts: "
          + ", ".join(f"{(st.get(k) or {}).get('strict_no_answer', '—')} ({lab})" for k, lab in
                      (("focus256", "cluster 256"), ("focus256_local", "local 256"), ("focus_1024", "1,024"),
                       ("focus_2048", "2,048"))) + ".",
          "  - The local 256 control and the 2,048 run do not stop early, so this is specific to that run; it is not "
          "a trend in the window count.",
          f"- **Nemotron's FlipQuant-2k GSM8K strict-match loss is format only:** strict {d(nem.get('gsm8k'))}, "
          f"flexible-extract {d(nem.get('gsm8k_flex'))}.",
          f"- **Qwen3-1.7B trades general-text NLL for GSM8K** with more math / code windows: FlipQuant 2,048 vs 256 "
          f"GSM8K {d((q17.get('FQ16-2k vs FQ16-256') or {}).get('gsm8k'))}, ΔNLL Wiki "
          f"{dn(((q17.get('FQ16-2k vs FQ16-256') or {}).get('nll') or {}).get('wiki'))}. FOCUS shows it less (ΔNLL Wiki "
          f"{dn(((q17.get('FOCUS-2k vs FOCUS-256') or {}).get('nll') or {}).get('wiki'))}).",
          "- **E0M3 shares grow with the steps.** Llama-1B goes "
          + " / ".join(f"{100 * x['e0m3_share']:.1f}" for x in sorted((c for c in ll["cost"] if c["method"] ==
                                                                      "FlipQuant 16x64"), key=lambda c: c["windows"]))
          + " % at 256 / 1,024 / 2,048. Its 2,048 map stopped the study's 25 % gate; it was evaluated, and the gate was "
          "raised to 40 % for the rest of the study.",
          f"  - In that map, {100 * th.get('e0m3_within_one_step', float('nan')):.1f} % of the "
          f"{th.get('e0m3_tiles', 0):,} E0M3 tiles have theta < 0.02 (within one lr step of the threshold), and "
          f"{100 * th.get('all_within_one_step', float('nan')):.1f} % of all {th.get('tiles', 0):,} tiles have "
          f"|theta| < 0.02.",
          "  - These are measured by a `--save-theta` rerun that reproduces the map bit for bit. On every evaluation the "
          "2,048 map is statistically indistinguishable from the 1,024 map.",
          "- **The Qwen3.8-27B FOCUS deviation:** its 1,024 / 2,048 runs and the 256 control ran locally with offloaded "
          "AdamW on one GPU, not on 2 x H200 like its published 256 state; only the summation order differs.",
          "- **Data draw:** FlipQuant is deterministic, but a different window draw was not tried.", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "README.md").write_text("\n".join(L) + "\n")
    for f in copied:
        (args.out / f).write_text(show(commit, f"{SRC}/{f}"))
    print(json.dumps(dict(commit=commit, copied=copied, left_out=left), indent=1))


if __name__ == "__main__":
    main()

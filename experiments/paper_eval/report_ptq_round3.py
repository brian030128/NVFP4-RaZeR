"""Part O reports (PROTOCOL.md part O): round 3 of tab:ptq. Run with n16k64's interpreter (torch, CPU only).

    python report_ptq_round3.py shares      # O1: E0M3 shares, Hadamard-basis vs release 16x64 maps, per projection type
    python report_ptq_round3.py pilot       # O0: the BF16 pilot (batch rule)
    python report_ptq_round3.py report      # O0-O4: results/paper_eval/ptq_round3/{REPORT.md, round3.json, samples/}

Shares count 16x64 tiles. "both" = tiles E0M3 in both maps; "by chance" = its expectation if the maps chose independently
within each module; Jaccard = both / (E0M3 in either).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval" / "ptq_round3"
RUN = G.RUN
ROOT = RUN / "ptq_round3"
MODELS = [("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2"), ("qwen3.8-27b", "Qwen3.8-27B")]
# projection types by the module name's last component, grouped by block kind
FAMILY = {"nemotron-nano-9b-v2": {"Mamba": ("in_proj", "out_proj"), "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
                                  "MLP": ("up_proj", "down_proj")},
          "qwen3.8-27b": {"Gated DeltaNet": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"),
                          "attention": ("q_proj", "k_proj", "v_proj", "o_proj"), "MLP": ("gate_proj", "up_proj", "down_proj")}}


def load_map(path):
    sys.path.insert(0, "/home/dev/n16k64_campaign/sm120runs/wt")
    from flipquant import maps as M
    tiles, unit, meta = M.load(path)
    return tiles, unit, meta


def compare(a, b, model):
    """Per projection type and in all: tiles, E0M3 in a / b, both, Jaccard. a, b: {module: bool tensor}."""
    assert set(a) == set(b) and all(a[n].shape == b[n].shape for n in a), "the maps cover different modules / shapes"
    leaf = {}
    for fam, leaves in FAMILY[model].items():
        for x in leaves:
            leaf[x] = fam
    groups = {}
    for n in a:
        x = n.split(".")[-1]
        assert x in leaf, f"{model}: unknown projection type {n}"
        for g in (f"{leaf[x]} / {x}", leaf[x], "all"):
            s = groups.setdefault(g, dict(modules=0, tiles=0, e0m3_a=0, e0m3_b=0, both=0, chance=0.0))
            s["modules"] += 1
            s["tiles"] += a[n].numel()
            s["e0m3_a"] += int(a[n].sum())
            s["e0m3_b"] += int(b[n].sum())
            s["both"] += int((a[n] & b[n]).sum())
            s["chance"] += int(a[n].sum()) * int(b[n].sum()) / a[n].numel()     # E[both] if independent within a module
    for s in groups.values():
        either = s["e0m3_a"] + s["e0m3_b"] - s["both"]
        s.update(share_a=s["e0m3_a"] / s["tiles"], share_b=s["e0m3_b"] / s["tiles"],
                 jaccard=s["both"] / either if either else None)
    return groups


def order(model, groups):
    keys = []
    for fam, leaves in FAMILY[model].items():
        keys += [f"{fam} / {x}" for x in leaves if f"{fam} / {x}" in groups] + [fam]
    return keys + ["all"]


def shares():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(note="16x64 tiles; a = the Hadamard-basis map (part H, trained in n16k64-fast), b = the release map "
                    "(trained in n16k64)", models={})
    md = ["# O1: E0M3 share of the Hadamard-basis FlipQuant 16x64 maps vs the unrotated release maps", "",
          "Hadamard: part H's map (`RUN/ptq/<model>/fq-16x64_hadamard.pt`; TM-OPT+TC with the release settings in the "
          "block-16 Hadamard basis, n16k64-fast). Release: `flipquant_release/<model>/flipquant_16x64.pt` (n16k64). "
          "A block-16 Hadamard acts within each 16-column scale group, so a 16x64 tile covers the same weights in both "
          "bases and the tiles can be compared position by position. **Env caveat:** part H trained the Hadamard maps "
          "in n16k64-fast and the release maps were trained in n16k64, so the rotated vs unrotated shares mix the "
          "basis change with the env. Both = E0M3 in both maps; by chance = the expected count if the two maps chose "
          "their E0M3 tiles independently within each module (sum of a_l b_l / tiles_l); Jaccard = both / E0M3 in "
          "either.", ""]
    for model, title in MODELS:
        a, ua, ma = load_map(RUN / "ptq" / model / "fq-16x64_hadamard.pt")
        b, ub, mb = load_map(G.REL / model / "flipquant_16x64.pt")
        assert ua == ub == "16x64" and ma.get("rotate") == "hadamard" and ma.get("rotate_block") == 16, (ua, ub, ma.get("rotate"))
        g = compare(a, b, model)
        rec["models"][model] = dict(hadamard_map=str(RUN / "ptq" / model / "fq-16x64_hadamard.pt"),
                                    release_map=str(G.REL / model / "flipquant_16x64.pt"), groups=g)
        md += [f"## {title}", "", "| projection | modules | tiles | E0M3 Hadamard | E0M3 release | Hadamard − release (pp) | "
               "both | both by chance | Jaccard |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k in order(model, g):
            s = g[k]
            bold = "**" if " / " not in k else ""
            md.append(f"| {bold}{k}{bold} | {s['modules']} | {s['tiles']:,} | {s['e0m3_a']:,} ({100 * s['share_a']:.3f} %) | "
                      f"{s['e0m3_b']:,} ({100 * s['share_b']:.3f} %) | {100 * (s['share_a'] - s['share_b']):+.3f} | "
                      f"{s['both']:,} | {s['chance']:,.0f} | {s['jaccard']:.3f} |" if s["jaccard"] is not None else
                      f"| {bold}{k}{bold} | {s['modules']} | {s['tiles']:,} | 0 | 0 | +0.000 | 0 | 0 | - |")
        md.append("")
    (OUT / "o1_shares.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "O1_SHARES.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ----------------------------------------------------------------------------------------------------- O0 pilot
KV_KIB = {"nemotron-nano-9b-v2": 16, "qwen3.8-27b": 64}   # KV cache per token (PROTOCOL.md part O, O0)
LIMIT_MIB = 97280                                        # 95 GiB
LADDER = (64, 48, 32, 16)


def pilot():
    """O0's batch rule: batch b fits if its pilot completes and nvidia-smi's peak + b x 1024 x KV per token <= 95 GiB;
    the full run uses the largest batch that fits."""
    out = OUT / "o0_pilot"
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(rule="pilot rc 0 and nvidia-smi peak + b x 1024 tokens x KV/token <= 97280 MiB", models={})
    md = ["# O0 pilot: the BF16 GSM8K batch (not a result)", "",
          "Part N's pilot documents per model (the 64 longest questions + 64 others), BF16, part N's command. A batch b "
          "fits if the pilot completes and nvidia-smi's peak plus the KV growth bound b x 1024 tokens x KV per token "
          "(64 KiB Qwen3.8-27B, 16 KiB Nemotron-Nano-9B-v2) is at most 97,280 MiB (95 GiB).", "",
          "| model | batch | rc | nvidia-smi peak (MiB) | + KV bound (MiB) | total (MiB) | torch peak allocated (GiB) | "
          "seconds | strict / flexible (pilot docs) | fits |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for model, title in MODELS:
        d = ROOT / "bf16" / "pilot" / model
        fits = []
        for b in LADDER:
            mem_f, rep_f = d / f"pilot_b{b}.mem.json", d / f"pilot_b{b}.json"
            if not mem_f.exists():
                continue
            mem = json.loads(mem_f.read_text())
            rep = json.loads(rep_f.read_text()) if rep_f.exists() else {}
            t = (rep.get("tasks") or {}).get("gsm8k_llama", {})
            kv = b * 1024 * KV_KIB[model] // 1024
            ok = mem["rc"] == 0 and rep.get("status") == "complete" and mem["peak_mib"] + kv <= LIMIT_MIB
            if ok:
                fits.append(b)
            m = t.get("metrics", {})
            r = rec["models"].setdefault(model, {})[b] = dict(
                rc=mem["rc"], nvidia_smi_peak_mib=mem["peak_mib"], kv_bound_mib=kv, total_mib=mem["peak_mib"] + kv,
                fits=ok, torch_peak_allocated_gib=t.get("peak_gpu_allocated_gib"), seconds=t.get("seconds"),
                strict=m.get("exact_match,strict_match"), flexible=m.get("exact_match,flexible_extract"))
            md.append(f"| {title} | {b} | {r['rc']} | {r['nvidia_smi_peak_mib']:,} | {kv:,} | {r['total_mib']:,} | "
                      f"{r['torch_peak_allocated_gib'] or 0:.2f} | {r['seconds'] or 0:.0f} | "
                      f"{r['strict']} / {r['flexible']} | {'yes' if ok else 'no'} |")
        rec["models"].setdefault(model, {})["chosen"] = max(fits) if fits else None
    md += ["", "Chosen (the largest batch that fits): " + ", ".join(
        f"{t} {rec['models'][m]['chosen']}" for m, t in MODELS), ""]
    (out / "pilot.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "PILOT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------------------------------------------- the report
FMTS = ("nvfp4", "fo6", "fq-16x64")
GSM_ROWS = ("gptq_nvfp4", "gptq_fo6", "gptq_fq-16x64", "gptqcand_fq-16x64")
ROW_NAMES = {"gptq_nvfp4": "GPTQ, act. quant. on: NVFP4", "gptq_fo6": "GPTQ, act. quant. on: FourOverSix",
             "gptq_fq-16x64": "GPTQ, act. quant. on: FlipQuant 16x64 (release map fixed)",
             "gptqcand_fq-16x64": "GPTQ candidates + retrained map: FlipQuant 16x64"}


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def ppl_rec(p):
    r = jload(p)
    return r if r and "resources" in r else None


def ppl_of(r, c):
    import math
    v = r["results"][c]["nll"]
    return math.exp(sum(v) / len(v))


def file_sha256(path, cache):
    """sha256 of a (large) file, streamed; cached by size and mtime."""
    import hashlib
    st = Path(path).stat()
    hit = cache.get(str(path))
    if hit and hit["size"] == st.st_size and hit["mtime"] == st.st_mtime:
        return hit["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    cache[str(path)] = dict(size=st.st_size, mtime=st.st_mtime, sha256=h.hexdigest())
    return cache[str(path)]["sha256"]


def fmt_delta(d):
    return f"{1e3 * d['delta']:+.2f} ± {1e3 * d['two_se']:.2f}{'*' if d['significant'] else ''}"


def fmt_pp(p):
    return (f"{100 * p['delta']:+.2f} ± {100 * p['two_se']:.2f}{' *' if p['significant'] else ''} "
            f"({p['a_only']} / {p['b_only']}, p {p['mcnemar_p']:.3g})")


def report():
    import shutil
    import report_gsm8k as RM
    import report_gsm8k_lmeval as RN
    import run_gsm8k as M
    import run_ptq_round3 as P
    import tables_cpu as T
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    cache_f = ROOT / "codes_sha256.json"
    cache = jload(cache_f) or {}
    rec = dict(o0={}, o2={}, o3={}, o4={}, checks={}, codes_root=str(P.CODES_ROOT), files={})
    md = ["# Round 3 of tab:ptq (part O)", "",
          "PROTOCOL.md part O (registered 2026-10-10 07:34 UTC; amendment 1: GPTQ code files on the user's vault, the "
          "trainer in n16k64). Native sm120 (build_V, `--kernel-set auto`), n16k64-fast, release revisions; flipquant "
          "paper-sm120-runs 276ce86. ΔNLL: per-window paired difference x 1e-3 ± 2 SE (* beyond 2 SE). GSM8K: part N's "
          "protocol (lm-eval 0.4.11 gsm8k_llama, chat template, greedy, 1024 tokens, thinking off, 1,319 problems); "
          "accuracy ± 2 SE (binomial); paired per problem: A − B in percentage points ± 2 SE (A-only / B-only, McNemar "
          "exact p).", ""]

    def gsm_ex(path):
        rep = RN.load(path)
        return (rep, rep["tasks"][RN.TASK]["examples"]) if rep else (None, None)

    def pair(a, b):
        return {f: RM.paired({d: dict(correct=v) for d, v in RN.per_filter(a, f).items()},
                             {d: dict(correct=v) for d, v in RN.per_filter(b, f).items()}) for f in RN.FILTERS}

    def copy_samples(path, name):
        src = Path(path).with_suffix("").with_name(Path(path).stem + f".{RN.TASK}.jsonl.gz")
        if src.exists():
            shutil.copyfile(src, OUT / "samples" / name)

    # ---------------------------------------------------------------- O0
    md += ["## O0: the BF16 reference row", "",
           "| model | batch | strict-match (%) | flexible-extract (%) | cannot extract (strict / flexible) | used the 1024 "
           "budget | mean / max tokens | WikiText-2 | C4 |", "|---|---:|---:|---:|---|---:|---|---:|---:|"]
    bf16 = {}
    for model, title in MODELS:
        path = ROOT / "bf16" / model / "bf16.json"
        rep, ex = gsm_ex(path)
        pr = ppl_rec(RUN / "pplfast" / model / "bf16.json")
        ppl = {c: ppl_of(pr, c) for c, _ in T.CORPORA} if pr else {}
        if rep is None:
            md.append(f"| {title} | | TBD | TBD | | | | {ppl.get('wiki', 0):.4f} | {ppl.get('c4', 0):.4f} |")
            continue
        bf16[model] = ex
        s = RN.summary(rep)
        g = s["generated"]
        rec["o0"][model] = dict(gsm8k=s, ppl=ppl, ppl_source=str(RUN / "pplfast" / model / "bf16.json"), paired={})
        rec["checks"].setdefault("N1", {})[f"{model}/bf16"] = s["think_markers"] == 0
        rec["checks"].setdefault("N4", {})[f"{model}/bf16"] = len(ex) == RN.N_DOCS and all(
            all(f in e for f in RN.FILTERS) for e in ex.values())
        copy_samples(path, f"{model}__bf16.{RN.TASK}.jsonl.gz")
        md.append(f"| {title} | {s['batch']} | {100 * s['strict_match']['accuracy']:.2f} ± "
                  f"{100 * s['strict_match']['two_se']:.2f} | {100 * s['flexible_extract']['accuracy']:.2f} ± "
                  f"{100 * s['flexible_extract']['two_se']:.2f} | {s['strict_match']['cannot_extract']} / "
                  f"{s['flexible_extract']['cannot_extract']} | {g['used_budget']} | {g['mean_tokens']:.0f} / "
                  f"{g['max_tokens']} | {ppl['wiki']:.4f} | {ppl['c4']:.4f} |")
    md += ["", "BF16 PPL: part F's records (`RUN/pplfast/<model>/bf16.json`; n16k64-fast, release revisions). Qwen3.8-27B's "
           "BF16 GSM8K ran at batch 48 (batch 64 failed the O0 memory rule); every quantized row ran at 64.", "",
           "Part N's 18 configurations vs BF16 (A = the configuration, B = BF16):", "",
           "| model | configuration | strict-match Δ (pp) | flexible-extract Δ (pp) |", "|---|---|---:|---:|"]
    for model, title in MODELS:
        if model not in bf16:
            continue
        for method in M.METHODS:
            for fmt in M.FMTS:
                _, ex = gsm_ex(RN.NL.ROOT / model / f"{method}_{fmt}.json")
                if ex is None:
                    continue
                p = pair(ex, bf16[model])
                rec["o0"][model]["paired"][f"{method}_{fmt}"] = p
                md.append(f"| {title} | {RM.METHOD_NAMES[method]} {RM.FMT_NAMES[fmt]} | {fmt_pp(p['strict_match'])} | "
                          f"{fmt_pp(p['flexible_extract'])} |")
    md += ["", "## O1: E0M3 shares, Hadamard-basis vs release maps", "", "See `O1_SHARES.md` (all rows, per projection "
           "type). In all: Nemotron-Nano-9B-v2 1.066 % vs 1.064 %, Qwen3.8-27B 0.427 % vs 0.443 %; Jaccard 0.010 / "
           "0.005 (1.6x / 1.8x chance). Env caveat: part H trained the Hadamard maps in n16k64-fast, the release maps "
           "were trained in n16k64.", ""]

    # ---------------------------------------------------------------- O2
    md += ["## O2: GPTQ with activation quantization on during calibration", "",
           "| model | format | WikiText-2 | C4 | ΔNLL wiki vs RTN FO6 | ΔNLL C4 vs RTN FO6 | ΔNLL wiki vs part H GPTQ | "
           "ΔNLL C4 vs part H GPTQ | propagate / Hessian input | codes_sha256 |", "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for model, title in MODELS:
        ref = ppl_rec(RUN / "pplfast" / model / "fo6.json")
        for fmt in FMTS:
            r = ppl_rec(ROOT / model / f"gptq_{fmt}.json")
            if r is None:
                md.append(f"| {title} | {RM.FMT_NAMES[fmt]} | TBD | TBD | | | | | | |")
                continue
            h = ppl_rec(RUN / "ptq" / model / f"gptq_{fmt}.json")
            pt = r.get("ptq") or {}
            cells = {c: ppl_of(r, c) for c, _ in T.CORPORA}
            vs_rtn = {c: T.paired(r["results"][c]["nll"], ref["results"][c]["nll"]) for c, _ in T.CORPORA}
            vs_h = {c: T.paired(r["results"][c]["nll"], h["results"][c]["nll"]) for c, _ in T.CORPORA} if h else {}
            path = P.codes(P.CODES_ROOT, model, fmt)
            sha = file_sha256(path, cache) if path.exists() else None
            rec["o2"].setdefault(model, {})[fmt] = dict(ppl=cells, vs_rtn_fo6=vs_rtn, vs_part_h_gptq=vs_h,
                                                        codes_sha256=pt.get("codes_sha256"), file=str(path),
                                                        file_sha256=sha, key={k: pt.get(k) for k in (
                                                            "weight", "act", "propagate", "hessian_input", "calib",
                                                            "windows", "damp", "block", "map_sha256")},
                                                        map=(r.get("map") or {}).get("path"))
            rec["files"][str(path)] = sha
            rec["checks"].setdefault("O-1 ppl", {})[f"{model}/gptq_{fmt}"] = (
                pt.get("hessian_input") == "quantized" and pt.get("propagate") is None and pt.get("calib") == "release")
            if fmt == "fq-16x64":
                rec["checks"].setdefault("O-2", {})[f"{model}/gptq_{fmt}"] = (r.get("map") or {}).get("path") == str(
                    P.release_map(model))
            md.append(f"| {title} | {RM.FMT_NAMES[fmt]} | {cells['wiki']:.4f} | {cells['c4']:.4f} | "
                      f"{fmt_delta(vs_rtn['wiki'])} | {fmt_delta(vs_rtn['c4'])} | "
                      + (f"{fmt_delta(vs_h['wiki'])} | {fmt_delta(vs_h['c4'])}" if vs_h else " | ")
                      + f" | {pt.get('propagate') or 'quantized'} / {pt.get('hessian_input')} | "
                        f"`{str(pt.get('codes_sha256'))[:16]}` |")

    # ---------------------------------------------------------------- O3
    md += ["", "## O3: GPTQ candidates and a retrained map", ""]
    for model, title in MODELS:
        o3 = rec["o3"].setdefault(model, {})
        e0 = jload(ROOT / model / "gptq_e0m3.json")
        if e0 and e0.get("status") == "complete":
            path = Path(e0["cache"])
            o3["e0m3_candidate"] = dict(file=str(path), file_sha256=file_sha256(path, cache), codes_sha256=(
                e0.get("gptq") or {}).get("codes_sha256"), report_cache_sha256=e0.get("cache_sha256"),
                mean_codes_changed=(e0.get("gptq") or {}).get("mean_codes_changed"))
            rec["files"][str(path)] = o3["e0m3_candidate"]["file_sha256"]
        release_tiles, _, _ = load_map(G.REL / model / "flipquant_16x64.pt")
        run_rel = jload(G.REL / model / "records" / "16x64" / "run.json") or {}
        for src in ("rtn", "gptq"):
            mp = P.trained_map(model, src)
            mrep = jload(mp.with_suffix(".json"))
            if not (mrep and mrep.get("status") == "complete"):
                continue
            tiles, _, meta = load_map(mp)
            ent = o3.setdefault(f"map_{src}", dict(map=str(mp), map_file_sha256=mrep.get("map_file_sha256"),
                                                    trainer_map_sha256=meta.get("source_map_sha256"),
                                                    summary=mrep.get("map_summary"), candidates=meta.get("candidates")))
            if src == "rtn":
                same = set(tiles) == set(release_tiles) and all(bool((tiles[n] == release_tiles[n]).all())
                                                                 for n in tiles)
                ent.update(release_run_map_sha256=run_rel.get("run_map_sha256"), tiles_equal_release=same,
                           trainer_sha_equal_release=meta.get("source_map_sha256") == run_rel.get("run_map_sha256"))
                rec["checks"].setdefault("O-3 (c)", {})[model] = same and ent["trainer_sha_equal_release"]
            else:
                ent["vs_release"] = compare(tiles, release_tiles, model)
        r = ppl_rec(ROOT / model / "gptqcand_fq-16x64.json")
        if r is not None:
            pt = r.get("ptq") or {}
            cells = {c: ppl_of(r, c) for c, _ in T.CORPORA}
            refs = {"GPTQ FO6 (O2)": ppl_rec(ROOT / model / "gptq_fo6.json"),
                    "GPTQ FlipQuant, fixed map (O2)": ppl_rec(ROOT / model / "gptq_fq-16x64.json"),
                    "RTN FlipQuant (part F)": ppl_rec(RUN / "pplfast" / model / "fq-16x64.json")}
            vs = {k: {c: T.paired(r["results"][c]["nll"], v["results"][c]["nll"]) for c, _ in T.CORPORA}
                  for k, v in refs.items() if v}
            comp = pt.get("composed") or {}
            o3["ppl"] = dict(ppl=cells, paired=vs, codes_sha256=pt.get("codes_sha256"), composed={
                k: dict(path=v.get("path"), codes_sha256=v.get("codes_sha256")) for k, v in comp.items()})
            fo6 = (rec["o2"].get(model) or {}).get("fo6") or {}
            rec["checks"].setdefault("O-5", {})[model] = (
                (comp.get("e2m1") or {}).get("codes_sha256") == fo6.get("codes_sha256") is not None
                and (comp.get("e0m3") or {}).get("codes_sha256") == (o3.get("e0m3_candidate") or {}).get("codes_sha256"))
        md += [f"### {title}", ""]
        if "e0m3_candidate" in o3:
            c = o3["e0m3_candidate"]
            md.append(f"- E0M3 candidate: GPTQ on the all-E0M3 grid, `{c['file']}` (sha256 `{c['file_sha256'][:16]}`), "
                      f"codes changed vs RTN {100 * (c['mean_codes_changed'] or 0):.1f} % (mean per linear).")
        if "map_rtn" in o3:
            e = o3["map_rtn"]
            md.append(f"- Test (c): the hook with RTN candidates (n16k64) reproduces the release map: tiles "
                      f"{'equal' if e['tiles_equal_release'] else 'DIFFER'}; trainer map.pt sha256 `{str(e['trainer_map_sha256'])[:16]}` vs "
                      f"release `{str(e['release_run_map_sha256'])[:16]}`.")
        if "map_gptq" in o3:
            e, g = o3["map_gptq"], o3["map_gptq"]["vs_release"]["all"]
            md.append(f"- Retrained map (GPTQ candidates, n16k64): {g['e0m3_a']:,} E0M3 tiles ({100 * g['share_a']:.3f} %) vs "
                      f"the release map's {g['e0m3_b']:,} ({100 * g['share_b']:.3f} %); both {g['both']:,} (chance "
                      f"{g['chance']:,.0f}), Jaccard {g['jaccard']:.3f}.")
        if "ppl" in o3:
            p = o3["ppl"]
            md.append(f"- Native PPL: WikiText-2 {p['ppl']['wiki']:.4f}, C4 {p['ppl']['c4']:.4f}. ΔNLL x 1e-3 vs " + "; ".join(
                f"{k}: wiki {fmt_delta(v['wiki'])}, C4 {fmt_delta(v['c4'])}" for k, v in p["paired"].items()) + ".")
        if "map_gptq" in o3:
            md += ["", "| projection | E0M3 retrained | E0M3 release | both | by chance | Jaccard |",
                   "|---|---:|---:|---:|---:|---:|"]
            gg = o3["map_gptq"]["vs_release"]
            for k in order(model, gg):
                s = gg[k]
                md.append(f"| {k} | {s['e0m3_a']:,} ({100 * s['share_a']:.3f} %) | {s['e0m3_b']:,} ({100 * s['share_b']:.3f} %) "
                          f"| {s['both']:,} | {s['chance']:,.0f} | {s['jaccard'] if s['jaccard'] is None else round(s['jaccard'], 3)} |")
        md.append("")

    # ---------------------------------------------------------------- O4
    md += ["## O4: GSM8K for the new rows (batch 64)", "",
           "| model | row | strict-match (%) | flexible-extract (%) | cannot extract | used the budget | mean / max tokens |",
           "|---|---|---:|---:|---|---:|---|"]
    o4ex = {}
    for model, title in MODELS:
        for row in GSM_ROWS:
            path = ROOT / "gsm8k" / model / f"{row}.json"
            rep, ex = gsm_ex(path)
            if rep is None:
                md.append(f"| {title} | {ROW_NAMES[row]} | TBD | TBD | | | |")
                continue
            o4ex[(model, row)] = ex
            s = RN.summary(rep)
            g = s["generated"]
            key = f"{model}/{row}"
            rec["o4"].setdefault(model, {})[row] = dict(gsm8k=s)
            rec["checks"].setdefault("N1", {})[key] = s["think_markers"] == 0
            rec["checks"].setdefault("N4", {})[key] = len(ex) == RN.N_DOCS and all(
                all(f in e for f in RN.FILTERS) for e in ex.values())
            cov = rep.get("coverage") or {}
            rec["checks"].setdefault("N5", {})[key] = (cov.get("native_called") == cov.get("native")
                                                       and not cov.get("remaining_bf16_linears_called"))
            pol = rep.get("policy") or {}
            pr = ppl_rec(ROOT / model / f"{row}.json")
            a, b = (pol.get("ptq") or {}).get("codes_sha256"), ((pr or {}).get("ptq") or {}).get("codes_sha256")
            rec["checks"].setdefault("O-1 gsm8k", {})[key] = a is not None and a == b
            copy_samples(path, f"{model}__{row}.{RN.TASK}.jsonl.gz")
            md.append(f"| {title} | {ROW_NAMES[row]} | {100 * s['strict_match']['accuracy']:.2f} ± "
                      f"{100 * s['strict_match']['two_se']:.2f} | {100 * s['flexible_extract']['accuracy']:.2f} ± "
                      f"{100 * s['flexible_extract']['two_se']:.2f} | {s['strict_match']['cannot_extract']} / "
                      f"{s['flexible_extract']['cannot_extract']} | {g['used_budget']} | {g['mean_tokens']:.0f} / "
                      f"{g['max_tokens']} |")
    md += ["", "| model | comparison (A vs B) | strict-match Δ (pp) | flexible-extract Δ (pp) |", "|---|---|---:|---:|"]
    for model, title in MODELS:
        _, rtn_fq = gsm_ex(RN.NL.ROOT / model / "rtn_fq-16x64.json")
        _, gptq_h_fq = gsm_ex(RN.NL.ROOT / model / "gptq_fq-16x64.json")
        pairs = [("gptq_nvfp4", "gptq_fo6"), ("gptq_fq-16x64", "gptq_fo6"), ("gptqcand_fq-16x64", "gptq_fo6"),
                 ("gptqcand_fq-16x64", "gptq_fq-16x64")]
        cmp = {f"{ROW_NAMES[a]} vs {ROW_NAMES[b]}": (o4ex.get((model, a)), o4ex.get((model, b))) for a, b in pairs}
        for a in ("gptq_fq-16x64", "gptqcand_fq-16x64"):
            cmp[f"{ROW_NAMES[a]} vs RTN FlipQuant (part N)"] = (o4ex.get((model, a)), rtn_fq)
        cmp[f"{ROW_NAMES['gptq_fq-16x64']} vs part N's GPTQ FlipQuant (BF16 propagation)"] = (
            o4ex.get((model, "gptq_fq-16x64")), gptq_h_fq)
        for row in GSM_ROWS:
            cmp[f"{ROW_NAMES[row]} vs BF16"] = (o4ex.get((model, row)), bf16.get(model))
        for k, (a, b) in cmp.items():
            if a and b:
                p = pair(a, b)
                rec["o4"].setdefault(model, {}).setdefault("paired", {})[k] = p
                md.append(f"| {title} | {k} | {fmt_pp(p['strict_match'])} | {fmt_pp(p['flexible_extract'])} |")

    # ---------------------------------------------------------------- checks, files, table
    rec["checks_summary"] = {k: dict(passed=sum(bool(x) for x in v.values()), of=len(v)) for k, v in rec["checks"].items()}
    md += ["", "## Checks", "", "| check | passed |", "|---|---:|"]
    md += [f"| {k} | {v['passed']} / {v['of']} |" for k, v in rec["checks_summary"].items()]
    md += ["", f"GPTQ code files (`{P.CODES_ROOT}`), sha256:", ""]
    md += [f"- `{p}`: `{s}`" for p, s in sorted(rec["files"].items())]
    tex = ["% tab:ptq round 3 (part O): WikiText-2 / C4 PPL (native sm120, 16x64) and GSM8K strict-match (%), "
           "lm-eval gsm8k_llama, thinking off; GPTQ with activation quantization on during calibration"]
    for model, title in MODELS:
        tex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\")
        o0 = rec["o0"].get(model)
        if o0:
            tex.append(f"BF16 & -- & {o0['ppl']['wiki']:.2f} & {o0['ppl']['c4']:.2f} & "
                       f"{100 * o0['gsm8k']['strict_match']['accuracy']:.1f} \\\\")
        for row, (method, label) in zip(GSM_ROWS, (("GPTQ (act.)", "NVFP4"), ("GPTQ (act.)", "FourOverSix"),
                                                     ("GPTQ (act.)", "FlipQuant (16$\\times$64)"),
                                                     ("GPTQ cand. + map", "FlipQuant (16$\\times$64)"))):
            fmt = row.split("_", 1)[1]
            cells = ((rec["o2"].get(model) or {}).get(fmt) or {}).get("ppl") if row.startswith("gptq_") else (
                (rec["o3"].get(model) or {}).get("ppl") or {}).get("ppl")
            g = ((rec["o4"].get(model) or {}).get(row) or {}).get("gsm8k")
            gt = f"{100 * g['strict_match']['accuracy']:.1f}" if g else "TBD"
            tex.append(f"{method} & {label} & " + (f"{cells['wiki']:.2f} & {cells['c4']:.2f}" if cells else "TBD & TBD")
                       + f" & {gt} \\\\")
    cache_f.write_text(json.dumps(cache, indent=1) + "\n")
    (OUT / "round3.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    (OUT / "table_ptq_round3.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "shares":
        shares()
    elif what == "pilot":
        pilot()
    elif what == "report":
        report()
    else:
        sys.exit(__doc__)

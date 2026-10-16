"""Part Z (PROTOCOL.md): B (memory) and C1 (ownership) for Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct, in B's and
C1's table formats (report_bc.py), plus the paper's two claims. Writes results/paper_eval/llama_bc/ (LLAMA_BC.md,
llama_bc.json); B's and C1's own records (results/paper_eval/memory, verify) are not touched.

Claims checked:
- installed weights: FlipQuant's quantized linears and whole model equal NVFP4's and FourOverSix's to the byte;
- memory: after load and at the 1x2048 prefill peak, FlipQuant differs from NVFP4 / FourOverSix by at most 0.02 GiB.
Ownership: every map exact, with 0 format mismatches; the 24-map total adds these 6 maps to C1's 18.

    python report_llama_bc.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_bc as BC  # noqa: E402

RUN = BC.RUN
OUT = BC.OUT / "llama_bc"
MODELS = [("llama3.2-1b-instruct", "Llama-3.2-1B-Instruct"), ("llama3.1-8b-instruct", "Llama-3.1-8B-Instruct")]
POLS = ["bf16", "nvfp4", "fo6", "fq-8x64", "fq-16x64", "fq-256x64"]
FQS = ["fq-8x64", "fq-16x64", "fq-256x64"]
G = BC.G
TOL_GIB = 0.02


def map_path(m, u):
    return RUN / "llama_instruct" / m / "maps" / f"flipquant_{u}.pt"


def main():
    rec = dict(memory={}, memory_claims={}, ownership={}, ownership_totals={})
    L = ["# Part Z: memory (B) and ownership (C1) for the two Llama-Instruct models", "",
         "B's and C1's protocols (run_gpu.memory / run_gpu.ownership, a30df07 / C1), run with part P's maps on flipquant "
         "paper-sm120-runs (its code = 4bbe0b1; main 120173a's registry lacks both models) and build_V. PROTOCOL.md part "
         "Z. The six models of B and C1 are in results/paper_eval/memory and results/paper_eval/verify.", "",
         "## B: memory on the RTX PRO 6000", "",
         "The columns are B's (MEMORY.md):",
         "- **Weights:** every parameter and buffer on the GPU after the load and install.",
         "- **After load:** torch.cuda.memory_allocated.",
         "- **Peak:** torch.cuda.max_memory_allocated during one 1x2048 eager prefill.", "",
         "| model | policy | weights | quantized linears / BF16 | whole model / BF16 | after load | 1x2048 peak |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for m, title in MODELS:
        for p in POLS:
            f = RUN / "memory" / m / f"{p}.json"
            if f.exists():
                rec["memory"].setdefault(m, {})[p] = json.loads(f.read_text())
        mm = rec["memory"].get(m, {})
        if "bf16" not in mm:
            continue
        bf = mm["bf16"]["model_bytes"]
        for p, r in mm.items():
            q = r["quantized_linears"]
            L.append(f"| {title} | {p} | {r['model_bytes'] / G:.2f} GiB | {q['installed_over_bf16']:.4f} | "
                     f"{r['model_bytes'] / bf:.3f} | {r['allocated_after_load'] / G:.2f} GiB | "
                     f"{r['peak_prefill_1x2048'] / G:.2f} GiB |")
        # the claims
        if all(p in mm for p in ("nvfp4", "fo6", *FQS)):
            c = dict(installed_equal={}, model_equal={}, after_load_max_diff_gib=0.0, peak_max_diff_gib=0.0)
            for fq in FQS:
                for ref in ("nvfp4", "fo6"):
                    a, b = mm[fq], mm[ref]
                    c["installed_equal"][f"{fq} vs {ref}"] = (a["quantized_linears"]["installed_bytes"]
                                                              == b["quantized_linears"]["installed_bytes"])
                    c["model_equal"][f"{fq} vs {ref}"] = a["model_bytes"] == b["model_bytes"]
                    c["after_load_max_diff_gib"] = max(c["after_load_max_diff_gib"],
                                                      abs(a["allocated_after_load"] - b["allocated_after_load"]) / G)
                    c["peak_max_diff_gib"] = max(c["peak_max_diff_gib"],
                                                 abs(a["peak_prefill_1x2048"] - b["peak_prefill_1x2048"]) / G)
            c["byte_equal"] = all(c["installed_equal"].values()) and all(c["model_equal"].values())
            c["within_tolerance"] = c["after_load_max_diff_gib"] <= TOL_GIB and c["peak_max_diff_gib"] <= TOL_GIB
            c["installed_bytes"] = {p: mm[p]["quantized_linears"]["installed_bytes"] for p in ("nvfp4", "fo6", *FQS)}
            rec["memory_claims"][m] = c
    L += ["", "**The paper's memory claims, per model:**", ""]
    for m, title in MODELS:
        c = rec["memory_claims"].get(m)
        if c is None:
            L.append(f"- {title}: TBD")
            continue
        L += [f"- **{title}:**",
              f"  - FlipQuant's installed quantized linears and whole model equal NVFP4's and FourOverSix's to the byte: "
              f"**{'yes' if c['byte_equal'] else 'NO'}**. Quantized linears: "
              f"{c['installed_bytes']['nvfp4']:,} B for NVFP4, "
              + ", ".join(f"{c['installed_bytes'][p]:,} B for {p}" for p in ("fo6", *FQS)) + ".",
              f"  - After load and at the 1x2048 peak, FlipQuant differs from NVFP4 / FourOverSix by at most "
              f"{c['after_load_max_diff_gib']:.4f} / {c['peak_max_diff_gib']:.4f} GiB. Within 0.02 GiB: "
              f"**{'yes' if c['within_tolerance'] else 'NO'}**."]
    # C1
    sys.path.insert(0, "/home/dev/n16k64_campaign/fqopt/wt")
    from flipquant import maps as M
    L += ["", "## C1: ownership check of the 6 Llama-Instruct maps", "",
          "C1's check and columns (VERIFY.md):",
          "- Every weight element is decoded by the kernel itself (an identity activation through the GEMM), and must "
          "equal its stored value under the map's format (flipquant `--ownership-check`; the install raises on any "
          "mismatch).",
          "- Weight elements: the checked modules' out x in, from the checkpoint's safetensors headers.", "",
          "| model | unit | modules | weight elements checked | informative elements | E0M3 tiles | E0M3 elements observed | exact | format mismatches |",
          "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for m, title in MODELS:
        for u in BC.UNITS:
            f = RUN / "ownership" / f"{m}_{u}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            own = r["native"]["ownership"]
            tiles, unit, _ = M.load(map_path(m, u))
            elements = BC.weight_elements(m, r["source"], own)
            o = dict(modules=len(own), elements=elements, informative=sum(v["informative_elements"] for v in own.values()),
                     e0m3_tiles=sum(v["e0m3_tiles"] for v in own.values()),
                     e0m3_elements=sum(v["e0m3_elements_observed"] for v in own.values()),
                     exact=all(v["exact"] for v in own.values()),
                     mismatches=sum(v["format_mismatches"] for v in own.values()), map_modules=len(tiles),
                     map=str(map_path(m, u)))
            rec["ownership"].setdefault(m, {})[u] = o
            L.append(f"| {title} | {u} | {o['modules']} of {o['map_modules']} | "
                     f"{'—' if o['elements'] is None else format(o['elements'], ',')} | {o['informative']:,} | "
                     f"{o['e0m3_tiles']:,} | {o['e0m3_elements']:,} | {o['exact']} | {o['mismatches']} |")
    six = [o for mm in rec["ownership"].values() for o in mm.values()]
    c1 = json.loads((BC.OUT / "verify" / "verify.json").read_text())["ownership"]
    eighteen = [o for mm in c1.values() for o in mm.values()]
    tot = dict(llama_maps=len(six), llama_elements=sum(o["elements"] or 0 for o in six),
               c1_maps=len(eighteen), c1_elements=sum(o["elements"] or 0 for o in eighteen))
    tot.update(maps=tot["llama_maps"] + tot["c1_maps"], elements=tot["llama_elements"] + tot["c1_elements"],
               all_exact=all(o["exact"] for o in six + eighteen), mismatches=sum(o["mismatches"] for o in six + eighteen),
               all_modules_checked=all(o["modules"] == o["map_modules"] for o in six + eighteen))
    rec["ownership_totals"] = tot
    L += ["", f"**Totals.**",
          f"- The 6 Llama maps: {tot['llama_elements']:,} weight elements checked.",
          f"- With C1's 18 maps ({tot['c1_elements']:,}): {tot['maps']} maps, {tot['elements']:,} weight elements.",
          f"- Every map exact: **{'yes' if tot['all_exact'] else 'NO'}**. Format mismatches: {tot['mismatches']}. Every "
          f"module of every map checked: {'yes' if tot['all_modules_checked'] else 'NO'}."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "llama_bc.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (OUT / "LLAMA_BC.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

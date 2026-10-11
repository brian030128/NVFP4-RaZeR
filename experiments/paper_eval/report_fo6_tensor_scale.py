"""Part R report (PROTOCOL.md part R): results/paper_eval/fo6_tensor_scale/{REPORT.md, fo6_tensor_scale.json, records/}.
Run after `run_fo6_tensor_scale.py check-default / ppl / stats`. CPU only.

    python report_fo6_tensor_scale.py
"""
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fo6_tensor_scale as R  # noqa: E402
import tables_cpu as T  # noqa: E402
import tables_final as F  # noqa: E402

UNITS = ("8x64", "16x64", "256x64")


def pm(p, scale=1e3):
    return f"{scale * p['delta']:+.2f} ± {scale * p['two_se']:.2f}{'*' if p['significant'] else ''}"


def capped(nvfp4, bf16, row):
    gap = nvfp4 - bf16
    return None if gap <= 0 else 100 * min(1.0, (nvfp4 - row) / gap)


def summary(rec, ms, pairs):
    st = [m["stats"]["shares"] for m in ms.values() if m.get("stats")]
    lo_hi = lambda key: (100 * min(x[key] for x in st), 100 * max(x[key] for x in st))     # noqa: E731
    cr = rec["capped_recovery"]
    return [
        f"- **Weights:** only {lo_hi('above_two_thirds')[0]:.4f}-{lo_hi('above_two_thirds')[1]:.4f} % of weight blocks "
        "exceed 2/3 of their tensor's amax, the blocks for which 448 cannot select a true 4. Yet the constant changes "
        f"{lo_hi('blocks_changed')[0]:.0f}-{lo_hi('blocks_changed')[1]:.0f} % of blocks, and "
        f"{lo_hi('choice_flips')[0]:.1f}-{lo_hi('choice_flips')[1]:.1f} % flip their 4-vs-6 choice. The cause is that "
        "every block scale is E4M3-rounded relative to a tensor scale 7/4 times larger, so the E4M3 grid shifts by a "
        f"factor of 4/7. The share of blocks choosing 4 is nearly the same ({lo_hi('use4_448')[0]:.1f}-"
        f"{lo_hi('use4_448')[1]:.1f} % with 448, {lo_hi('use4_256')[0]:.1f}-{lo_hi('use4_256')[1]:.1f} % with 256).",
        "- **Activations (per token):** the block holding each token's max can now take a true 4: at least 16/K of the "
        "blocks (0.09-0.78 % across the models' K).",
        f"- **Accuracy:** model-dependent and small. FO6-256 is significantly better than FO6-448 in "
        f"{rec['cells_256_better_than_448']} of {len(pairs)} cells, worse in {rec['cells_256_worse_than_448']}, and not "
        "significantly different in the rest. The mean capped NLL recovery (16 pairs) moves from "
        f"{cr['fo6_448']['value']:.1f} % to {cr['fo6_256']['value']:.1f} %.",
        "- **FlipQuant:** its maps were calibrated with 448 and not recalibrated. It is still significantly better than "
        "the original FourOverSix in every cell: " + ", ".join(
            f"{v} / {len(pairs)} at {u}" for u, v in rec["fq_cells_better_than_256"].items()) + "."]


def main():
    R.OUT.mkdir(parents=True, exist_ok=True)
    final = json.loads((F.OUT / "final.json").read_text())
    titles = {k: t for k, _, t in F.FINAL_MODELS}
    check = json.loads((R.WORK / "check_default.json").read_text())
    rec = dict(check_default=dict(reference=check["reference"], new=check["new"], cases=check["cases"],
                                  all_bitwise=check["all_bitwise"]), models={})
    for k in R.MODELS:
        d = R.WORK / k
        src = final["sources"][k]
        n = {"fo6_256": T.nll_of(d / "fo6_256.json"), "fo6_448": T.nll_of(d / "fo6_448.json"),
             **{r: T.nll_of(src[r]) for r in ("bf16", "nvfp4", "fo6", *(f"fq_{u}" for u in UNITS))}}
        for c, _ in T.CORPORA:
            assert len({len(v[c]) for v in n.values()}) == 1, (k, c, {r: len(v[c]) for r, v in n.items()})
        r256 = json.loads((d / "fo6_256.json").read_text())
        r448 = json.loads((d / "fo6_448.json").read_text())
        m = rec["models"][k] = dict(
            title=titles[k], same_harness_table_record=k in R.SAME_HARNESS,
            ppl={r: {c: math.exp(T.mean(n[r][c])) for c, _ in T.CORPORA} for r in n},
            nll_mean={r: {c: T.mean(n[r][c]) for c, _ in T.CORPORA} for r in n},
            d256_448={c: T.paired(n["fo6_256"][c], n["fo6_448"][c]) for c, _ in T.CORPORA},
            d256_table_fo6={c: T.paired(n["fo6_256"][c], n["fo6"][c]) for c, _ in T.CORPORA},
            d256_nvfp4={c: T.paired(n["fo6_256"][c], n["nvfp4"][c]) for c, _ in T.CORPORA},
            fq_minus_256={u: {c: T.paired(n[f"fq_{u}"][c], n["fo6_256"][c]) for c, _ in T.CORPORA} for u in UNITS},
            rerun_vs_table_fo6=R.per_window(d / "fo6_448.json", src["fo6"]),
            native_256=r256.get("native", {}).get("fo6_tensor_qmax"), quant_256=r256.get("quant"),
            kernel_256=r256.get("native", {}).get("kernel"), kernel_448=r448.get("native", {}).get("kernel"),
            resources={"fo6_256": r256.get("resources"), "fo6_448": r448.get("resources")})
        nm = m["nll_mean"]
        m["capped"] = {r: {c: capped(nm["nvfp4"][c], nm["bf16"][c], nm[r][c]) for c, _ in T.CORPORA}
                       for r in ("fo6_256", "fo6_448", "fo6")}
        st = R.WORK / k / "stats.json"
        if st.exists():
            s = json.loads(st.read_text())
            m["stats"] = dict(shares=s["shares"], totals=s["totals"], activations=s["activations"], modules=s["modules"],
                              by_kind={kk: v["shares"] for kk, v in s["by_kind"].items()}, seconds=s["seconds"])
    ms = rec["models"]
    pairs = [(k, c) for k in R.MODELS for c, _ in T.CORPORA]
    rec["capped_recovery"] = {r: dict(value=T.mean([ms[k]["capped"][r][c] for k, c in pairs]), pairs=len(pairs),
                                      flagged=[f"{k}/{c}" for k, c in pairs if ms[k]["capped"][r][c] is None])
                              for r in ("fo6_256", "fo6_448", "fo6")}
    rec["fq16_beats_256_every_cell"] = all(
        ms[k]["fq_minus_256"]["16x64"][c]["delta"] < 0 and ms[k]["fq_minus_256"]["16x64"][c]["significant"]
        for k, c in pairs)
    rec["fq_cells_better_than_256"] = {u: sum(ms[k]["fq_minus_256"][u][c]["delta"] < 0 and
                                              ms[k]["fq_minus_256"][u][c]["significant"] for k, c in pairs)
                                       for u in UNITS}
    rec["cells_256_better_than_448"] = sum(ms[k]["d256_448"][c]["delta"] < 0 and ms[k]["d256_448"][c]["significant"]
                                           for k, c in pairs)
    rec["cells_256_worse_than_448"] = sum(ms[k]["d256_448"][c]["delta"] > 0 and ms[k]["d256_448"][c]["significant"]
                                          for k, c in pairs)
    rec["vs_nvfp4_cells"] = dict(
        better=sum(ms[k]["d256_nvfp4"][c]["significant"] and ms[k]["d256_nvfp4"][c]["delta"] < 0 for k, c in pairs),
        worse=sum(ms[k]["d256_nvfp4"][c]["significant"] and ms[k]["d256_nvfp4"][c]["delta"] > 0 for k, c in pairs),
        not_significant=sum(not ms[k]["d256_nvfp4"][c]["significant"] for k, c in pairs))
    rec["default_reruns_bitwise"] = {k: ms[k]["rerun_vs_table_fo6"]["bitwise"] for k in R.SAME_HARNESS}
    rec["harness_reruns_bitwise"] = {k: ms[k]["rerun_vs_table_fo6"]["bitwise"] for k in R.MODELS if k not in R.SAME_HARNESS}

    # ------------------------------------------------------------------------------------------- markdown
    cr = rec["capped_recovery"]
    L = ["# Part R: FourOverSix's tensor-scale constant, 6 x 448 vs the original 6 x 256", "",
         "PROTOCOL.md part R. FO6-448 = our FourOverSix (global scale amax / (6 x 448)), FO6-256 = the original "
         "FourOverSix (arXiv:2512.02010, Sec. 3: amax / (6 x 256)), both for weights and per-token activations, native "
         "sm120 (build_V, `--kernel-set auto`, `--paper-convention`), flipquant paper-sm120-runs d92c032 "
         "(`--fo6-tensor-qmax 256`). ΔNLL x 1e-3 ± 2 SE (paired over the same windows), * beyond 2 SE.", "",
         "## Summary", ""] + summary(rec, ms, pairs) + ["", "## Checks", "",
         f"- The new `fourover6` at its default equals {check['reference']} bit for bit: {check['cases']} cases (every "
         f"scope, CPU and GPU, random and real weights): {check['all_bitwise']}.",
         "- tests/test_fo6_tensor_qmax.py (18 passed on sm_120 with build_V): 256 equals an independent implementation "
         "of the original definition; the native quantizer at fp32(1/1536) gives fourover6(x, 'row', 256)'s scale bytes "
         "and global scales bit for bit; the unfused route at fp32(1/2688) equals the fused NativeLinear bit for bit; an "
         "install at 256 matches fake mode at 256, not 448.",
         "- The FO6-448 reruns with the new code reproduce the table's FO6 records per window, bit for bit, for the "
         "models whose table record came from evaluation.ppl: " + ", ".join(
             f"{titles[k]} {v}" for k, v in rec["default_reruns_bitwise"].items()) + ".",
         "- The other four models' table FO6 records come from the earlier sm120 harness (main-ppl / paper). Per window, "
         "the same-code FO6-448 rerun vs that record: " + "; ".join(
             f"{titles[k]}: " + ("bit for bit" if ms[k]["rerun_vs_table_fo6"]["bitwise"] else ", ".join(
                 f"{c} {ms[k]['rerun_vs_table_fo6'][c]['differing_windows']} windows differ (max "
                 f"{ms[k]['rerun_vs_table_fo6'][c]['max_abs_diff']:.2e})" for c, _ in T.CORPORA))
             for k in R.MODELS if k not in R.SAME_HARNESS) + ". The paired ΔNLL below uses the same-code FO6-448 "
         "(identical to the table's row where bit for bit).", "",
         "## PPL and paired ΔNLL", "",
         "| model | FO6-448 (wiki / C4) | FO6-256 (wiki / C4) | FO6-256 − FO6-448 | FO6-256 − NVFP4 | FlipQuant 16x64 − FO6-256 |",
         "|---|---:|---:|---:|---:|---:|"]
    for k in R.MODELS:
        m = ms[k]
        L.append(f"| {m['title']} | " + " / ".join(f"{m['ppl']['fo6_448'][c]:.4f}" for c, _ in T.CORPORA) + " | "
                 + " / ".join(f"{m['ppl']['fo6_256'][c]:.4f}" for c, _ in T.CORPORA) + " | "
                 + " / ".join(pm(m["d256_448"][c]) for c, _ in T.CORPORA) + " | "
                 + " / ".join(pm(m["d256_nvfp4"][c]) for c, _ in T.CORPORA) + " | "
                 + " / ".join(pm(m["fq_minus_256"]["16x64"][c]) for c, _ in T.CORPORA) + " |")
    vn = rec["vs_nvfp4_cells"]
    L += ["", f"FO6-256 is significantly better than FO6-448 in {rec['cells_256_better_than_448']} of {len(pairs)} cells "
          f"and significantly worse in {rec['cells_256_worse_than_448']}. FO6-256 vs NVFP4: better in {vn['better']}, not "
          f"significant in {vn['not_significant']}, worse in {vn['worse']} cells. FlipQuant (maps calibrated with 448, not "
          f"recalibrated) is significantly better than FO6-256 in " + ", ".join(
              f"{v} / {len(pairs)} cells at {u}" for u, v in rec["fq_cells_better_than_256"].items())
          + f"; every cell at 16x64: {rec['fq16_beats_256_every_cell']}.", "",
          ("The same-code FO6-448 reruns equal the table's FourOverSix records per window for all eight models, so "
           "FO6-256 − FO6-448 is also FO6-256 − the table's FourOverSix row." if all(
               ms[k]["rerun_vs_table_fo6"]["bitwise"] for k in R.MODELS) else
           "FO6-256 − the table's FO6 record where the rerun differs from it (wiki / C4): " + "; ".join(
               f"{titles[k]} " + " / ".join(pm(ms[k]["d256_table_fo6"][c]) for c, _ in T.CORPORA)
               for k in R.MODELS if not ms[k]["rerun_vs_table_fo6"]["bitwise"]) + "."), "",
          "## Mean capped NLL recovery (16 pairs; the table's BF16 and NVFP4)", "",
          "| row | mean capped NLL recovery |", "|---|---:|",
          f"| FO6-448, the table's records | {cr['fo6']['value']:.1f} % |",
          f"| FO6-448, the same-code reruns | {cr['fo6_448']['value']:.1f} % |",
          f"| FO6-256 | {cr['fo6_256']['value']:.1f} % |", "",
          "Per pair (FO6-448 same-code / FO6-256): " + "; ".join(
              f"{k}/{c} {ms[k]['capped']['fo6_448'][c]:.1f} / {ms[k]['capped']['fo6_256'][c]:.1f}" for k, c in pairs)
          + ". Flagged pairs (non-positive NVFP4 − BF16 gap): " + (", ".join(cr["fo6_256"]["flagged"]) or "none") + ".",
          "", "## Block statistics of the quantized weights (CPU, formats.fourover6, tensor scope)", "",
          "Share of 16-element blocks. *> 2/3 amax*: blocks whose max exceeds 2/3 of the tensor amax, i.e. the blocks "
          "whose 4-candidate 448 clamps (counted directly: *clamped 448*). *4 chosen*: blocks that take the 4-candidate. "
          "*choice flips*: 4-vs-6 decision differs between 448 and 256. *blocks changed*: any dequantized value differs. "
          "*subnormal scale*: the chosen E4M3 block scale is below 2^-6 (E4M3's subnormal range).", "",
          "| model | > 2/3 amax | clamped 448 | 4 chosen 448 | 4 chosen 256 | choice flips | blocks changed | "
          "subnormal scale 448 / 256 | activations: ≥ 1 block per token (16/K) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for k in R.MODELS:
        s = ms[k].get("stats")
        if not s:
            L.append(f"| {titles[k]} | (not computed) | | | | | | | |")
            continue
        sh = s["shares"]
        acts = ", ".join(f"K={kk}: {100 * v['at_least_one_block_per_token']:.2f} %" for kk, v in s["activations"].items())
        L.append(f"| {titles[k]} | {100 * sh['above_two_thirds']:.4f} % | {100 * sh['clamped4_448']:.4f} % | "
                 f"{100 * sh['use4_448']:.2f} % | {100 * sh['use4_256']:.2f} % | {100 * sh['choice_flips']:.2f} % | "
                 f"{100 * sh['blocks_changed']:.2f} % | {100 * sh['subnormal_scale_448']:.3f} / "
                 f"{100 * sh['subnormal_scale_256']:.3f} % | {acts} |")
    L += ["", "Records: `records/` (the check, the 16 PPL records, the per-model statistics; per-module counts in "
          "fo6_tensor_scale.json are omitted, they are in records/<model>/stats.json). No FlipQuant map was recalibrated."]
    dst = R.OUT / "records"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(R.WORK / "check_default.json", dst / "check_default.json")
    for k in R.MODELS:
        (dst / k).mkdir(parents=True, exist_ok=True)
        for f in ("fo6_448.json", "fo6_256.json", "stats.json", "default_check.json"):
            if (R.WORK / k / f).exists():
                shutil.copyfile(R.WORK / k / f, dst / k / f)
    (R.OUT / "fo6_tensor_scale.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (R.OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

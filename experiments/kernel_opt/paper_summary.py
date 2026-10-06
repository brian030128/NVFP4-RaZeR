"""The SM120 kernel section's numbers, recomputed from the committed records only (no GPU): results/kernel_opt/paper_summary/
(summary.json, tables.md, overhead.png). SUMMARY.md is written by hand around them.

Every comparison is a ratio of two per-forward GEMM sums measured in ONE session (deviation-2 method: isolated launches,
cold weights, CUPTI, 3 rotated rounds x 30; per forward = the sum over the quantized text Linears). A comparison that
needs two sessions is a chain (the product of paired links whose builds and tables match exactly) or a splice (two
overheads against different references), and is labelled as such.

- before: the paper kernels against the paper stock (the paper's own run, T >= 128, and the kernel-opt sessions that
  measured them, T from 1);
- after: the build_V kernels against stock_ko (amendments 17 and 18);
- each optimization's paired effect; the chain from the paper kernel to the current one per unit;
- the 4096^3 decompositions (C2 records) and the E0M3-share summary (C3v).

    python experiments/kernel_opt/paper_summary.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = R / "kernel_opt" / "paper_summary"
MODELS = ("llama8b", "mistral7b", "phi4", "qwen27b")
NAMES = {"llama8b": "Llama-3.1-8B", "mistral7b": "Mistral-7B-v0.3", "phi4": "Phi-4", "qwen27b": "Qwen3.8-27B"}
T12 = (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
T7 = (128, 256, 512, 1024, 2048, 4096, 8192)
BANDS = {"T ≤ 16": (1, 4, 16), "32–128": (32, 64, 128), "256–1024": (256, 512, 1024), "≥ 2048": (2048, 4096, 8192)}
TAGS = ("typical", "worst")


def load(rel):
    return json.loads((R / rel).read_text())


def per_model(rel, key=None):
    r = load(rel)
    if key:
        r = r[key]
    return {m: r[m]["sums"] for m in MODELS}


# The M1 records: per model, the per-forward sums '<config>@<T>' in µs.
SOURCES = {
    "paper_dev2": dict(file="paper/tables/tables.json", session="2026-09-30 06:23–06:36", commit="5f64905",
                       what="the paper's deviation-2 GEMM run (step 06b)", T=T7,
                       get=lambda: {m: v["per_forward_gemm_us"]
                                    for m, v in load("paper/tables/tables.json")["gemm_isolated"]["models"].items()}),
    "ab1": dict(file="kernel_opt/ab1.json", session="2026-09-30 10:41–10:52", commit="540e1da",
                what="optimization 1, M1", T=T12, get=lambda: per_model("kernel_opt/ab1.json", "gemm")),
    "ab1b": dict(file="kernel_opt/ab1b.json", session="2026-09-30 14:21–14:34", commit="56dc82d",
                 what="optimization 1b (amendment 1), M1", T=T12, get=lambda: per_model("kernel_opt/ab1b.json", "gemm")),
    "ab2": dict(file="kernel_opt/opt2/ab2.json", session="2026-09-30 17:53–18:43", commit="74dfce1",
                what="#2 (amendment 2), M1′", T=T12, get=lambda: per_model("kernel_opt/opt2/ab2.json", "gemm")),
    "abA1": dict(file="kernel_opt/A1/abA1.json", session="2026-09-30 19:24–19:49", commit="95fa44f",
                 what="A′ (amendment 3), M1", T=T12, get=lambda: per_model("kernel_opt/A1/abA1.json", "gemm")),
    "abR": dict(file="kernel_opt/retune/b/abR.json", session="2026-09-30 22:23–22:55", commit="fefdba4",
                what="4b re-tune (amendment 4b), M1", T=T12, get=lambda: per_model("kernel_opt/retune/b/abR.json", "models")),
    "ab4": dict(file="kernel_opt/4/ab4.json", session="2026-10-01 00:09–00:57", commit="52b81c3",
                what="#4 (amendment 5), M1", T=T12, get=lambda: per_model("kernel_opt/4/ab4.json")),
    "abT": dict(file="kernel_opt/t0/abT.json", session="2026-10-01 03:17–04:18", commit="fe21c52",
                what="t0 (amendments 6/6b), M1", T=T12, get=lambda: per_model("kernel_opt/t0/abT.json")),
    "cum": dict(file="kernel_opt/cum/cum_gemm.json", session="2026-10-01 05:55–09:13", commit="32b4335",
                what="the cumulative 16x64 run (amendment 9), M1", T=T12, get=lambda: per_model("kernel_opt/cum/cum_gemm.json")),
    "w8": dict(file="kernel_opt/w8/w8.json", session="2026-10-01 13:16–13:30", commit="1dc4b32",
               what="the 8x64 baseline (amendment 10), M1", T=T12, get=lambda: per_model("kernel_opt/w8/w8.json")),
    "w8p2": dict(file="kernel_opt/w8/p2/w8p2.json", session="2026-10-02 15:36–16:10", commit="fb434c7",
                 what="8x64 P2 + P1 (amendment 11), M1", T=T12, get=lambda: per_model("kernel_opt/w8/p2/w8p2.json")),
    "w8p3b": dict(file="kernel_opt/w8/p3b/w8p3b.json", session="2026-10-02 17:27–17:43", commit="0c8dbcb",
                  what="8x64 P3b (amendment 12b), M1", T=T12, get=lambda: per_model("kernel_opt/w8/p3b/w8p3b.json")),
    "U": dict(file="kernel_opt/U/U.json", session="2026-10-03 08:11–08:49", commit="01fdfbc",
              what="U (amendment 17), M1", T=T12, get=lambda: per_model("kernel_opt/U/U.json")),
    "V": dict(file="kernel_opt/V/V.json", session="2026-10-03 11:16–11:58", commit="5c6e080",
              what="V (amendment 18), M1", T=T12, get=lambda: per_model("kernel_opt/V/V.json")),
}
_cache = {}


def sums(src):
    if src not in _cache:
        _cache[src] = SOURCES[src]["get"]()
    return _cache[src]


def ratio(src, a, b, m, t):
    s = sums(src)[m]
    return 100.0 * (s[f"{a}@{t}"] / s[f"{b}@{t}"] - 1.0)


def stats(cells, ts):
    allc = [cells[m][t] for m in MODELS for t in ts]
    return dict(per_t_median={t: statistics.median(cells[m][t] for m in MODELS) for t in ts},
                bands={n: statistics.median(cells[m][t] for m in MODELS for t in tt if t in ts)
                       for n, tt in BANDS.items() if any(t in ts for t in tt)},
                median=statistics.median(allc), min=min(allc), max=max(allc), below_zero=sum(v < 0 for v in allc),
                cells=len(allc), median_t128=statistics.median(cells[m][t] for m in MODELS for t in ts if t >= 128))


def compare(src, a, b, **meta):
    """a vs b in one session, per model and T (%)."""
    ts = SOURCES[src]["T"]
    cells = {m: {t: ratio(src, a, b, m, t) for t in ts} for m in MODELS}
    return dict(source=src, file=SOURCES[src]["file"], session=SOURCES[src]["session"], a=a, b=b,
                basis="paired, one session", per_model=cells, **stats(cells, ts), **meta)


def product(parts, ts, **meta):
    """prod(1 + x_i) - 1 per model and T: a chain of paired links or a splice of two sessions."""
    cells = {m: {t: 100.0 * (_prod(p["per_model"][m][t] for p in parts) - 1.0) for t in ts} for m in MODELS}
    return dict(per_model=cells, links=[f"{p['source']}: {p['a']} / {p['b']}" for p in parts], **stats(cells, ts), **meta)


def _prod(xs):
    out = 1.0
    for x in xs:
        out *= 1.0 + x / 100.0
    return out


# Each optimization's paired effect: (unit, name, amendment, source, after, before, status).
STEPS = [
    ("16x64", "#2: frequency-aware dispatch (all-E2M1 pattern first)", "2", "ab2", "freq_16x64_{}", "mixed_16x64_{}",
     "adopted (74dfce1)"),
    ("16x64", "t0: site-0 PRMT tags dropped, default dispatch", "6/6b", "abT", "t0_16x64_{}", "mixed_16x64_{}",
     "measured; the adopted form is t0 on #2's dispatch"),
    ("16x64", "t0 on #2's dispatch", "6/6b", "abT", "freqt0_16x64_{}", "freq_16x64_{}", "adopted (038b448)"),
    ("16x64", "#4: 64×64 epilogue tile + tile-scheduler order", "5", "ab4", "e_16x64_{}", "mixed_16x64_{}",
     "adopted (038b448)"),
    ("16x64", "4b: act-warm width re-tune (on #2's dispatch, build_freq)", "4b", "abR", "m16_{}_new", "m16_{}_cur",
     "adopted (038b448)"),
    ("16x64", "all of the above with schedule rows: build_7freq mixed_ko vs the paper set", "9", "cum",
     "comb_16x64_{}", "paper_16x64_{}", "the deployed path from 038b448"),
    ("16x64", "REDUX: uniform-branch dispatch on the wide tiles", "17 A", "U", "U16_{}", "A16_{}", "adopted (9c81422)"),
    ("8x64", "opt 1: narrow token tiles (m16/m32/m64) for weights on B", "base", "ab1", "auto_wB_{}", "n8k64_wB_{}",
     "adopted (540e1da)"),
    ("8x64", "opt 1b: cooperative 128×64 tile for mid T", "1", "ab1b", "auto_wB_{}", "auto_wB1_{}",
     "adopted (56dc82d)"),
    ("8x64", "#2's dispatch on the 1b set", "10", "w8", "wBfreq_{}", "wB_{}", "descriptive; adopted with t0 in 11"),
    ("8x64", "t0 for the wB family, default dispatch", "11", "w8p2", "wBt0_{}", "wB_{}", "measured"),
    ("8x64", "t0 + #2's dispatch (P2 with P1)", "11", "w8p2", "wBt0freq_{}", "wB_{}", "adopted (0c8dbcb)"),
    ("8x64", "P3b: reduced width table + scheduler rows", "12b", "w8p3b", "wBp3b_{}", "wBp2_{}", "adopted (0c8dbcb)"),
    ("8x64", "pipelined flag reads", "17 A", "U", "U8_{}", "A8_{}", "adopted (9c81422)"),
    ("256x64", "#2's dispatch on the 16-arm paper kernel", "2", "ab2", "freq_256x64_{}", "mixed_256x64_{}",
     "superseded by A′"),
    ("256x64", "A′: 4-arm kernel with 32-row granules (g32)", "3", "abA1", "g32_256x64_{}", "mixed_256x64_{}",
     "adopted (2b60b01)"),
    ("256x64", "t0 on A′", "6/6b", "abT", "m256t0_{}", "m256_{}", "not adopted then (amendment 7); part of 18"),
    ("256x64", "4b widths on A′", "4b", "abR", "m256_{}_new", "m256_{}_cur", "not adopted (A′ kept the paper table)"),
    ("256x64", "amendment 18 builds (REDUX + t0 + e64) at A′'s widths", "18", "V", "B_{}", "A_{}", "part of 18"),
    ("256x64", "amendment 18 table (table_v) on build_V", "18", "V", "C_{}", "B_{}", "part of 18"),
    ("256x64", "amendment 18 total: mixed256_ko vs A′", "18", "V", "C_{}", "A_{}", "adopted (4ae044a)"),
    ("stock", "#4 on stock", "5", "ab4", "stock_e", "stock_wA", "adopted (038b448)"),
    ("stock", "4b widths on stock", "4b", "abR", "stock_wA_new", "stock_wA_cur", "adopted (038b448)"),
    ("stock", "stock_ko vs the paper stock (#4 + 4b + rows)", "9", "cum", "stock_ko", "stock_paper", "the deployed stock"),
    ("stock", "stock_wB_ko vs stock_wB (weights on B, tuned alike)", "12b", "w8p3b", "stock_wB_ko", "stock_wB",
     "adopted (0c8dbcb); not a FourOverSix path"),
]
# The chains of paired links from the paper kernel to the build_V kernel; each link's 'before' is the previous link's
# 'after' (same builds, same table rows), measured in another session.
CHAINS = {
    "16x64": [("cum", "comb_16x64_{}", "paper_16x64_{}"), ("U", "U16_{}", "A16_{}")],
    "8x64": [("ab1", "auto_wB_{}", "n8k64_wB_{}"), ("ab1b", "auto_wB_{}", "auto_wB1_{}"), ("w8p2", "wBt0freq_{}", "wB_{}"),
             ("w8p3b", "wBp3b_{}", "wBp2_{}"), ("U", "U8_{}", "A8_{}")],
    "256x64": [("abA1", "g32_256x64_{}", "mixed_256x64_{}"), ("V", "C_{}", "A_{}")],
}


def kernel_tables():
    rec = dict(before={}, after={}, paper_vs_stock_ko={}, steps=[], chains={})
    B, A, P = rec["before"], rec["after"], rec["paper_vs_stock_ko"]
    for tag in TAGS:
        # before: the paper kernels against the paper stock
        B[f"16x64 {tag}"] = dict(
            paper_run=compare("paper_dev2", f"mixed_16x64_{tag}", "stock_wA", ref="paper stock_wA"),
            m1_2=compare("ab2", f"mixed_16x64_{tag}", "stock_wA", ref="paper stock_wA"),
            m1_9=compare("cum", f"paper_16x64_{tag}", "stock_paper", ref="paper stock"))
        B[f"256x64 {tag}"] = dict(
            paper_run=compare("paper_dev2", f"mixed_256x64_{tag}", "stock_wA", ref="paper stock_wA"),
            m1_2=compare("ab2", f"mixed_256x64_{tag}", "stock_wA", ref="paper stock_wA"))
        B[f"8x64 {tag}"] = dict(
            paper_run=compare("paper_dev2", f"n8k64_wB_{tag}", "stock_wB", ref="paper stock_wB (same placement)"),
            paper_run_wA=compare("paper_dev2", f"n8k64_wB_{tag}", "stock_wA", ref="paper stock_wA"),
            m1_opt1=compare("ab1", f"n8k64_wB_{tag}", "stock_wB", ref="paper stock_wB (same placement)"),
            m1_opt1_wA=compare("ab1", f"n8k64_wB_{tag}", "stock_wA", ref="paper stock_wA"))
        # after: build_V against stock_ko
        A[f"16x64 {tag}"] = compare("U", f"U16_{tag}", "stock_ko", ref="stock_ko", kernel="mixed_ko (build_U = build_V)")
        A[f"8x64 {tag}"] = compare("U", f"U8_{tag}", "stock_ko", ref="stock_ko", kernel="mixed_wB_ko (build_U = build_V)")
        A[f"256x64 {tag}"] = compare("V", f"C_{tag}", "stock_ko", ref="stock_ko", kernel="mixed256_ko (build_V, table_v)")
    stock_shift = compare("cum", "stock_paper", "stock_ko", ref="stock_ko")
    P["paper stock vs stock_ko"] = stock_shift
    for tag in TAGS:
        P[f"16x64 {tag}"] = compare("cum", f"paper_16x64_{tag}", "stock_ko", ref="stock_ko")
        P[f"8x64 {tag} (SPLICE)"] = product(
            [compare("ab1", f"n8k64_wB_{tag}", "stock_wA"), stock_shift], T12, basis="SPLICE",
            note="opt-1 session (paper n8k64_wB vs paper stock_wA) x amendment 9 (paper stock vs stock_ko)")
        P[f"256x64 {tag} (SPLICE)"] = product(
            [compare("ab2", f"mixed_256x64_{tag}", "stock_wA"), stock_shift], T12, basis="SPLICE",
            note="#2 session (paper set on 256x64 maps vs paper stock_wA) x amendment 9 (paper stock vs stock_ko)")
    for unit, name, amend, src, a, b, status in STEPS:
        for tag in (TAGS if "{}" in a else (None,)):
            rec["steps"].append(compare(src, a.format(tag), b.format(tag), unit=unit, name=name, amendment=amend,
                                        tag=tag, status=status, commit=SOURCES[src]["commit"]))
    for unit, links in CHAINS.items():
        for tag in TAGS:
            parts = [compare(s, a.format(tag), b.format(tag)) for s, a, b in links]
            rec["chains"][f"{unit} {tag}"] = product(parts, T12, basis="CHAIN of paired links")
    rec["chains"]["stock"] = product([compare("cum", "stock_ko", "stock_paper")], T12, basis="paired, one session")
    return rec


def r(a, b):
    return 100.0 * (a / b - 1.0)


def c2_tables():
    """4096^3 decompositions per mode (b2b / isolated / sustained), from the C2 records; % of the reference."""
    modes = ("b2b", "isolated", "sustained")
    out = {}
    # 16x64 before: the paper set (default dispatch) against the paper stock, amendment 6/6b's C2''' (one session)
    t = load("kernel_opt/t0/abT.json")["c2"]["summary"]
    rows = {}
    for md in modes:
        s = t[md]
        ceil_t0 = 1 + s["nodisp_t0 vs stock_wA"] / 100
        ceil = ceil_t0 / (1 + s["nodisp_t0 vs nodisp"] / 100)
        e2m1, real, e0m3 = (ceil_t0 * (1 + s[f"default_{k} vs nodisp_t0"] / 100) for k in ("e2m1", "real", "e0m3"))
        rows[md] = {"tiles (t0 ceiling vs stock)": r(ceil_t0, 1), "site-0 tags (tagged vs t0 ceiling)": r(ceil, ceil_t0),
                    "dispatch (all-E2M1 vs the kernel's ceiling)": r(e2m1, ceil),
                    "E0M3 tiles, real map (real vs all-E2M1)": r(real, e2m1), "all-E0M3 vs all-E2M1": r(e0m3, e2m1),
                    "total, real map vs stock": r(real, 1)}
    out["16x64 before: the paper set (default dispatch) vs the paper stock_wA; C2‴ (amendment 6/6b)"] = rows
    # 16x64 and 8x64 after: C2U (amendment 17), against stock_ko; the ceilings are the adopted paths' t0 ceilings
    u = load("kernel_opt/U/U.json")["c2"]
    rows16, rows16a, rows8, rows8a = {}, {}, {}, {}
    for md in modes:
        s = u[md]
        for ceil_key, k_after, k_prev, ra, rp in (("ceil16", "U16", "A16", rows16, rows16a), ("ceil8", "U8", "A8", rows8, rows8a)):
            c = 1 + s[f"{ceil_key} vs stock_ko"] / 100
            for k, rows_ in ((k_after, ra), (k_prev, rp)):
                e2m1, real, e0m3 = (1 + s[f"{k}_{x} vs stock_ko"] / 100 for x in ("e2m1", "real", "e0m3"))
                rows_[md] = {"tiles (t0 ceiling vs stock)": r(c, 1), "site-0 tags (tagged vs t0 ceiling)": 0.0,
                             "dispatch (all-E2M1 vs the kernel's ceiling)": r(e2m1, c),
                             "E0M3 tiles, real map (real vs all-E2M1)": r(real, e2m1), "all-E0M3 vs all-E2M1": r(e0m3, e2m1),
                             "total, real map vs stock": r(real, 1)}
    out["16x64 after: mixed_ko (build_V) vs stock_ko; C2U (amendment 17)"] = rows16
    out["16x64 just before REDUX: build_7freq mixed_ko vs stock_ko; C2U"] = rows16a
    # 8x64 before: the paper n8k64_wB (default dispatch), amendment 10's C2w (one session), against stock_ko
    w = load("kernel_opt/w8/w8.json")["c2"]["summary"]
    rows = {}
    for md in modes:
        s = w[md]
        wb = 1 + s["stock_wB vs stock_wA"] / 100
        ko = 1 + s["stock_ko vs stock_wA"] / 100
        ceil_t0 = wb * (1 + s["nodisp_wB_t0 vs stock_wB"] / 100)
        ceil = wb * (1 + s["nodisp_wB vs stock_wB"] / 100)
        e2m1, real, e0m3 = (ceil * (1 + s[f"wBdefault_{k} vs nodisp_wB"] / 100) for k in ("e2m1", "real", "e0m3"))
        rows[md] = {"tiles (t0 ceiling vs stock)": r(ceil_t0, ko),
                    "  of which placement and #4 (stock_wB vs stock_ko)": r(wb, ko),
                    "  of which the 1x8 arrangement (t0 ceiling vs stock_wB)": r(ceil_t0, wb),
                    "site-0 tags (tagged vs t0 ceiling)": r(ceil, ceil_t0),
                    "dispatch (all-E2M1 vs the kernel's ceiling)": r(e2m1, ceil),
                    "E0M3 tiles, real map (real vs all-E2M1)": r(real, e2m1), "all-E0M3 vs all-E2M1": r(e0m3, e2m1),
                    "total, real map vs stock": r(real, ko),
                    "total, real map vs stock_wB (the paper's same-placement reference)": r(real, wb)}
    out["8x64 before: the paper n8k64_wB (default dispatch) vs stock_ko; C2w (amendment 10)"] = rows
    out["8x64 after: mixed_wB_ko (build_V) vs stock_ko; C2U (amendment 17)"] = rows8
    out["8x64 just before the pipelined flags: build_P3freq mixed_wB_ko vs stock_ko; C2U"] = rows8a
    # 256x64: the paper set and A' against the paper stock (amendment 3's C2''), build_V against stock_ko (C2V)
    a1 = load("kernel_opt/A1/abA1.json")["c2"]["modes"]
    rows, rowsA = {}, {}
    for md in modes:
        m = {k: v["us"] for k, v in a1[md].items()}
        for pre, rows_ in (("default", rows), ("g32", rowsA)):
            rows_[md] = {"tiles + dispatch (all-E2M1 vs stock)": r(m[f"{pre}_e2m1"], m["stock_wA"]),
                         "E0M3 tiles, real map (real vs all-E2M1)": r(m[f"{pre}_real"], m[f"{pre}_e2m1"]),
                         "all-E0M3 vs all-E2M1": r(m[f"{pre}_e0m3"], m[f"{pre}_e2m1"]),
                         "total, real map vs stock": r(m[f"{pre}_real"], m["stock_wA"])}
        rows[md]["  of which tiles (the 16-arm kernel's tagged ceiling vs stock)"] = r(m["nodisp"], m["stock_wA"])
    out["256x64 before: the paper set on 256x64 maps (16-arm) vs the paper stock_wA; C2″ (amendment 3)"] = rows
    out["256x64 A′ (g32) vs the paper stock_wA; C2″ (amendment 3)"] = rowsA
    v = load("kernel_opt/V/V.json")["c2"]
    rows = {}
    for md in modes:
        s = v[md]
        c = 1 + s["ceil vs stock_ko"] / 100
        e2m1, real, e0m3 = (1 + s[f"C_{k} vs stock_ko"] / 100 for k in ("e2m1", "real", "e0m3"))
        rows[md] = {"tiles (t0 ceiling vs stock)": r(c, 1), "dispatch (all-E2M1 vs the kernel's ceiling)": r(e2m1, c),
                    "E0M3 tiles, real map (real vs all-E2M1)": r(real, e2m1), "all-E0M3 vs all-E2M1": r(e0m3, e2m1),
                    "total, real map vs stock": r(real, 1)}
    out["256x64 after: mixed256_ko (build_V) vs stock_ko; C2V (amendment 18)"] = rows
    return out


def c3v_tables():
    s = load("kernel_opt/c3v/C3v_summary.json")
    out = {}
    for unit in ("16x64", "8x64", "256x64"):
        d = s[unit]
        # the adopted path against the paper kernel in absolute time, per cell (f = 0 and the real map); the paper
        # kernel is measured against the paper stock and the adopted path against stock_ko, in the same session
        abs_f0, abs_real = [], []
        for cell, x in d["decomposition"].items():
            ko = 1 + x["stock_ko_vs_stock"] / 100
            abs_f0.append(r((1 + x["adopted"] / 100) * ko, 1 + x["paper"] / 100))
            y = d["real"][cell]
            abs_real.append(r((1 + y["adopted"] / 100) * ko, 1 + y["paper"] / 100))
        out[unit] = dict(per_t=d["summary"],
                         adopted_vs_paper_abs=dict(f0_median=statistics.median(abs_f0), f0_min=min(abs_f0),
                                                   f0_max=max(abs_f0), real_median=statistics.median(abs_real)))
    return out


def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(head))) + "|"]
                     + ["| " + " | ".join(x) + " |" for x in rows])


def pct(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", "−") + " %"


def write_tables(rec):
    L = ["# Paper summary tables", "",
         "Generated by `experiments/kernel_opt/paper_summary.py` from the committed records; do not edit by hand.", "",
         "Per-forward GEMM sums (deviation-2 method), overheads in %. A T row is the median over the four models; "
         "'all T' is the median over the 48 (model, T) cells (28 for the paper's own run, T ≥ 128); a band is the median "
         "over its cells.", ""]
    L += ["## 0. Headline: medians over the (model, T) cells, T ≥ 128 (the paper's range, 28 cells) / all T (48 cells)", ""]
    rows = []
    for unit in ("16x64", "8x64", "256x64"):
        for tag in TAGS:
            b = rec["before"][f"{unit} {tag}"]
            m1 = b["m1_opt1" if unit == "8x64" else "m1_2"]
            pk = rec["paper_vs_stock_ko"][f"{unit} {tag}" + ("" if unit == "16x64" else " (SPLICE)")]
            a, c = rec["after"][f"{unit} {tag}"], rec["chains"][f"{unit} {tag}"]
            rows.append([unit, tag, pct(b["paper_run"]["median_t128"]), f"{pct(m1['median_t128'])} / {pct(m1['median'])}",
                         f"{pct(a['median_t128'])} / {pct(a['median'])}",
                         f"{pct(pk['median_t128'])} / {pct(pk['median'])}" + ("" if unit == "16x64" else " (splice)"),
                         f"{pct(c['median_t128'])} / {pct(c['median'])}"])
    L += [md_table(["unit", "tags", "before: paper run (T ≥ 128)", "before: M1", "after: build_V vs stock_ko",
                    "paper kernel vs stock_ko", "build_V vs paper kernel, absolute time (chain)"], rows), "",
          "Before: against the paper stock_wA (16x64, 256x64) or the paper stock_wB (8x64, the paper's same-placement "
          "reference). After and 'paper kernel vs stock_ko': against stock_ko.", ""]
    L += ["## 1. Before (the paper kernels vs the paper stock) and after (build_V vs stock_ko)", ""]
    for unit in ("16x64", "8x64", "256x64"):
        cols = []
        for tag in TAGS:
            b = rec["before"][f"{unit} {tag}"]
            cols.append((f"before {tag}, paper run", b["paper_run"]))
            cols.append((f"before {tag}, M1 ({'opt-1' if unit == '8x64' else '#2'} session)", b["m1_opt1" if unit == "8x64" else "m1_2"]))
            cols.append((f"after {tag}", rec["after"][f"{unit} {tag}"]))
        rows = [[str(t)] + [pct(c[1]["per_t_median"][t]) if t in c[1]["per_t_median"] else "—" for c in cols] for t in T12]
        rows.append(["**all T**"] + [f"**{pct(c[1]['median'])}**" for c in cols])
        ref = "the paper stock_wB (same placement)" if unit == "8x64" else "the paper stock_wA"
        L += [f"### {unit}: before against {ref}, after against stock_ko", "", md_table(["T"] + [c[0] for c in cols], rows), ""]
        if unit == "8x64":
            rows2 = []
            for t in T12:
                row = [str(t)]
                for tag in TAGS:
                    b = rec["before"][f"8x64 {tag}"]
                    x = b["paper_run_wA"]["per_t_median"].get(t)
                    row.append(f"{pct(x) if x is not None else '—'} / {pct(b['m1_opt1_wA']['per_t_median'][t])}")
                rows2.append(row)
            L += ["The 8x64 paper kernel against the paper stock with the weights on A (paper run / M1):", "",
                  md_table(["T"] + [f"{tag}" for tag in TAGS], rows2), ""]
    L += ["## 2. Per model (typical / worst)", ""]
    for unit in ("16x64", "8x64", "256x64"):
        for label, key in ((f"before (M1, {'opt-1 session, vs the paper stock_wB' if unit == '8x64' else '#2 session, vs the paper stock_wA'})", "b"),
                           ("after (build_V vs stock_ko)", "a")):
            def get(tag):
                if key == "a":
                    return rec["after"][f"{unit} {tag}"]
                return rec["before"][f"{unit} {tag}"]["m1_opt1" if unit == "8x64" else "m1_2"]
            rows = [[str(t)] + [f"{pct(get('typical')['per_model'][m][t], 1)} / {pct(get('worst')['per_model'][m][t], 1)}"
                                for m in MODELS] for t in T12]
            L += [f"### {unit}, {label}", "", md_table(["T"] + [NAMES[m] for m in MODELS], rows), ""]
    L += ["## 3. The paper kernels against stock_ko", ""]
    P = rec["paper_vs_stock_ko"]
    rows = [[str(t)] + [pct(v["per_t_median"][t]) for v in P.values()] for t in T12]
    rows.append(["**all T**"] + [f"**{pct(v['median'])}**" for v in P.values()])
    L += [md_table(["T"] + list(P), rows), "",
          "16x64 and the stock shift: amendment 9, one session. 8x64 and 256x64 are SPLICES: no session measured those "
          "paper kernels and stock_ko together, so the paper kernel's overhead against the paper stock_wA (its own session) "
          "is multiplied by the paper stock against stock_ko (amendment 9).", ""]
    L += ["## 4. Each optimization's paired effect (after vs before, one session)", ""]
    rows = [[s["unit"], s["name"], s["amendment"], f"`{s['file']}` ({s['session']}; {s['commit']})", s["tag"] or "—",
             pct(s["median"]), *[pct(s["bands"][b]) for b in BANDS], f"{s['below_zero']}/{s['cells']}", s["status"]]
            for s in rec["steps"]]
    L += [md_table(["unit", "optimization", "amendment", "record (session UTC; results commit)", "tags", "all T", *BANDS,
                    "cells < 0", "status"], rows), ""]
    L += ["## 5. The paper kernel → the build_V kernel in absolute time (CHAIN of paired links)", ""]
    rows = [[str(t)] + [pct(v["per_t_median"][t]) for v in rec["chains"].values()] for t in T12]
    rows.append(["**all T**"] + [f"**{pct(v['median'])}**" for v in rec["chains"].values()])
    L += [md_table(["T"] + list(rec["chains"]), rows), ""]
    L += [f"- {k}: " + " × ".join(f"`{x}`" for x in v["links"]) for k, v in rec["chains"].items()] + [""]
    L += ["## 6. 4096³ decompositions (%; b2b / isolated / sustained)", "",
          "Each block is one session. The real map is Llama-3.1-8B layer 0 o_proj of the unit's map. In the sustained mode "
          "the card sits at its 500 W cap.", ""]
    for name, rows_ in rec["c2"].items():
        keys = list(next(iter(rows_.values())).keys())
        rows = [[k] + [pct(rows_[md][k]) for md in ("b2b", "isolated", "sustained")] for k in keys]
        L += [f"### {name}", "", md_table(["part", "b2b", "isolated", "sustained"], rows), ""]
    L += ["## 7. C3v (amendment 19, one session): overhead against stock_ko vs the E0M3 share", "",
          "Mean of the 3 shapes (4096x4096, 14336x4096, 4096x14336), random placement; tiles = the adopted path's ceiling vs "
          "stock_ko, dispatch = the adopted path at f = 0 vs its ceiling.", ""]
    for unit, d in rec["c3v"].items():
        rows = [[t, pct(x["adopted_f0"], 1), pct(x["adopted_real"], 1), pct(x["adopted_f100"], 1), pct(x["tiles"], 1),
                 pct(x["dispatch"], 1), f"{x['real_share']:.1f} %"] for t, x in d["per_t"].items()]
        a = d["adopted_vs_paper_abs"]
        L += [f"### {unit}", "", md_table(["T", "f = 0", "real map", "f = 100 %", "tiles", "dispatch", "real-map share"], rows), "",
              f"The adopted path against C3v's paper kernel, absolute time (same session, 18 cells): f = 0 median "
              f"{pct(a['f0_median'], 1)} ({pct(a['f0_min'], 1)} … {pct(a['f0_max'], 1)}); real maps median {pct(a['real_median'], 1)}.", ""]
    (OUT / "tables.md").write_text("\n".join(L) + "\n")


def figure(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, unit in zip(axes, ("16x64", "8x64", "256x64")):
        for tag, ls in (("typical", "-"), ("worst", "--")):
            b = rec["before"][f"{unit} {tag}"]
            m1 = b["m1_opt1" if unit == "8x64" else "m1_2"]
            ax.plot(T12, [m1["per_t_median"][t] for t in T12], ls, color="tab:red", marker="o", ms=3,
                    label=f"paper kernel vs paper stock{'_wB' if unit == '8x64' else ''} (M1), {tag}")
            ax.plot(T7, [b["paper_run"]["per_t_median"][t] for t in T7], "x", color="tab:red", ms=6,
                    label="the paper's own run" if tag == "typical" else None)
            a = rec["after"][f"{unit} {tag}"]
            ax.plot(T12, [a["per_t_median"][t] for t in T12], ls, color="tab:blue", marker="o", ms=3,
                    label=f"build_V vs stock_ko, {tag}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(T12)
        ax.set_xticklabels([str(t) for t in T12], fontsize=7)
        ax.set_title(f"{unit}: per-forward GEMM overhead, median of 4 models")
        ax.set_xlabel("T (tokens)")
        ax.set_ylabel("% against the matching stock")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "overhead.png", dpi=110)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = kernel_tables()
    rec["c2"] = c2_tables()
    rec["c3v"] = c3v_tables()
    rec["sources"] = {k: {kk: vv for kk, vv in v.items() if kk != "get"} for k, v in SOURCES.items()}
    (OUT / "summary.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    write_tables(rec)
    figure(rec)


if __name__ == "__main__":
    main()

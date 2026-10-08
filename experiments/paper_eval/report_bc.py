"""B (memory) and C (ownership; native vs simulated) reports (PROTOCOL.md): results/paper_eval/{memory,verify}/.

    python report_bc.py memory|verify
"""
import json
import math
import sys
from pathlib import Path

RUN = Path("/home/dev/n16k64_campaign/paper_eval")
REL = Path("/home/dev/flipquant_release")
OUT = Path(__file__).resolve().parents[2] / "results" / "paper_eval"
MODELS = [("qwen3-1.7b", "Qwen3-1.7B"), ("qwen3-8b", "Qwen3-8B"), ("mistral-7b", "Mistral-7B-Instruct-v0.3"),
          ("nemotron-nano-9b-v2", "Nemotron-Nano-9B-v2"), ("phi4-14b", "Phi-4"), ("qwen3.8-27b", "Qwen3.8-27B")]
UNITS = ("8x64", "16x64", "256x64")
HYBRID = ("nemotron-nano-9b-v2", "qwen3.8-27b")
G = 2 ** 30


def memory():
    pols = ["bf16", "nvfp4", "fo6", "fq-8x64", "fq-16x64", "fq-256x64"]
    rec, L = {}, ["# B: memory on the RTX PRO 6000 (mem_probe.py, flipquant's loader, build_V)", "",
                  "Weights: every parameter and buffer the model holds on the GPU after the load and install. Quantized "
                  "linears: packed FP4 + placed UE4M3 scales + BF16 bias, against the same linears' BF16 size. After "
                  "load: torch.cuda.memory_allocated. Peak: torch.cuda.max_memory_allocated during one 1x2048 prefill "
                  "(an eager forward that writes the KV cache and returns the logits).", "",
                  "| model | policy | weights | quantized linears / BF16 | whole model / BF16 | after load | 1x2048 peak |",
                  "|---|---|---:|---:|---:|---:|---:|"]
    for m, title in MODELS:
        for p in pols:
            f = RUN / "memory" / m / f"{p}.json"
            if f.exists():
                rec.setdefault(m, {})[p] = json.loads(f.read_text())
        if "bf16" not in rec.get(m, {}):
            continue
        bf = rec[m]["bf16"]["model_bytes"]
        for p, r in rec[m].items():
            q = r["quantized_linears"]
            L.append(f"| {title} | {p} | {r['model_bytes'] / G:.2f} GiB | {q['installed_over_bf16']:.4f} | "
                     f"{r['model_bytes'] / bf:.3f} | {r['allocated_after_load'] / G:.2f} GiB | "
                     f"{r['peak_prefill_1x2048'] / G:.2f} GiB |")
    (OUT / "memory").mkdir(parents=True, exist_ok=True)
    (OUT / "memory" / "memory.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    (OUT / "memory" / "MEMORY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def paired(x, y):
    d = [a - b for a, b in zip(x, y)]
    n, mu = len(d), sum(d) / len(d)
    se = math.sqrt(sum((v - mu) ** 2 for v in d) / (n - 1) / n)
    return dict(delta=mu, two_se=2 * se, n=n, agree=abs(mu) <= 2 * se)


def verify():
    rec = dict(ownership={}, native_vs_fake={})
    L = ["# C: verification (Section 4.3)", "", "## C1: ownership check of the 18 release artifacts", "",
         "Every weight element decoded by the kernel itself (an identity activation through the GEMM) must equal its "
         "stored value under the map's format (flipquant `--ownership-check`; the install raises on any mismatch).", "",
         "| model | unit | modules | weight elements checked | informative elements | E0M3 tiles | E0M3 elements observed | exact | format mismatches |",
         "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    sys.path.insert(0, "/home/dev/n16k64_campaign/fqopt/wt")
    from flipquant import maps as M
    for m, title in MODELS:
        for u in UNITS:
            f = RUN / "ownership" / f"{m}_{u}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            own = r["native"]["ownership"]
            tiles, unit, _ = M.load(REL / m / f"flipquant_{u}.pt")
            rows, cols = M.parse_unit(unit)
            elements = sum(t.numel() for t in tiles.values()) * rows * cols
            o = dict(modules=len(own), elements=elements, informative=sum(v["informative_elements"] for v in own.values()),
                     e0m3_tiles=sum(v["e0m3_tiles"] for v in own.values()),
                     e0m3_elements=sum(v["e0m3_elements_observed"] for v in own.values()),
                     exact=all(v["exact"] for v in own.values()),
                     mismatches=sum(v["format_mismatches"] for v in own.values()),
                     map_modules=len(tiles))
            rec["ownership"].setdefault(m, {})[u] = o
            L.append(f"| {title} | {u} | {o['modules']} of {o['map_modules']} | {o['elements']:,} | {o['informative']:,} | "
                     f"{o['e0m3_tiles']:,} | {o['e0m3_elements']:,} | {o['exact']} | {o['mismatches']} |")
    L += ["", "## C2: native vs simulated (fake), per-window NLL, paper convention", "",
          "Paired ΔNLL native − simulated (nats/token, ± 2 SE) on the same windows; a cell agrees when |Δ| ≤ 2 SE. Both "
          "sides in one env: n16k64 (the release PPL records) for the four non-hybrid models, n16k64-fast (part F's "
          "records) for Nemotron-Nano-9B-v2.", "",
          "| model | unit | WikiText-2 | C4 |", "|---|---|---:|---:|"]
    agree = cells = 0
    for m, title in MODELS[:5] if True else MODELS:
        for u in UNITS:
            f = RUN / "fakeppl" / f"{m}_{u}.json"
            if not f.exists():
                continue
            fr = json.loads(f.read_text())
            if "resources" not in fr:                      # still running (rewritten after every corpus)
                continue
            fake = fr["results"]
            # the native reference from the same env as the simulated run: n16k64-fast (part F) for the hybrid model
            natf = RUN / "pplfast" / m / f"fq-{u}.json" if m in HYBRID else REL / m / "ppl" / f"flipquant_{u}.json"
            if not natf.exists():
                continue
            nat = json.loads(natf.read_text())["results"]
            row = {}
            for c in ("wiki", "c4"):
                row[c] = paired(nat[c]["nll"], fake[c]["nll"])
                agree += row[c]["agree"]
                cells += 1
            rec["native_vs_fake"].setdefault(m, {})[u] = row
            L.append(f"| {title} | {u} | " + " | ".join(
                f"{row[c]['delta']:+.5f} ± {row[c]['two_se']:.5f}{'' if row[c]['agree'] else ' ✗'}" for c in ("wiki", "c4")) + " |")
    L += ["", f"**{agree} of {cells} cells agree within 2 SE.**"]
    rec["agree"], rec["cells"] = agree, cells
    # amendment 9: the registered reference (the release records, fallback env) for the hybrid model, as a secondary
    # comparison; it mixes the env change (fast vs fallback kernels) into the native - simulated difference
    xs = []
    for m, title in MODELS:
        if m not in HYBRID:
            continue
        for u in UNITS:
            f = RUN / "fakeppl" / f"{m}_{u}.json"
            if not f.exists():
                continue
            fr = json.loads(f.read_text())
            if "resources" not in fr:
                continue
            fake = fr["results"]
            nat = json.loads((REL / m / "ppl" / f"flipquant_{u}.json").read_text())["results"]
            row = {c: paired(nat[c]["nll"], fake[c]["nll"]) for c in ("wiki", "c4")}
            rec.setdefault("native_fallback_vs_fake_fast", {}).setdefault(m, {})[u] = row
            xs.append(f"| {title} | {u} | " + " | ".join(
                f"{row[c]['delta']:+.5f} ± {row[c]['two_se']:.5f}{'' if row[c]['agree'] else ' ✗'}" for c in ("wiki", "c4")) + " |")
    if xs:
        L += ["", "As registered (amendment 9): the release records' native NLL (fallback env) − simulated (n16k64-fast); "
              "this includes the env change:", "", "| model | unit | WikiText-2 | C4 |", "|---|---|---:|---:|"] + xs
    (OUT / "verify").mkdir(parents=True, exist_ok=True)
    (OUT / "verify" / "verify.json").write_text(json.dumps(rec, indent=1) + "\n")
    (OUT / "verify" / "VERIFY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"memory": memory, "verify": verify}[sys.argv[1]]()

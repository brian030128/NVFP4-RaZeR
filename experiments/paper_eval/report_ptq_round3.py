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


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "shares":
        shares()
    else:
        sys.exit(__doc__)

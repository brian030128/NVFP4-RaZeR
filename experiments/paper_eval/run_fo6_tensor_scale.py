"""Part R (PROTOCOL.md): FourOverSix's tensor-scale constant, 6 x 448 (ours) vs 6 x 256 (the original FourOverSix),
with flipquant paper-sm120-runs' opt-in --fo6-tensor-qmax (d92c032).

    python run_fo6_tensor_scale.py check-default        # short GPU job: the new fourover6 at its default vs aa12408's
    python run_fo6_tensor_scale.py ppl [--models M,..]   # GPU: FO6-448 (same code, default) and FO6-256, 8 models
    python run_fo6_tensor_scale.py stats [--models M,..] # CPU: block statistics of every quantized weight
    python run_fo6_tensor_scale.py report               # CPU: results/paper_eval/fo6_tensor_scale/

Records under RUN/fo6_tensor_scale/: check_default.json, <model>/{fo6_448,fo6_256}.json (evaluation.ppl), and
<model>/stats.json. The FO6-448 rerun of a model whose table FO6 record was made by evaluation.ppl (Nemotron-Nano-9B-v2,
Qwen3.8-27B, the two Llama-Instruct models) must reproduce that record per window, bit for bit; the run stops if not.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu as G  # noqa: E402
import run_llama_instruct as P  # noqa: E402
import run_sm120 as S  # noqa: E402
import tables_cpu as T  # noqa: E402
import tables_final as F  # noqa: E402

WORK = G.RUN / "fo6_tensor_scale"
OUT = T.ROOT / "results" / "paper_eval" / "fo6_tensor_scale"
PRE = "aa12408"                     # paper-sm120-runs before the option
MODELS = [k for k, _, _ in F.FINAL_MODELS]
# the models whose table FO6 record was made by flipquant's evaluation.ppl (parts F and P); the others' come from the
# earlier sm120 evaluation harness (main-ppl / paper)
SAME_HARNESS = ("nemotron-nano-9b-v2", "qwen3.8-27b", "llama3.2-1b-instruct", "llama3.1-8b-instruct")
FO6 = [*G.NATIVE, "--kernel-set", "auto", "--weight", "fourover6", "--act", "fourover6", "--paper-convention"]
LABELS = {"fo6_448": [], "fo6_256": ["--fo6-tensor-qmax", "256"]}


def rev(model):
    return P.REV[model] if model in P.REV else S.REV[model]


def table_record(model, row):
    return Path(json.loads((F.OUT / "final.json").read_text())["sources"][model][row])


def per_window(a, b):
    """Per-window NLL of two records: bitwise equal per corpus, and the largest difference."""
    x, y = T.nll_of(a), T.nll_of(b)
    out = {}
    for c, _ in T.CORPORA:
        same_n = len(x[c]) == len(y[c])
        out[c] = dict(windows=(len(x[c]), len(y[c])), bitwise=same_n and x[c] == y[c],
                      max_abs_diff=max((abs(p - q) for p, q in zip(x[c], y[c])), default=None),
                      differing_windows=sum(p != q for p, q in zip(x[c], y[c])))
    out["bitwise"] = all(out[c]["bitwise"] for c, _ in T.CORPORA)
    return out


# ------------------------------------------------------------------------------------------------- check-default
def check_default():
    """fourover6 at its default (and with tensor_qmax=448) vs aa12408's formats.py, bit for bit: every scope, CPU and
    GPU, random heavy-tailed tensors and real weights (Llama-3.2-1B-Instruct, Qwen3-1.7B)."""
    import torch
    from safetensors import safe_open
    sys.path.insert(0, str(S.FQ2))
    from flipquant import formats as new
    WORK.mkdir(parents=True, exist_ok=True)
    src = subprocess.run(["git", "-C", str(S.FQ2), "show", f"{PRE}:flipquant/formats.py"], capture_output=True,
                         text=True, check=True).stdout
    path = WORK / f"formats_{PRE}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"formats_{PRE}", path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old                   # its dataclass looks its module up
    spec.loader.exec_module(old)
    G.idle()
    G.log("START fo6ts_check_default")

    def heavy(shape, seed):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(shape, generator=g) * torch.exp(1.5 * torch.randn(shape, generator=g)).clamp(max=50.0)
                ).to(torch.bfloat16)

    tensors = {f"random{i}": heavy(s, i) for i, s in enumerate([(64, 256), (3, 40, 512), (128, 2048), (7, 4096)])}
    hub = Path("/home/dev/.cache/huggingface/hub")
    for model, repo, keys in (("llama3.2-1b-instruct", "meta-llama/Llama-3.2-1B-Instruct",
                               ("model.layers.0.self_attn.q_proj.weight", "model.layers.15.mlp.down_proj.weight")),
                              ("qwen3-1.7b", "Qwen/Qwen3-1.7B",
                               ("model.layers.0.mlp.gate_proj.weight", "model.layers.27.self_attn.o_proj.weight"))):
        snap = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / rev(model)
        for f in sorted(snap.glob("*.safetensors")):
            with safe_open(str(f), "pt") as h:
                for k in keys:
                    if k in h.keys():
                        tensors[f"{model}:{k}"] = h.get_tensor(k)
    assert sum(":" in k for k in tensors) == 4, sorted(tensors)
    rows, ok = [], True
    for dev in ("cpu", "cuda"):
        for name, x in tensors.items():
            x = x.to(dev)
            for scope in ("tensor", "row", "document", "static"):
                g = (x.float().abs().amax() / 2000.0) if scope == "static" else None
                a = old.fourover6(x, scope, g)
                for label, b in (("default", new.fourover6(x, scope, g)),
                                 ("tensor_qmax=448", new.fourover6(x, scope, g, tensor_qmax=448.0))):
                    same = (torch.equal(a.codes, b.codes) and torch.equal(a.scale.view(torch.int32), b.scale.view(torch.int32))
                            and torch.equal(a.gscale.float().view(torch.int32), b.gscale.float().view(torch.int32)))
                    rows.append(dict(device=dev, tensor=name, shape=list(x.shape), scope=scope, new=label, bitwise=same))
                    ok &= same
    rec = dict(reference=f"flipquant {PRE}:flipquant/formats.py", new=subprocess.run(
        ["git", "-C", str(S.FQ2), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        cases=len(rows), all_bitwise=ok, rows=rows)
    (WORK / "check_default.json").write_text(json.dumps(rec, indent=1) + "\n")
    G.log(f"END fo6ts_check_default all_bitwise={ok} cases={len(rows)}")
    print(f"check-default: {len(rows)} cases, all bitwise: {ok}")
    if not ok:
        G.stop("part R: the new fourover6 default differs from aa12408's")


# ------------------------------------------------------------------------------------------------------------ ppl
def ppl(models):
    for model in models:
        for label, extra in LABELS.items():
            out = WORK / model / f"{label}.json"
            cmd = [G.py(model), "-m", "evaluation.ppl", "--model", model, "--revision", rev(model), *FO6, *extra,
                   "--out", out]
            rc = G.run(f"fo6ts_{model}_{label}", cmd, out, S.FQ2)
            if rc not in (None, 0):
                logf = (G.RUN / "logs" / f"fo6ts_{model}_{label}.log").read_text(errors="replace")
                oom = "OutOfMemoryError" in logf or "CUDA out of memory" in logf
                G.stop(f"part R: {model} {label} rc={rc}" + (" (out of GPU memory)" if oom else ""))
            if label == "fo6_448" and model in SAME_HARNESS:
                cmp = per_window(out, table_record(model, "fo6"))
                (WORK / model / "default_check.json").write_text(json.dumps(cmp, indent=1) + "\n")
                if not cmp["bitwise"]:
                    G.stop(f"part R: {model}: FO6-448 does not reproduce the table's FO6 record bit for bit: {cmp}")


# ---------------------------------------------------------------------------------------------------------- stats
def decide(w, q, Fm):
    """formats.fourover6's computation (tensor scope) with its per-block decision exposed:
    (codes, scale, use4, gs, peak); checked against formats.fourover6 by the caller."""
    import torch
    b, gs = Fm._grouped(w, "tensor", None, Fm.global_qmax(q))
    ws = b / gs
    peak = ws.abs().amax(-1, keepdim=True)
    table = Fm.lut(Fm.FMT_E2M1, w.device)
    cand = []
    for qm in (6.0, 4.0):
        scale = Fm._e4m3(peak / qm)
        y = ws / scale
        codes = Fm._round_e2m1(y, "down") | Fm._sign(y)
        cand.append((codes, scale, (table[codes] * scale - ws).square().sum(-1, keepdim=True)))
    use4 = cand[1][2] < cand[0][2]
    return (torch.where(use4, cand[1][0], cand[0][0]), torch.where(use4, cand[1][1], cand[0][1]), use4, gs, peak)


def block_stats(w, Fm):
    import torch
    k = w.shape[-1]
    bm = w.float().reshape(-1, k // 16, 16).abs().amax(-1)
    amax = bm.max()
    res, dq = dict(blocks=bm.numel(), zero_blocks=int((bm == 0).sum()),
                   above_two_thirds=int((bm > (2.0 / 3.0) * amax).sum())), {}
    for q in (448.0, 256.0):
        codes, scale, use4, gs, peak = decide(w, q, Fm)
        ref = Fm.fourover6(w, tensor_qmax=q)
        if not (torch.equal(codes.reshape(w.shape), ref.codes.long().reshape(w.shape)) and
                torch.equal(scale.squeeze(-1).reshape(ref.scale.shape), ref.scale)):
            raise RuntimeError("the decision replica differs from formats.fourover6")
        t = int(q)
        res[f"use4_{t}"] = int(use4.sum())
        res[f"clamped4_{t}"] = int((peak / 4.0 > Fm.E4M3_MAX).sum())
        res[f"subnormal_scale_{t}"] = int((scale < 2.0 ** -6).sum())
        res[f"floor_scale_{t}"] = int((scale <= 2.0 ** -9).sum() - (bm == 0).sum())
        res[f"_use4_{t}"] = use4.reshape(-1)
        dq[t] = ref.dequantize(torch.float32).reshape(-1, k // 16, 16)
    res["choice_flips"] = int((res.pop("_use4_448") != res.pop("_use4_256")).sum())
    res["blocks_changed"] = int((dq[448] != dq[256]).any(-1).sum())
    return res


def kind(name):
    return name.rsplit(".", 1)[-1]


def stats(models):
    import torch
    sys.path.insert(0, str(S.FQ2))
    from flipquant import formats as Fm
    from models.loader import load, quantizable_linears
    torch.set_num_threads(48)              # the GPU jobs keep the rest of the 128 cores
    for model in models:
        out = WORK / model / "stats.json"
        if out.exists():
            continue
        t0 = time.time()
        G.log(f"START fo6ts_stats_{model} (CPU)")
        m, _, spec = load(model, device_map="cpu", revision=rev(model))
        lin = quantizable_linears(m, spec)
        per, ks = {}, {}
        with torch.no_grad():
            for name, layer in lin.items():
                per[name] = block_stats(layer.weight.data, Fm)
                ks.setdefault(layer.in_features, 0)
                ks[layer.in_features] += 1
        tot = {key: sum(v[key] for v in per.values()) for key in next(iter(per.values()))}
        by_kind = {}
        for name, v in per.items():
            d = by_kind.setdefault(kind(name), {key: 0 for key in v})
            for key in v:
                d[key] += v[key]
        rec = dict(model=model, revision=rev(model), modules=len(per), totals=tot,
                   shares={key: tot[key] / tot["blocks"] for key in tot if key != "blocks"},
                   by_kind={kk: dict(v, shares={key: v[key] / v["blocks"] for key in v if key != "blocks"})
                            for kk, v in by_kind.items()},
                   activations={str(k): dict(modules=n, at_least_one_block_per_token=16 / k) for k, n in sorted(ks.items())},
                   per_module=per, seconds=time.time() - t0,
                   note="CPU values of formats.fourover6 (tensor scope); near-tie blocks can differ on the GPU")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1) + "\n")
        G.log(f"END fo6ts_stats_{model} {time.time() - t0:.0f}s")
        print(model, {k: round(v, 5) for k, v in rec["shares"].items()}, flush=True)
        del m, lin


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check-default", "ppl", "stats", "report"))
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if m]
    assert set(models) <= set(MODELS), models
    G.log(f"fo6_tensor_scale (part R) {a.what} models {models}")
    if a.what == "check-default":
        check_default()
    elif a.what == "ppl":
        ppl(models)
    elif a.what == "stats":
        stats(models)
    else:
        import report_fo6_tensor_scale
        report_fo6_tensor_scale.main()

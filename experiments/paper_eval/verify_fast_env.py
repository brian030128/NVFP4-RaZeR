"""Amendment 2's env check: does transformers take the fast kernels for Nemotron-Nano-9B-v2 (Mamba2: mamba_ssm,
causal_conv1d) and Qwen3.8-27B (gated delta rule: fla, causal_conv1d)?

Run in an env: it records which implementation each wrapped kernel function resolved to (the decorator's closure:
the package's function, or the torch fallback), then loads the model in BF16 with flipquant's loader, runs one
1x2048 prefill under torch.profiler, and records the CUDA kernels' names that identify each path, plus the time of a
second (warm) 1x2048 prefill with CUDA events.

    python verify_fast_env.py MODEL OUT.json
"""
import inspect
import json
import sys
import time
from pathlib import Path

FQ = Path("/home/dev/n16k64_campaign/fqopt/wt")
sys.path.insert(0, str(FQ))

import torch  # noqa: E402

FUNCS = {"nemotron-nano-9b-v2": ("transformers.models.nemotron_h.modeling_nemotron_h",
                                 ["causal_conv1d_fn", "causal_conv1d_update", "mamba_split_conv1d_scan_combined",
                                  "selective_state_update", "mamba_chunk_scan_combined"]),
         "qwen3.8-27b": ("transformers.models.qwen3_5.modeling_qwen3_5",
                         ["causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                          "fused_recurrent_gated_delta_rule"])}
MARKERS = ("chunk_scan", "chunk_state", "state_passing", "bmm_chunk", "selective_state", "causal_conv1d", "conv1d",
           "gated_delta", "chunk_fwd", "fused_recurrent", "segsum", "cumsum")


def resolved(module_name, names):
    import importlib
    mod = importlib.import_module(module_name)
    out = {}
    for n in names:
        f = getattr(mod, n, None)
        if f is None:
            out[n] = "missing"
            continue
        impl = inspect.getclosurevars(f).nonlocals.get("implementation")
        out[n] = f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__qualname__', getattr(impl, '__name__', '?'))}"
    return out


def main(model_key, out):
    from models.loader import load
    rec = dict(model=model_key, python=sys.executable, torch=torch.__version__)
    for pkg in ("mamba_ssm", "causal_conv1d", "fla", "transformers", "triton"):
        try:
            m = __import__(pkg)
            rec[f"{pkg}_version"] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            rec[f"{pkg}_version"] = f"not importable: {type(e).__name__}: {str(e)[:200]}"
    rec["resolved"] = resolved(*FUNCS[model_key])
    t0 = time.time()
    model, _, spec = load(model_key, device_map="cuda", attn="sdpa")
    rec["load_seconds"] = time.time() - t0
    ids = torch.randint(100, 20000, (1, 2048), device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    with torch.no_grad():
        model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            model(input_ids=ids, use_cache=True)
            torch.cuda.synchronize()
        names = {}
        for e in prof.key_averages():
            if any(m in e.key.lower() for m in MARKERS):
                names[e.key[:120]] = round(getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) / 1e3, 3)
        rec["path_kernels_ms"] = dict(sorted(names.items(), key=lambda kv: -kv[1])[:40])
        ts = []
        for _ in range(3):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            out_ = model(input_ids=ids, use_cache=True)
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
        rec["prefill_1x2048_ms"] = sorted(ts)[1]
        rec["logits_sample"] = out_.logits[0, -1, :8].float().tolist()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "path_kernels_ms"}, indent=1))
    print("kernels:", list(rec["path_kernels_ms"])[:12])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

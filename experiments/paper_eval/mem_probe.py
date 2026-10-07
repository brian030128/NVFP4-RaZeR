"""Part B: the memory of one (model, policy) on the sm120 path, with flipquant's own loader (evaluation.latency's
load_policy, the same as the prefill benchmark of part A).

Records, after the load and install:
- the bytes of every parameter and buffer the model holds on the GPU, split into the quantized linears (NativeLinear:
  packed FP4 nibbles, placed UE4M3 scales, BF16 bias) and the rest (embeddings, norms, LM head, unquantized layers);
- the quantized linears' BF16 size (out x in x 2 bytes, plus bias), so the ratio installed / BF16 per linear set;
  for the BF16 policy the same linears (models.loader.quantizable_linears) are measured as loaded;
- torch.cuda.memory_allocated / memory_reserved after the load;
- the peak allocation during one 1x2048 prefill (one eager forward writing the KV cache and returning the logits, the
  prefill benchmark's tokens), and its increase over the after-load allocation.

    python mem_probe.py --model M <flipquant model args> --out OUT.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

FQ = Path("/home/dev/n16k64_campaign/fqopt/wt")
sys.path.insert(0, str(FQ))

import torch  # noqa: E402

from evaluation import latency as L  # noqa: E402
from models import sm120 as NS  # noqa: E402
from models.cli import add_model_args  # noqa: E402
from models.loader import quantizable_linears  # noqa: E402


def unique_bytes(tensors):
    seen, total = set(), 0
    for t in tensors:
        if t is None or not t.is_cuda:
            continue
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key in seen:
            continue
        seen.add(key)
        total += t.untyped_storage().nbytes()
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap, act_scope="row")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    B, _ = L.bench()
    B.require_idle()
    t0 = time.time()
    model, spec, layers, info, seconds = L.load_policy(args)
    torch.cuda.synchronize()
    res = dict(model=args.model, label=args.label, mode=args.mode, weight=args.weight if args.mode != "bf16" else "bf16",
               policy=info, seconds=seconds, gpu=B.gpu_info(),
               allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"))
    res["allocated_after_load"] = torch.cuda.memory_allocated()
    res["reserved_after_load"] = torch.cuda.memory_reserved()
    native = NS.native_modules(model)
    if native:
        q_tensors = [t for m in native.values() for t in (m.packed, m.sf, m.bias_bf16)]
        q_bf16 = sum(m.out_features * m.in_features * 2 + (m.out_features * 2 if m.bias_bf16 is not None else 0)
                     for m in native.values())
        q_ids = {id(t) for t in q_tensors if t is not None}
        quantized = dict(modules=len(native), installed_bytes=unique_bytes(q_tensors), bf16_bytes=q_bf16,
                         packed_bytes=unique_bytes([m.packed for m in native.values()]),
                         scales_bytes=unique_bytes([m.sf for m in native.values()]),
                         bias_bytes=unique_bytes([m.bias_bf16 for m in native.values()]))
    else:                                      # BF16: the same linears as loaded
        lin = quantizable_linears(model, spec)
        q_tensors = [t for m in lin.values() for t in (m.weight, m.bias)]
        q_ids = {id(t) for t in q_tensors if t is not None}
        b = unique_bytes(q_tensors)
        quantized = dict(modules=len(lin), installed_bytes=b, bf16_bytes=b, packed_bytes=None, scales_bytes=None,
                         bias_bytes=unique_bytes([m.bias for m in lin.values()]))
    rest = [t for t in list(model.parameters()) + list(model.buffers()) if id(t) not in q_ids]
    quantized["installed_over_bf16"] = quantized["installed_bytes"] / quantized["bf16_bytes"]
    res["quantized_linears"] = quantized
    res["other_bytes"] = unique_bytes(rest)
    res["model_bytes"] = quantized["installed_bytes"] + res["other_bytes"]
    # one 1x2048 prefill
    torch.cuda.reset_peak_memory_stats()
    ids = L.tokens(1, 2048)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    res["peak_prefill_1x2048"] = torch.cuda.max_memory_allocated()
    res["peak_prefill_over_load"] = res["peak_prefill_1x2048"] - res["allocated_after_load"]
    res["logits_dtype"] = str(out.logits.dtype)
    del out
    if native:
        cov = NS.coverage(model)
        assert cov["native_called"] == cov["native"] == len(native), cov
        res["coverage"] = cov
    res["wall_seconds"] = time.time() - t0
    res["status"] = "complete"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    g = 2 ** 30
    print(f"{args.model} {args.label}: model {res['model_bytes'] / g:.2f} GiB (quantized linears "
          f"{quantized['installed_bytes'] / g:.2f} GiB = {quantized['installed_over_bf16']:.4f} x BF16), after load "
          f"{res['allocated_after_load'] / g:.2f} GiB, 1x2048 peak {res['peak_prefill_1x2048'] / g:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()

"""
    Perplexity sweep over quantization formats.

    Unlike `run_ppl.py`, which evaluates a single configuration per process, this script loads the
    model and the evaluation data ONCE and then re-quantizes the weights in place for every
    configuration in the sweep. That removes N-1 model loads and, more importantly, guarantees that
    every configuration sees the byte-identical evaluation set (relevant for C4, whose documents are
    sampled at random).

    Weights are restored from a pristine CPU copy before each configuration, so quantization never
    compounds across runs.

    Example:
        python run_ppl_sweep.py --model_name llama-2-7b --datasets wikitext --output res.json

    Shard a sweep across GPUs:
        CUDA_VISIBLE_DEVICES=0 python run_ppl_sweep.py ... --shard_id 0 --num_shards 3
"""

import argparse
import json
import os
import random
import time
import warnings

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

from quantize import QuantConfig, quant_weight, collect_importance
from utils import load_model_and_tokenizer, set_seed, model2path


# (label, w_dtype, w_type_block, a_dtype, a_type_block)
# The MixFP4 type blocks with K < 64 cannot be expressed by a single mxf4nvf4 MMA operand and are
# included as accuracy upper bounds only -- see CLAUDE.md.
SWEEP_W4A16 = [
    ("fp16",                  "fp16",             "1x16",   "fp16", "1x16"),
    ("mxfp4",                 "mxfp4",            "1x16",   "fp16", "1x16"),
    ("nvfp4",                 "nvfp4",            "1x16",   "fp16", "1x16"),
    ("nvfp4_4over6",          "nvfp4_4over6",     "1x16",   "fp16", "1x16"),
    ("nvif4",                 "nvif4",            "1x16",   "fp16", "1x16"),
    ("razer_e3m3",            "nvfp4_razer_e3m3", "1x16",   "fp16", "1x16"),
]

SWEEP_W4A4 = [
    ("fp16",                  "fp16",             "1x16",   "fp16",             "1x16"),
    ("mxfp4",                 "mxfp4",            "1x16",   "mxfp4",            "1x16"),
    ("nvfp4",                 "nvfp4",            "1x16",   "nvfp4",            "1x16"),
    ("nvfp4_4over6",          "nvfp4_4over6",     "1x16",   "nvfp4_4over6",     "1x16"),
    ("nvif4",                 "nvif4",            "1x16",   "nvif4",            "1x16"),
    ("razer",                 "nvfp4_razer_e3m3", "1x16",   "nvfp4_razer_e4m3", "1x16"),
]

# MixFP4 type-block shapes, and the selection metric used to choose 4-vs-6 per scale block and
# E2M1-vs-E0M3 per type block. The metric rides on the data type name so that result file names
# stay distinct without extra config plumbing.
TYPE_BLOCKS  = ["1x16", "8x64", "16x64", "32x64", "32x128"]
MIX_VARIANTS = ["mix_4_6", "mix_4_6_m2"]

# The A operand tile is 16 rows, the B operand tile is 8, so a weight block of 8x64 pairs with an
# activation block of 16x64. Everything else pairs with itself.
A_BLOCK_FOR = {"8x64": "16x64"}


def _mix_rows(quantize_activations: bool):
    rows = []
    for dtype in MIX_VARIANTS:
        for tb in TYPE_BLOCKS:
            # "mix_4_6_8x64" for MSE keeps the labels of the already-computed results, so a rerun
            # skips them instead of repeating ~8 GPU-minutes each
            label = f"{dtype}_{tb}"
            if quantize_activations:
                rows.append((label, dtype, tb, dtype, A_BLOCK_FOR.get(tb, tb)))
            else:
                rows.append((label, dtype, tb, "fp16", "1x16"))
    return rows


SWEEP_W4A16 += _mix_rows(quantize_activations=False)
SWEEP_W4A4  += _mix_rows(quantize_activations=True)

# Weights mixed, activations held at the plain 4/6 baseline. Every other W4A4 mix_4_6 row in this
# file quantizes BOTH operands with mix_4_6, which confounds the weight-side and the activation-side
# effect; this sweep isolates the weight side. The activation type block is unused by nvfp4_4over6.
SWEEP_W4A4_WMIX = [
    ("nvfp4_4over6",   "nvfp4_4over6", "1x16",  "nvfp4_4over6", "1x16"),
    ("mix_4_6_w16x64", "mix_4_6",      "16x64", "nvfp4_4over6", "1x16"),
    # Same weight type block, but E0M3 is elected only when the tile's mean gain exceeds 2 standard
    # errors of the per-scale-block gain -- the guard against an aggregate MSE win that harms
    # individual scale blocks. See _elect_e0m3.
    ("mix_4_6_m2_w16x64", "mix_4_6_m2", "16x64", "nvfp4_4over6", "1x16"),
    # Elect E0M3 only when NO scale block in the tile is harmed. This is the pointwise guarantee
    # that 1x16 has for free, so it cannot be worse than plain 4over6 on any block -- the invariant
    # that argmin gives up in exchange for an aggregate weight-MSE win.
    ("mix_4_6_dom_w16x64", "mix_4_6_dom", "16x64", "nvfp4_4over6", "1x16"),
    # HQQ-style robust selection loss, sum |e|^0.7 instead of sum e^2. Squared error is dominated by
    # the elements nearest the block maximum -- the outliers a 4-bit grid cannot place well in any
    # case -- so it elects whichever grid coddles them. p < 1 saturates on those and decides the
    # election on the bulk instead.
    ("mix_4_6_p0.7_w16x64", "mix_4_6_p0.7", "16x64", "nvfp4_4over6", "1x16"),
]

# Post-rounding-fix rerun. `nvfp4_4over6` and `mix_4_6_dom` are omitted deliberately: the baseline
# is untouched by the fix (llama-3.1-8b 6.8773/9.8177, qwen3-8b 10.0208/13.7555 on A6000), and
# dominance now provably equals it, since it elects E0M3 on 0.0% of 16x64 tiles and `elect="never"`
# is bit-identical to `quant_nvfp4_4over6` again.
SWEEP_W4A4_WMIX2 = [
    ("mix_4_6_w16x64",    "mix_4_6",    "16x64", "nvfp4_4over6", "1x16"),
    ("mix_4_6_m2_w16x64", "mix_4_6_m2", "16x64", "nvfp4_4over6", "1x16"),
    # bounded harm: elect only when no scale block is more than eps worse than its 4over6 loss.
    # Measured firing rates on llama-3.1-8b gate_proj at 16x64: b1 0.1%, b2 13.0%, b5 37.2%,
    # against argmin's 37.9% -- so b2/b3 are the points that are genuinely between the two rules.
    ("mix_4_6_b2_w16x64", "mix_4_6_b2", "16x64", "nvfp4_4over6", "1x16"),
    ("mix_4_6_b3_w16x64", "mix_4_6_b3", "16x64", "nvfp4_4over6", "1x16"),
    # HQQ-style robust loss. Pre-fix it was a large qwen win (-0.0320 wikitext) and a llama
    # non-event (+0.0016), from an intervention that changes the 4-vs-6 choice on ~20% of scale
    # blocks in BOTH models -- so the asymmetry is not explained by how often it fires.
    ("mix_4_6_p0.7_w16x64", "mix_4_6_p0.7", "16x64", "nvfp4_4over6", "1x16"),
]

# Calibrated TYPE election only. The 4-vs-6 scale search stays on plain MSE (verified bit-identical
# to the uncalibrated path), and the nvfp4_4over6 baseline needs no calibration at all -- so any win
# here is attributable to electing the element type better, not to a better scale search.
# Importance is E[x_j^2] from the wikitext TRAIN split, never the evaluation data.
SWEEP_W4A4_WMIX3 = [
    ("mix_4_6_hesst_w16x64",    "mix_4_6_hesst",    "16x64", "nvfp4_4over6", "1x16"),
    ("mix_4_6_hesst_m1_w16x64", "mix_4_6_hesst_m1", "16x64", "nvfp4_4over6", "1x16"),
]

SWEEPS = {"w4a16": SWEEP_W4A16, "w4a4": SWEEP_W4A4, "w4a4_wmix": SWEEP_W4A4_WMIX,
          "w4a4_wmix2": SWEEP_W4A4_WMIX2, "w4a4_wmix3": SWEEP_W4A4_WMIX3}


def build_calibration(tokenizer, seq_len, num_batches, seed=0):
    """
        Calibration sequences for the activation-importance estimate.

        Drawn from the wikitext TRAIN split, never from the evaluation data -- estimating the
        importance on the same tokens we then report perplexity on would leak the test set into the
        quantization decisions.
    """
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer("\n\n".join(train["text"][:20000]), return_tensors="pt").input_ids
    rng = random.Random(seed)
    max_start = max(ids.shape[1] - seq_len - 1, 0)
    return [ids[:, i:i + seq_len] for i in
            (rng.randint(0, max_start) for _ in range(num_batches))]


def build_wikitext(tokenizer, seq_len):
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    return testenc.input_ids


def build_c4(tokenizer, seq_len, num_samples=256, seed=0):
    """Mirrors the C4 sampling of run_ppl.py, but built once so every config sees the same data."""
    rng = random.Random(seed)
    testenc = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    valenc = []
    for _ in range(num_samples):
        while True:
            i = rng.randint(0, len(testenc) - 1)
            tmp = tokenizer(testenc[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] > (seq_len + 1):
                break
        i = rng.randint(0, tmp.input_ids.shape[1] - seq_len - 1)
        valenc.append(tmp.input_ids[:, i : i + seq_len])
    return torch.hstack(valenc)


@torch.no_grad()
def eval_ppl(model, input_ids, seq_len, desc=""):
    nsamples = input_ids.numel() // seq_len
    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    for i in tqdm(range(nsamples), desc=desc, leave=False):
        batch = input_ids[:, (i * seq_len) : ((i + 1) * seq_len)].to(model.device)
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, (i * seq_len) : ((i + 1) * seq_len)][:, 1:].to(shift_logits.device)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append((loss.float() * seq_len).item())
    return torch.exp(torch.tensor(nlls).sum() / (nsamples * seq_len)).item()


def snapshot_weights(model):
    """Pristine CPU copy of every weight that `quant_weight` would touch."""
    return {
        n: m.weight.data.detach().to("cpu", copy=True)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and ("head" not in n)
    }


def restore_weights(model, snapshot):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and ("head" not in n):
            m.weight.data.copy_(snapshot[n].to(m.weight.device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--datasets", type=lambda s: s.split(","), default=["wikitext"])
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--sweep", type=str, default="w4a16", choices=list(SWEEPS.keys()))
    parser.add_argument("--groupsize", type=int, default=16)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--w_outlier", type=float, default=8.0)
    parser.add_argument("--calib_batches", type=int, default=4,
                        help="Calibration sequences used to estimate activation importance.")
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="Cap the number of evaluated windows (smoke tests).")
    args = parser.parse_args()

    set_seed(0)
    configs = SWEEPS[args.sweep]
    configs = [c for i, c in enumerate(configs) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(configs)} configs: "
          f"{[c[0] for c in configs]}", flush=True)

    # The model modules keep a reference to this object, so mutating it between configurations
    # changes the activation quantization without reloading the model.
    quant_config = QuantConfig(
        w_bits=4, w_dtype="fp16", w_outlier=args.w_outlier,
        a_bits=4, a_dtype="fp16",
        w_groupsize=args.groupsize, a_groupsize=args.groupsize,
        w_type_block="1x16", a_type_block="1x16",
    )

    print(f"Loading {args.model_name} ({model2path[args.model_name]}) ...", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, quant_config=quant_config, device_map="cuda:0", use_fp16=False
    )
    model.seq_len = args.seq_len

    print("Snapshotting pristine weights to CPU ...", flush=True)
    pristine = snapshot_weights(model)

    print(f"Preparing evaluation data: {args.datasets}", flush=True)
    data = {}
    for ds in args.datasets:
        if ds == "wikitext":
            data[ds] = build_wikitext(tokenizer, args.seq_len)
        elif ds == "c4":
            data[ds] = build_c4(tokenizer, args.seq_len)
        else:
            raise ValueError(f"Unknown dataset {ds}")
        if args.limit_samples is not None:
            data[ds] = data[ds][:, : args.limit_samples * args.seq_len]
        print(f"  {ds}: {data[ds].numel() // args.seq_len} windows of {args.seq_len}", flush=True)

    # Importance must be measured on the UNQUANTIZED model, before any weight is overwritten.
    # Only the *_hess configurations consume it.
    importance = None
    if any("hess" in c[1] or "hess" in c[3] for c in configs):
        t0 = time.time()
        calib = build_calibration(tokenizer, args.seq_len, args.calib_batches)
        importance = collect_importance(model, calib, device=model.device)
        print(f"Collected activation importance for {len(importance)} layers "
              f"in {time.time() - t0:.0f}s", flush=True)

    results = {}
    if os.path.isfile(args.output):
        results = json.load(open(args.output))

    for label, w_dtype, w_tb, a_dtype, a_tb in configs:
        if label in results:
            print(f"[skip] {label} already in {args.output}", flush=True)
            continue

        t0 = time.time()
        restore_weights(model, pristine)

        quant_config.w_dtype      = w_dtype
        quant_config.w_type_block = w_tb
        quant_config.a_dtype      = a_dtype
        quant_config.a_type_block = a_tb
        quant_config.w_bits       = 16 if w_dtype == "fp16" else 4
        quant_config.a_bits       = 16 if a_dtype == "fp16" else 4

        if w_dtype != "fp16":
            quant_weight(model, quant_config, importance=importance)
        t_quant = time.time() - t0

        entry = {"w_dtype": w_dtype, "w_type_block": w_tb, "a_dtype": a_dtype,
                 "a_type_block": a_tb, "quant_sec": round(t_quant, 1)}
        for ds, ids in data.items():
            t1 = time.time()
            entry[ds] = eval_ppl(model, ids, args.seq_len, desc=f"{label}/{ds}")
            entry[f"{ds}_sec"] = round(time.time() - t1, 1)
            print(f"  {label:18s} {ds:9s} ppl = {entry[ds]:.4f}  "
                  f"(quant {t_quant:.0f}s, eval {entry[f'{ds}_sec']:.0f}s)", flush=True)

        results[label] = entry
        # write after every config so a crash never loses completed work
        if os.path.isfile(args.output):
            results = {**json.load(open(args.output)), **results}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results in {args.output}", flush=True)


if __name__ == "__main__":
    main()

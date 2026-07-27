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

from quantize import QuantConfig, quant_weight
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
    ("mixfp4_1x16",           "mixfp4",           "1x16",   "fp16", "1x16"),
    ("mixfp4_16x16",          "mixfp4",           "16x16",  "fp16", "1x16"),
    ("mixfp4_256x16",         "mixfp4",           "256x16", "fp16", "1x16"),
    ("mixfp4_8x64",           "mixfp4",           "8x64",   "fp16", "1x16"),
    ("mixfp4_16x64",          "mixfp4",           "16x64",  "fp16", "1x16"),
    ("mixfp4_32x64",          "mixfp4",           "32x64",  "fp16", "1x16"),
    ("mixfp4_32x128",         "mixfp4",           "32x128", "fp16", "1x16"),
]

SWEEP_W4A4 = [
    ("fp16",                  "fp16",             "1x16",   "fp16",             "1x16"),
    ("mxfp4",                 "mxfp4",            "1x16",   "mxfp4",            "1x16"),
    ("nvfp4",                 "nvfp4",            "1x16",   "nvfp4",            "1x16"),
    ("nvfp4_4over6",          "nvfp4_4over6",     "1x16",   "nvfp4_4over6",     "1x16"),
    ("nvif4",                 "nvif4",            "1x16",   "nvif4",            "1x16"),
    ("razer",                 "nvfp4_razer_e3m3", "1x16",   "nvfp4_razer_e4m3", "1x16"),
    ("mixfp4_1x16",           "mixfp4",           "1x16",   "mixfp4",           "1x16"),
    ("mixfp4_16x16",          "mixfp4",           "16x16",  "mixfp4",           "16x16"),
    ("mixfp4_256x16",         "mixfp4",           "256x16", "mixfp4",           "256x16"),
    ("mixfp4_8x64",           "mixfp4",           "8x64",   "mixfp4",           "16x64"),
    ("mixfp4_16x64",          "mixfp4",           "16x64",  "mixfp4",           "16x64"),
    ("mixfp4_32x64",          "mixfp4",           "32x64",  "mixfp4",           "32x64"),
    ("mixfp4_32x128",         "mixfp4",           "32x128", "mixfp4",           "32x128"),
]

SWEEPS = {"w4a16": SWEEP_W4A16, "w4a4": SWEEP_W4A4}


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
            quant_weight(model, quant_config)
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

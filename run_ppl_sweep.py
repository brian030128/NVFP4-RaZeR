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

SWEEPS = {"w4a16": SWEEP_W4A16, "w4a4": SWEEP_W4A4}


# Formats that ignore --w_type_block entirely, so their label carries no type block.
TYPELESS = {"fp16", "mxfp4", "nvfp4", "nvfp4_4over6", "nvfp4_nover6", "nvif4",
            "nvfp4_razer_e3m3", "nvfp4_razer_e4m3"}

# The RaZeR weight/activation pair, so "razer" can be named as one config.
RAZER_PAIR = {"razer": ("nvfp4_razer_e3m3", "nvfp4_razer_e4m3")}


def parse_configs(spec: str, quantize_activations: bool):
    """
        Build the sweep from a command-line spec instead of the hardcoded lists, so that exploring
        new selection rules does not need a code edit (and a stale copy on every other GPU).

            "<dtype>[/<a_dtype>][@<type_block>]" , comma separated

        Examples:
            fp16,nvfp4_4over6,mix_4_6_clipc3_m2@8x64,mix_4_6_mae_rm2@32x128
            mix_4_6_perm_h3/mix_4_6_h3@8x64        # row sorting on weights only

        The optional "/<a_dtype>" gives the activations a DIFFERENT data type from the weights. That
        matters for anything that is only deployable on one operand: sorting rows is a weight
        rewrite done once offline, but on activations it would be a gather on every GEMM, so a
        W4A4 row must be able to say "sorted weights, unsorted activations".

        The label is "<dtype>_<type_block>", or just "<dtype>" for formats without a type block, so
        it matches the labels the hardcoded sweeps already wrote.
    """
    rows = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        head, _, tb = item.partition("@")
        dtype, _, a_override = head.partition("/")
        tb = tb or "1x16"
        w_dtype, a_dtype = RAZER_PAIR.get(dtype, (dtype, dtype))
        if a_override:
            a_dtype = a_override
        label = dtype if (dtype in TYPELESS or dtype in RAZER_PAIR) else f"{dtype}_{tb}"
        if a_override:
            label = f"{label}__a-{a_override}"

        if dtype == "fp16":
            rows.append((label, "fp16", "1x16", "fp16", "1x16"))
        elif quantize_activations:
            rows.append((label, w_dtype, tb, a_dtype, A_BLOCK_FOR.get(tb, tb)))
        else:
            rows.append((label, w_dtype, tb, "fp16", "1x16"))
    return rows


def build_calibration(tokenizer, seq_len, num_batches, seed=0, source="wikitext"):
    """
        Calibration sequences for the activation-importance estimate.

        Drawn from the wikitext TRAIN split by default, never from the evaluation data -- estimating
        the importance on the same tokens we then report perplexity on would leak the test set into
        the quantization decisions.

        `source` selects the corpus, so that calibrating on one and evaluating on another can be
        measured rather than assumed. It appears not to matter much: importance estimated on c4
        reproduces 94.3% of the wikitext-derived tile elections, against 95.7% for a single wikitext
        batch and 88.1% for random token ids -- so using REAL text matters, which text much less.
    """
    if source == "c4":
        train = load_dataset(
            "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train")
        ids = tokenizer("\n\n".join(train[i]["text"] for i in range(4000)),
                        return_tensors="pt").input_ids
    elif source == "random":
        # the control: no real text at all
        g = torch.Generator().manual_seed(seed)
        vocab = getattr(tokenizer, "vocab_size", 32000)
        return [torch.randint(0, vocab, (1, seq_len), generator=g) for _ in range(num_batches)]
    else:
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
    parser.add_argument("--sweep", type=str, default="w4a16", choices=list(SWEEPS.keys()),
                        help="Hardcoded sweep; also selects W4A16 vs W4A4 for --configs.")
    parser.add_argument("--configs", type=str, default=None,
                        help='Explicit config list, e.g. "nvfp4_4over6,mix_4_6_clipc3_m2@8x64". '
                             "Overrides the hardcoded --sweep list.")
    parser.add_argument("--groupsize", type=int, default=16)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--w_outlier", type=float, default=8.0)
    parser.add_argument("--calib_source", type=str, default="wikitext",
                        choices=["wikitext", "c4", "random"],
                        help="Corpus for the importance estimate. Lets calibrate-on-one, "
                             "evaluate-on-another be measured instead of assumed.")
    parser.add_argument("--calib_batches", type=int, default=4,
                        help="Calibration sequences used to estimate activation importance.")
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="Cap the number of evaluated windows (smoke tests).")
    args = parser.parse_args()

    set_seed(0)
    if args.configs:
        configs = parse_configs(args.configs, quantize_activations=(args.sweep == "w4a4"))
    else:
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

    # Building c4 tokenizes 256 documents one at a time and takes ~15 minutes, which every shard of
    # every sweep was paying independently. The result is a deterministic function of
    # (model, dataset, seq_len) -- the sampling is seeded -- so it is cached to disk and shared.
    # Correctness note: the cache key includes the model, because the tokenizer differs per model,
    # and the eval set must stay byte-identical across configurations for the deltas to mean
    # anything. A stale cache would silently compare against different text.
    print(f"Preparing evaluation data: {args.datasets}", flush=True)
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".eval_cache")
    os.makedirs(cache_dir, exist_ok=True)
    data = {}
    for ds in args.datasets:
        cache = os.path.join(cache_dir, f"{args.model_name}_{ds}_{args.seq_len}.pt")
        if os.path.isfile(cache):
            data[ds] = torch.load(cache)
            print(f"  {ds}: loaded from {cache}", flush=True)
        elif ds == "wikitext":
            data[ds] = build_wikitext(tokenizer, args.seq_len)
        elif ds == "c4":
            data[ds] = build_c4(tokenizer, args.seq_len)
        else:
            raise ValueError(f"Unknown dataset {ds}")
        if not os.path.isfile(cache):
            # atomic-ish: write to a shard-private temp then rename, so concurrent shards racing on
            # the same key cannot leave a half-written file behind
            tmp = f"{cache}.tmp{args.shard_id}"
            torch.save(data[ds], tmp)
            os.replace(tmp, cache)
        if args.limit_samples is not None:
            data[ds] = data[ds][:, : args.limit_samples * args.seq_len]
        print(f"  {ds}: {data[ds].numel() // args.seq_len} windows of {args.seq_len}", flush=True)

    # Importance must be measured on the UNQUANTIZED model, before any weight is overwritten.
    # Only the *_hess configurations consume it.
    importance = None
    if any("hess" in c[1] or "hess" in c[3] for c in configs):
        t0 = time.time()
        calib = build_calibration(tokenizer, args.seq_len, args.calib_batches,
                                  source=args.calib_source)
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

"""
    Zero-shot accuracy sweep over quantization formats.

    This is `run_ppl_sweep.py` with lm-eval-harness in place of the perplexity loop. The
    quantization path is deliberately identical: the same `parse_configs` spec, the same
    `QuantConfig` mutated in place, the same pristine-CPU-snapshot restore before every
    configuration, and the same `collect_importance` pass taken on the UNQUANTIZED model before
    any weight is overwritten. That is what makes an accuracy delta comparable to the perplexity
    delta in `results/MIXFP4_REPORT.md` §1 -- both come out of the same weights.

    Why a sweep rather than N invocations of `run_zeroshot.py`: the model is loaded once and
    re-quantized in place, so an 8B model pays one load instead of seven, and every configuration
    sees byte-identical task data.

    Example (the §1 panel):
        python run_zeroshot_sweep.py --model_name qwen3-8b --sweep w4a4 \
            --configs "nvfp4/nvfp4_4over6,mix_4_6_clipa1_hess_impg16_h10/nvfp4_4over6@8x64" \
            --output results/zeroshot_w4a4/qwen3-8b.json

    Shard across GPUs exactly as the perplexity sweep does:
        CUDA_VISIBLE_DEVICES=0 python run_zeroshot_sweep.py ... --shard_id 0 --num_shards 2
"""

import argparse
import json
import os
import time
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

from quantize import QuantConfig, quant_weight, collect_importance
from run_ppl_sweep import build_calibration, parse_configs, restore_weights, snapshot_weights
from utils import load_model_and_tokenizer, set_seed, model2path


# The standard zero-shot multiple-choice panel. Script-based loaders are gone from datasets 4.x,
# so a task can fail to download for reasons that have nothing to do with the model; each is
# probed once up front and any that cannot load is recorded and skipped rather than failing the
# run (same handling as `run_zeroshot_check.py`).
DEFAULT_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "openbookqa",
                 "boolq", "winogrande", "piqa")

# lm-eval reports several metrics per task; take the one the literature quotes for that task.
# acc_norm (length-normalized) where the harness provides it, plain acc otherwise.
METRIC_ORDER = ("acc_norm,none", "acc,none")


def usable_tasks(tasks):
    """Probe each task's dataset once, so a download failure is reported rather than fatal."""
    import lm_eval

    ok, skipped = [], {}
    for task in tasks:
        try:
            lm_eval.tasks.TaskManager().load_task_or_group([task])
            ok.append(task)
        except Exception as exc:  # dataset availability, not a model property
            skipped[task] = repr(exc)[:200]
            print(f"SKIP {task}: {repr(exc)[:120]}", flush=True)
    assert ok, f"No task could be loaded: {skipped}"
    return ok, skipped


def evaluate(model, tokenizer, tasks, batch_size, limit=None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    # HFLM wraps the live module, and weights are rewritten in place, so it is rebuilt per
    # configuration only to reset the harness's own caches -- the model is never reloaded.
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    res = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), num_fewshot=0,
                                  batch_size=batch_size, limit=limit)["results"]
    acc = {}
    for task, values in res.items():
        for key in METRIC_ORDER:
            if key in values:
                acc[task] = dict(metric=key.split(",")[0], value=values[key],
                                 stderr=values.get(key.replace(",none", "_stderr,none")))
                break
    acc["mean"] = sum(v["value"] for k, v in acc.items() if k != "mean") / len(acc)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--sweep", type=str, default="w4a4", choices=["w4a16", "w4a4"],
                        help="Selects whether --configs quantizes activations.")
    parser.add_argument("--configs", type=str, required=True,
                        help='Config list in run_ppl_sweep.py syntax, e.g. '
                             '"nvfp4/nvfp4_4over6,mix_4_6_clipa1_hess_h10/nvfp4_4over6@8x64".')
    parser.add_argument("--tasks", type=lambda s: tuple(s.split(",")), default=DEFAULT_TASKS)
    parser.add_argument("--groupsize", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Calibration sequence length; matches the perplexity sweep.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--w_outlier", type=float, default=8.0)
    parser.add_argument("--calib_source", type=str, default="wikitext",
                        choices=["wikitext", "c4", "random"])
    parser.add_argument("--calib_batches", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap documents per task (smoke tests only -- not a reportable run).")
    args = parser.parse_args()

    assert os.environ.get("SLURM_JOB_ID"), "Use Slurm -- see /home/u4320956/CLAUDE.md"

    # Same seed as the perplexity sweep, so the calibration windows are the same windows.
    set_seed(0)
    configs = parse_configs(args.configs, quantize_activations=(args.sweep == "w4a4"))
    configs = [c for i, c in enumerate(configs) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(configs)} configs: "
          f"{[c[0] for c in configs]}", flush=True)

    tasks, skipped = usable_tasks(args.tasks)
    print(f"TASKS {tasks}", flush=True)

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

    # Importance must come off the unquantized model. Only *_hess configurations consume it.
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
    meta = {"model": args.model_name, "source": model2path[args.model_name],
            "job_id": os.environ["SLURM_JOB_ID"], "tasks": list(tasks),
            "tasks_skipped": skipped, "batch_size": args.batch_size,
            "calib_source": args.calib_source, "calib_batches": args.calib_batches,
            "torch_version": torch.__version__, "limit": args.limit}

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

        t1 = time.time()
        acc = evaluate(model, tokenizer, tasks, args.batch_size, limit=args.limit)
        entry = {"w_dtype": w_dtype, "w_type_block": w_tb, "a_dtype": a_dtype,
                 "a_type_block": a_tb, "quant_sec": round(t_quant, 1),
                 "eval_sec": round(time.time() - t1, 1), "accuracy": acc}
        print(f"  {label:44s} mean {acc['mean']:.4f}  " +
              " ".join(f"{k}={v['value']:.4f}" for k, v in acc.items() if k != "mean"),
              flush=True)

        results[label] = entry
        # write after every config so a crash never loses completed work
        if os.path.isfile(args.output):
            results = {**json.load(open(args.output)), **results}
        with open(args.output, "w") as f:
            json.dump({"_meta": meta, **{k: v for k, v in results.items() if k != "_meta"}},
                      f, indent=2)

    print(f"\nDone. Results in {args.output}", flush=True)


if __name__ == "__main__":
    main()

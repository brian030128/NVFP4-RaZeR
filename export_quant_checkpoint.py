"""
    Write a fake-quantized model out as an ordinary HuggingFace checkpoint.

    Everything else in this repo evaluates quantization inside its own process, which is fine for
    perplexity and lm-eval but useless for an agentic benchmark: Terminal-Bench drives an agent
    that talks to an OpenAI-compatible endpoint, so the quantized model has to be *served*. Since
    every format here is emulated and written back into BF16 weights, the quantized model IS an
    ordinary BF16 checkpoint -- so quantize, `save_pretrained`, and any serving stack can load it.

    What this does and does not preserve:

      * weights          -- quantized exactly as `run_ppl_sweep.py` would (same `quant_weight`,
                            same importance pass on the pristine model).
      * activations      -- NOT quantized. The activation path lives in `models/qmodule_*.py`
                            and does not survive `save_pretrained`, so a served checkpoint is
                            W4A16, not the W4A4 of MIXFP4_REPORT §1. Any comparison built on this
                            must say so.
      * lm_head / norms  -- untouched, matching `quant_weight`'s own name filter.

        python export_quant_checkpoint.py --model_name qwen3-8b \
            --config "mix_4_6_clipa1_hess_impg16_h10@8x64" --out /tmp/ckpt/mix
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize import QuantConfig, quant_weight, collect_importance
from run_ppl_sweep import build_calibration, parse_configs
from utils import model2path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--config", required=True,
                    help='One entry in run_ppl_sweep.py syntax, e.g. "nvfp4" or '
                         '"mix_4_6_clipa1_hess_impg16_h10@8x64". Activation overrides ("/...") '
                         'are rejected -- a saved checkpoint cannot carry them.')
    ap.add_argument("--out", required=True)
    ap.add_argument("--groupsize", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--calib_source", default="wikitext", choices=["wikitext", "c4", "random"])
    ap.add_argument("--calib_batches", type=int, default=4)
    args = ap.parse_args()

    assert os.environ.get("SLURM_JOB_ID"), "Use Slurm -- see /home/u4320956/CLAUDE.md"
    assert "/" not in args.config.split("@")[0], \
        "activation quantization cannot be saved into a checkpoint; drop the /a_dtype override"

    rows = parse_configs(args.config, quantize_activations=False)
    assert len(rows) == 1, f"expected one config, got {[r[0] for r in rows]}"
    label, w_dtype, w_tb, _, _ = rows[0]

    path = model2path[args.model_name]
    print(f"Loading {args.model_name} ({path}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    quant_config = QuantConfig(
        w_bits=16 if w_dtype == "fp16" else 4, w_dtype=w_dtype, w_outlier=8.0,
        a_bits=16, a_dtype="fp16",
        w_groupsize=args.groupsize, a_groupsize=args.groupsize,
        w_type_block=w_tb, a_type_block="1x16",
    )

    importance = None
    if "hess" in w_dtype:
        t0 = time.time()
        calib = build_calibration(tok, args.seq_len, args.calib_batches, source=args.calib_source)
        importance = collect_importance(model, calib, device=model.device)
        print(f"Collected importance for {len(importance)} layers in {time.time()-t0:.0f}s",
              flush=True)

    # A checksum over a fixed slice of one weight, before and after, is the cheapest evidence
    # that the checkpoint on disk is actually the quantized one and not a silent no-op.
    probe = next(m for n, m in model.named_modules()
                 if isinstance(m, nn.Linear) and "head" not in n)
    before = probe.weight.detach().flatten()[:4096].float().clone()

    if w_dtype != "fp16":
        t0 = time.time()
        quant_weight(model, quant_config, importance=importance)
        print(f"Quantized in {time.time()-t0:.0f}s", flush=True)
    after = probe.weight.detach().flatten()[:4096].float()
    changed = bool(not torch.equal(before, after))
    print(f"probe weight changed: {changed}  (max |delta| = {(after-before).abs().max():.3e})",
          flush=True)
    assert changed or w_dtype == "fp16", "quantization was a no-op -- config did not apply"

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    meta = {"model_name": args.model_name, "source": path, "label": label,
            "w_dtype": w_dtype, "w_type_block": w_tb, "w_groupsize": args.groupsize,
            "activations": "not quantized (W4A16); the qmodule activation path is not saved",
            "calib_source": args.calib_source if importance else None,
            "calib_batches": args.calib_batches if importance else None,
            "probe_weight_changed": changed,
            "job_id": os.environ["SLURM_JOB_ID"]}
    with open(os.path.join(args.out, "quantization_provenance.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {label} to {args.out}", flush=True)


if __name__ == "__main__":
    main()

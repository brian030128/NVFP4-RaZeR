"""
    Block-wise teacher-forced pseudo-perplexity for google/diffusiongemma-26B-A4B-it, comparing
    weight-only FP4 quantization formats.

    WHY "pseudo": DiffusionGemma is an encoder(causal MoE)-decoder(bidirectional diffusion) block
    model. Its forward exposes no likelihood/loss and the denoiser is time-independent, so a true
    generative NLL/ELBO is NOT reconstructible (see results/diffgemma_ppl/REPORT.md). What IS
    well-defined and monotone in model quality is a ONE-STEP denoiser cross-entropy:

        for each 256-token block of real text, encode the *clean previous blocks* into the read-only
        KV cache, feed a max-noise (uniform-random) canvas as decoder_input_ids with
        self-conditioning OFF, and take the position-aligned cross-entropy of logits[:, :L] against
        the true block tokens. Average over a few random-canvas draws.

    This is an upper bound on the model's effective per-token loss and is only meaningful as a
    RELATIVE metric (FP vs quant). The canvas realizations are seeded identically across configs so
    the comparison is paired.

    Quantization is weight-only (W4A16), matching results/diffgemma_mse. It is applied IN PLACE
    (encoder and decoder share weight storage; rewriting once with .data.copy_ keeps the tie and
    hits both towers) and deduped by data_ptr so each physical tensor is quantized exactly once.
    Quantized: attention q/k/v/o, dense-MLP gate/up/down, and the 3-D MoE experts
    (gate_up_proj/down_proj, reduction axis = last). Left FP: router, lm_head, embeddings,
    self-conditioning, vision projection, norms.

    Activation quantization ("keep activation 4over6") is NOT applied: it would need a qmodule that
    inserts quant_act at the DiffusionGemma activation boundaries, which does not exist. This run
    isolates the weight-quantization effect, exactly like the MSE study.
"""

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import DiffusionGemmaForBlockDiffusion, AutoTokenizer

from quantize.quantizer import quant_nvfp4_4over6, quant_mixfp4, quant_mix_4_6


# ---------------------------------------------------------------- quantization

def get_quant_fn(name):
    """name -> fn(w2d)->dq (quantizes along dim=-1), or None for FP."""
    if name == "fp":
        return None
    if name == "nvfp4_4over6":
        return lambda w: quant_nvfp4_4over6(w, groupsize=16)
    if name == "mixfp4-8x64":
        return lambda w: quant_mixfp4(w, groupsize=16, type_block=(8, 64))
    if name == "mix_4_6-8x64":
        return lambda w: quant_mix_4_6(w, groupsize=16, type_block=(8, 64))
    raise ValueError(f"unknown config {name}")


def _is_experts(m):
    return type(m).__name__ == "DiffusionGemmaTextExperts"


# leave these in FP (match a W4A16 weight study)
SKIP_SUBSTR = ("lm_head", ".router.", ".self_conditioning.", ".embed_vision.",
               "embedding_projection", "embed_tokens")


@torch.no_grad()
def quantize_model_weights_(model, quant_fn):
    """
        Rewrite quantizable weights in place. Dedup shared (tied) storage by data_ptr.
        Scope: the TEXT language model only (decoder + tied encoder language_model). The vision
        tower is left FP -- it is never exercised by a text corpus and its odd tile dims (e.g. 4304)
        are not valid 8x64 type blocks. This matches the text-decoder scope of results/diffgemma_mse.
    """
    if quant_fn is None:
        return 0, 0
    seen = set()
    n_lin = n_exp = 0

    def skip(name):
        padded = f".{name}."          # so ".router." etc. match regardless of leading context
        return any(s in padded for s in SKIP_SUBSTR)

    def in_text(name):
        padded = f".{name}."
        return ".decoder." in padded or ".language_model." in padded

    for name, module in model.named_modules():
        if not in_text(name):
            continue
        if isinstance(module, nn.Linear):
            if skip(name):
                continue
            w = module.weight
            if w.shape[-1] % 16 != 0:            # not tileable by the NVFP4 scale block
                continue
            if w.data_ptr() in seen:
                continue
            seen.add(w.data_ptr())
            w.data.copy_(quant_fn(w.data).to(w.dtype))
            n_lin += 1
        elif _is_experts(module):
            for pname in ("gate_up_proj", "down_proj"):
                p = getattr(module, pname)
                if p.data_ptr() in seen or p.shape[-1] % 16 != 0:
                    continue
                seen.add(p.data_ptr())
                for e in range(p.shape[0]):
                    p.data[e].copy_(quant_fn(p.data[e]).to(p.dtype))
                n_exp += 1
    return n_lin, n_exp


# ---------------------------------------------------------------- scoring

@torch.no_grad()
def extend_cache(model, past, clean_ids):
    """Append clean context tokens to the read-only encoder KV cache."""
    enc = model.model.encoder(input_ids=clean_ids, attention_mask=None,
                              past_key_values=past, position_ids=None)
    return enc.past_key_values


@torch.no_grad()
def decode_logits(model, past, canvas):
    """One denoiser forward against the current cache; returns (B, L, vocab) position-aligned."""
    out = model(input_ids=None, past_key_values=past, decoder_input_ids=canvas,
                self_conditioning_logits=None, self_conditioning_mask=None,
                decoder_attention_mask=None, decoder_position_ids=None)
    return out.logits[:, : canvas.shape[1]]


@torch.no_grad()
def pseudo_ppl(model, blocks, vocab, canvas_len, window_blocks, max_windows, n_draws, base_seed,
               reveal_frac=0.0):
    """
        Windowed block-wise teacher-forced denoiser CE. Each window: block 0 = clean context,
        blocks 1.. are scored given all earlier true blocks in the window (cache reset per window
        to bound the KV cache ~ window_blocks*canvas_len and mirror a seq_len window).

        reveal_frac in [0,1): fraction of canvas positions filled with the TRUE token (a partial-
        denoising probe); the CE is computed only over the remaining MASKED (random) positions.
        reveal_frac=0 => max-noise canvas, CE over all positions. Canvas noise AND the reveal mask
        are seeded by (global_block_index, draw) so every config sees identical probes (paired).
    """
    dev = model.device
    total_nll, total_tok, n_scored = 0.0, 0, 0
    windows = [blocks[i:i + window_blocks] for i in range(0, len(blocks), window_blocks)]
    if max_windows:
        windows = windows[:max_windows]

    gen = torch.Generator(device=dev)
    t0 = time.time()
    for wi, win in enumerate(windows):
        if len(win) < 2:
            continue
        past = extend_cache(model, None, win[0].unsqueeze(0).to(dev))   # context = first block
        for bi in range(1, len(win)):
            true = win[bi].unsqueeze(0).to(dev)
            L = true.shape[1]
            gidx = wi * window_blocks + bi
            for d in range(n_draws):
                gen.manual_seed(base_seed + gidx * 131 + d)
                canvas = torch.randint(0, vocab, (1, L), generator=gen, device=dev, dtype=torch.long)
                masked = torch.ones(L, dtype=torch.bool, device=dev)
                if reveal_frac > 0:
                    k = int(round(reveal_frac * L))
                    if k > 0:
                        rev = torch.randperm(L, generator=gen, device=dev)[:k]
                        canvas[0, rev] = true[0, rev]        # reveal true tokens
                        masked[rev] = False                  # score only the still-noisy ones
                logits = decode_logits(model, past, canvas).float()[0][masked]
                total_nll += F.cross_entropy(logits, true[0][masked], reduction="sum").item()
                total_tok += int(masked.sum().item())
            n_scored += 1
            past = extend_cache(model, past, true)                     # add clean block to context
        if (wi + 1) % 10 == 0 or wi == len(windows) - 1:
            ppl_so_far = math.exp(total_nll / total_tok)
            print(f"    window {wi+1}/{len(windows)}  scored_blocks={n_scored}  "
                  f"pPPL={ppl_so_far:.3f}  ({time.time()-t0:.1f}s)", flush=True)
    return math.exp(total_nll / total_tok), n_scored, total_tok


# ---------------------------------------------------------------- data / model

def load_blocks(tokenizer, canvas_len, max_tokens):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    if max_tokens:
        enc = enc[:max_tokens]
    n_full = enc.numel() // canvas_len
    enc = enc[: n_full * canvas_len]
    print(f"wikitext-2 test: {enc.numel()} tokens -> {n_full} full {canvas_len}-token blocks")
    return list(enc.split(canvas_len))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--configs", default="fp,nvfp4_4over6,mixfp4-8x64,mix_4_6-8x64",
                    type=lambda s: s.split(","))
    ap.add_argument("--out", required=True)
    ap.add_argument("--window_blocks", type=int, default=8)   # 8*256 = 2048-token context window
    ap.add_argument("--max_windows", type=int, default=32)    # 0 = all
    ap.add_argument("--n_draws", type=int, default=2)         # random-canvas draws averaged
    ap.add_argument("--reveal_frac", type=float, default=0.0)  # frac of true tokens shown (probe level)
    ap.add_argument("--max_tokens", type=int, default=0)      # 0 = all test tokens
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)

    # peek canvas_len / vocab from a config load (cheap)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_dir)
    canvas_len = cfg.canvas_length
    vocab = cfg.text_config.vocab_size
    print(f"canvas_length={canvas_len}  vocab_size={vocab}  configs={args.configs}")

    blocks = load_blocks(tok, canvas_len, args.max_tokens)

    rows = []
    fp_ppl = None
    for cname in args.configs:
        print(f"\n=================== config: {cname} ===================", flush=True)
        t0 = time.time()
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, device_map="cuda:0")
        model.eval()
        print(f"  loaded in {time.time()-t0:.1f}s  device={model.device}", flush=True)

        qfn = get_quant_fn(cname)
        n_lin, n_exp = quantize_model_weights_(model, qfn)
        print(f"  quantized: {n_lin} linear tensors, {n_exp} expert stacks "
              f"({'FP baseline' if qfn is None else cname})", flush=True)

        ppl, n_scored, n_tok = pseudo_ppl(
            model, blocks, vocab, canvas_len,
            args.window_blocks, args.max_windows, args.n_draws, args.seed,
            reveal_frac=args.reveal_frac)

        if cname == "fp":
            fp_ppl = ppl
        delta = (ppl - fp_ppl) if fp_ppl is not None else float("nan")
        rows.append({"config": cname, "pseudo_ppl": ppl, "delta_vs_fp": delta,
                     "scored_blocks": n_scored, "tokens": n_tok,
                     "n_lin": n_lin, "n_exp": n_exp})
        print(f"  ==> {cname}: pseudo-PPL = {ppl:.4f}"
              + (f"  (Δ vs fp = {delta:+.4f})" if fp_ppl is not None and cname != 'fp' else ""),
              flush=True)

        del model
        torch.cuda.empty_cache()

        # persist after every config so a later crash cannot lose completed results
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["config", "pseudo_ppl", "delta_vs_fp",
                                               "scored_blocks", "tokens", "n_lin", "n_exp"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  wrote {args.out} ({len(rows)} configs so far)", flush=True)

    print("\n================= SUMMARY (block-wise teacher-forced pseudo-PPL, weight-only) =================")
    print(f"{'config':<18}{'pseudo-PPL':>12}{'Δ vs fp':>12}")
    for r in rows:
        d = "" if r["config"] == "fp" else f"{r['delta_vs_fp']:+.4f}"
        print(f"{r['config']:<18}{r['pseudo_ppl']:>12.4f}{d:>12}")
    print("\n(relative metric only; NOT a calibrated perplexity — see REPORT.md)")


if __name__ == "__main__":
    main()

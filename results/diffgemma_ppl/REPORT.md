# diffusiongemma-26B-A4B-it — "PPL directly": a block-diffusion pseudo-perplexity

**Follow-up to `results/diffgemma_mse/REPORT.md`.** The question was to compare `nvfp4_4over6`,
`mixfp4` @ 8x64 and `mix_4_6` @ 8x64 by *perplexity* instead of weight-MSE.

## TL;DR

**DiffusionGemma has no perplexity you can compute, and the closest proxy does not give a stable
answer.** It is an encoder(causal MoE)–decoder(bidirectional block-diffusion) model: the forward
returns logits but **no loss/likelihood**, the denoiser is **time-independent**, and the noise
process is **uniform-state (no mask token)** — so the generative NLL/ELBO cannot be reconstructed
from the released code. The defensible fallback is a **block-wise teacher-forced denoiser
cross-entropy** (a *pseudo*-perplexity, weight-only W4A16). It is monotone in model quality, but its
**ranking of the three formats flips with the probe noise level**, and neither level agrees with the
MSE ranking. So there is no robust PPL verdict — only the consistent facts that (a) all three formats
cost a lot on this model and (b) `mix_4_6` is always the middle option.

## What was measured

For each 256-token block of wikitext-2 test: encode the clean previous blocks into the read-only
encoder KV cache, feed a canvas as `decoder_input_ids` with self-conditioning off, and take the
position-aligned cross-entropy of `logits[:, :L]` against the true block tokens (no AR shift — the
diffusion decoder predicts the token at the same index). Averaged over 3 paired random draws; 2048-
token context windows; 448 scored blocks. Weight-only fake-quant is applied **in place** to the
**text language model only** (attention q/k/v/o, dense MLP, and the 3-D MoE experts), deduped across
the tied encoder/decoder towers; router / lm_head / embeddings / self-conditioning / **vision tower**
stay FP. This matches the text-decoder scope of the MSE study. Two probes:

- **max-noise** (`reveal_frac=0`): canvas is pure uniform-random; CE over all 256 positions. This is
  the honest one-step conditional likelihood — an *upper bound* on the effective per-token loss.
- **50%-reveal** (`reveal_frac=0.5`): half the canvas positions are set to the true token; CE over
  only the masked half. A milder, mid-denoising probe.

Activation quantization ("keep activation 4over6") is **not** applied — it would need a qmodule that
inserts `quant_act` at the DiffusionGemma boundaries, which does not exist. This isolates the
weight-quantization effect, exactly like the MSE study.

**These numbers are a relative proxy, not a calibrated perplexity.** The absolute values (and the
fp-vs-quant gaps) depend entirely on the probe; only same-probe comparisons are meaningful.

## Results

Pseudo-PPL (lower = better), 1×H100:

| config | max-noise pPPL | Δ vs fp | 50%-reveal pPPL | Δ vs fp |
|---|---|---|---|---|
| fp (bf16) | 2885.0 | — | 92.9 | — |
| nvfp4_4over6 | 5530.9 | +2645.8 | **130.8** | **+37.9** |
| mixfp4-8x64 | **4571.7** | **+1686.6** | 216.4 | +123.6 |
| mix_4_6-8x64 | 5073.1 | +2188.1 | 157.1 | +64.2 |

## The ranking is probe-dependent — three metrics, three different winners

| | weight-MSE (NMSE) | pPPL max-noise | pPPL 50%-reveal |
|---|---|---|---|
| **best** | **mix_4_6** (7.297e-3) | **mixfp4** (Δ+1687) | **nvfp4_4over6** (Δ+38) |
| middle | nvfp4_4over6 (7.555e-3) | mix_4_6 (Δ+2188) | mix_4_6 (Δ+64) |
| **worst** | **mixfp4** (7.637e-3) | nvfp4_4over6 (Δ+2646) | mixfp4 (Δ+124) |

`mixfp4` is simultaneously the **worst** format on weight-MSE, the **best** under the max-noise
probe, and the **worst** again under the 50%-reveal probe. `nvfp4_4over6` does the reverse. Only
`mix_4_6` is stable — always the middle option, never the best.

There is a coherent (if unhelpful) reading of the crossover: as the probe gets *easier* (more true
tokens revealed → smaller, more local prediction problem), the ranking drifts toward the weight-MSE
ranking (`mixfp4` worst); as it gets *harder* (pure generation from context), it inverts. But both
ends are pseudo-metrics, and the real generative process (multi-step accept/renoise with
self-conditioning) is neither. Nothing here pins down which format is actually best in deployment.

This is the same lesson as the rest of the repo, sharpened: **weight-MSE does not predict behavior,
and for a model with no calibrated likelihood the behavioral proxy is itself unstable.** The MSE win
that `mix_4_6` shows (−3.4% NMSE vs `4over6`) buys nothing you can count on — under the max-noise
probe `mix_4_6` is *worse* than `4over6`; under the 50%-reveal probe it is worse still than the
`4over6` that MSE ranked below it.

## What is actually solid

1. **All three FP4 formats degrade this model substantially under any behavioral probe** — even the
   mild 50%-reveal probe shows +41% (`4over6`) to +133% (`mixfp4`) pseudo-PPL, far larger than the
   ~sub-percent NMSE gaps suggest. Weight-only W4A16 is not cheap on DiffusionGemma.
2. **No format is robustly best.** MSE, and the pseudo-PPL at two probe levels, each crown a
   different one. If a single choice is forced, `mix_4_6` is the only never-worst option, but it is
   also never the best.
3. **A true perplexity would require the training objective** (noise schedule + time-conditioning),
   which is not in transformers 5.15. A calibrated verdict needs a real end-task eval
   (e.g. accuracy on a generation benchmark), not a likelihood proxy.

## Reproduce

```bash
# max-noise probe (reveal_frac=0):
sbatch --export=ALL,MAXWIN=64,NDRAWS=3,REVEAL=0.0,TAG=full     slurm/diffgemma_ppl.sbatch
# 50%-reveal probe:
sbatch --export=ALL,MAXWIN=64,NDRAWS=3,REVEAL=0.5,TAG=reveal50 slurm/diffgemma_ppl.sbatch
# results: results/diffgemma_ppl/diffgemma_ppl_{full,reveal50}.csv
```

Runs in the layered venv `/home/u4320956/diffgemma_venv` (transformers 5.15 for the DiffusionGemma
class, reusing the cuda128 env's torch; the shared cuda128 env is left untouched). ~16 min/probe on
one H100 incl. a 52 GB node-local download.

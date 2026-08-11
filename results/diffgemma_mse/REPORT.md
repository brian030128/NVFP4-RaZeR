# diffusiongemma-26B-A4B-it — weight-MSE: nvfp4_4over6 vs mixfp4 @ 8x64

**Model:** `google/diffusiongemma-26B-A4B-it` — DiffusionGemma, a **block-diffusion MoE**
(30 decoder layers, 128 experts / top-8, `hidden=2816`, `moe_intermediate=704`, plus a vision
tower). Not a causal LM, so the weights were read **straight from the safetensors shards**
(`run_diffgemma_mse.py`) — no model graph, no transformers arch support needed.

**What was measured:** fake-quantize every quantizable weight of the **text decoder** and report the
relative quantization error `NMSE = ||w - w_q||^2 / ||w||^2` (energy-weighted: `sum noise / sum
signal`) and SQNR. Weight-only — activation quantization ("keep activation 4over6") does not enter a
weight MSE and was not simulated. Formats, all groupsize 16 (the NVFP4 scale block):

| format | E2M1 block-scale search | E0M3 | selection |
|---|---|---|---|
| `nvfp4` | `/6` only | — | E2M1 only (floor) |
| **`nvfp4_4over6`** | `/6` and `/4` (FourOverSix) | — | per-block argmin **← baseline** |
| **`mixfp4` @ 8x64** | `/6` only | `/7` | per-8x64-type-block argmin MSE **← comparison** |
| `mix_4_6` @ 8x64 | `/6` and `/4` | `/7` | per-8x64-type-block argmin MSE (context) |

Ran on one H100 (dev), model downloaded to node-local `/tmp`; 265 decoder tensors, 52.6 s of GPU
compute. The MoE experts are stacked `(E, out, in)` tensors — each expert is its own GEMM operand,
so it was sliced per-expert (K = `in` already last) and given its own NVFP4 per-tensor global scale.

## Result — 4over6 wins; plain mixfp4 @ 8x64 is slightly worse

Energy-weighted NMSE (SQNR dB), over all decoder weights:

| group | `nvfp4` | **`nvfp4_4over6`** | **`mixfp4` 8x64** | `mix_4_6` 8x64 |
|---|---|---|---|---|
| **ALL** | 8.99e-3 (20.5) | **7.555e-3 (21.2)** | **7.637e-3 (21.2)** | 7.297e-3 (21.4) |
| attention | 8.95e-3 (20.5) | 7.557e-3 (21.2) | 7.800e-3 (21.1) | 7.348e-3 (21.3) |
| dense_mlp | 8.85e-3 (20.5) | 7.549e-3 (21.2) | 8.109e-3 (20.9) | 7.452e-3 (21.3) |
| moe_experts | 9.00e-3 (20.5) | 7.555e-3 (21.2) | 7.620e-3 (21.2) | 7.291e-3 (21.4) |

**Headline (mixfp4 8x64 vs nvfp4_4over6, NMSE):**

| group | 4over6 | mixfp4 8x64 | Δ | winner |
|---|---|---|---|---|
| **ALL** | 7.555e-3 | 7.637e-3 | **+1.09%** | **4over6** |
| attention | 7.557e-3 | 7.800e-3 | +3.22% | 4over6 |
| dense_mlp | 7.549e-3 | 8.109e-3 | +7.41% | 4over6 |
| moe_experts | 7.555e-3 | 7.620e-3 | +0.87% | 4over6 |

**`nvfp4_4over6` beats `mixfp4` @ 8x64 everywhere** — by +1.09% NMSE overall (worst on the dense
MLP at +7.41%, least bad on the MoE experts at +0.87%).

## Why — the same pattern the rest of this repo keeps finding

Plain `mixfp4` is a pure **argmin-MSE** choice between E2M1 (block_max/6) and E0M3 (block_max/7),
with **no FourOverSix search on the E2M1 side**. On these weights E0M3 (the uniform grid) wins that
narrow argmin most of the time — **90.2%** of 8x64 type blocks elect E0M3:

| group | E0M3 elected — `mixfp4` | E0M3 elected — `mix_4_6` |
|---|---|---|
| ALL | 90.2% | 45.7% |
| attention | 79.4% | 35.1% |
| dense_mlp | 65.7% | 20.6% |
| moe_experts | 90.6% | 46.1% |

But once E2M1 is *allowed the `/4` block scale* (i.e. the same FourOverSix the baseline already
uses), it reclaims about half those blocks (E0M3 drops to 45.7%), and that combination
(`mix_4_6`, -3.4% vs 4over6) is what actually helps. `mixfp4` loses precisely because it commits to
E0M3 on a coarse tile by an MSE-argmin — the exact failure mode CLAUDE.md documents:

> A rule of the form "do X when it lowers the quantization error" always loses. The same rule with
> "…by a decisive margin" wins.

The takeaway matches the existing Llama/Qwen findings: **the width of the E2M1 block-scale search
(FourOverSix) is the free, robust win; adding E0M3 by a bare MSE-argmin on a realizable type block
is not.** If E0M3 is to be used on this model, it needs a margin/`h<λ>` election rather than argmin
(`mixfp4`) — that is what would need a perplexity run to tune, since (per CLAUDE.md) weight MSE only
certifies the sign of large changes, not small ones.

## Reproduce

```bash
sbatch slurm/diffgemma_mse.sbatch          # 1×H100 (dev), ~3 min incl. 52 GB download
# per-tensor NMSE: results/diffgemma_mse/diffgemma_mse_8x64.csv
```

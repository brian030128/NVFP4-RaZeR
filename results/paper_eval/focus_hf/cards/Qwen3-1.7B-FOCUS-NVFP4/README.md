---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags:
- quantization
- fp4
- nvfp4
- focus
---

# FOCUS NVFP4 scales for Qwen3-1.7B (our reproduction)

**This is our reproduction of FOCUS** ([“FOCUS: FP4 Optimization via Coupled-Relaxation and Dual-Granularity Scaling”](https://arxiv.org/abs/2608.01847), arXiv:2608.01847), **made as a baseline for the FlipQuant paper. It is not an official release by the FOCUS authors.**

## What this is

`focus.pt` holds the learned FOCUS scale parameters for [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, for every quantized linear layer (196 layers: all text-model linear layers except the LM head). Per layer, in FP32:

| field | count | role |
|---|---|---|
| `m` | one per 16-element block (88,080,384 in total) | the block-scale factor (Coupled-Relaxation Scaling); starts at 1 |
| `q` | one per 8-element sub-block, 2 per block (176,160,768 in total) | the sub-block relaxation logit (Dual-Granularity Scaling); coefficient `sigmoid(q)`, starts at `q = 6` |
| `s2` | one per tensor | the NVFP4 tensor scale `amax(abs(W)) / (6 · 448)`, frozen (it follows from the base weights) |

`m` and `q` are in row-major order of the weight `[out_features, in_features]`, 16-element blocks along `in_features`. Each layer also has `act_gscale = None`: activations use one scale per token (below), so there is no static activation scale. The file also stores the calibration settings (`config`) and the base model's id and revision (`source`).

**It is not a quantized checkpoint**: it holds no weights. Using it needs the BF16 base model at the pinned revision and the FOCUS implementation, which turns the base weights and these parameters into standard NVFP4 codes. For a block `b` of 16 weights with sub-blocks `k = 1, 2` of 8:

```
S_b   = e4m3( clamp( max|w_b| · m_b / (6 · s2), 2^-6, 448 ) )      # the block's E4M3 scale
c_i   = E2M1( clamp( w_i / (S_b · sigmoid(q_k) · s2), -6, 6 ) )      # element i in sub-block k
w'_i  = c_i · S_b · s2                                               # the deployed (dequantized) weight
```

`E2M1` rounds to the nearest FP4 value (ties to even); a block whose scaled maximum is 0 gets scale 1 before the clamp. `sigmoid(q)` only changes which code each element gets: the stored scales are `S_b` and `s2`, so the deployed model is plain NVFP4 (E2M1 elements, one E4M3 scale per 16 elements, one FP32 scale per tensor) and runs on standard NVFP4 kernels with no extra metadata or cost.

The code (our FOCUS implementation and the evaluation) will be released with the paper.

## Calibration

The hyperparameters follow the FOCUS paper's NVFP4 setting:

- learning rate 5e-3 for the block-scale factors `m`, 1e-3 for the relaxation logits `q`;
- 2 sub-blocks of 8 elements per 16-element block;
- KL-Top loss with k = 1000 (the BF16 model's top-1000 tokens per position, both distributions renormalized over them);
- AdamW (betas 0.9 / 0.999, weight decay 0) with a constant learning rate;
- 1 epoch, global batch 32;
- `q` initialized to 6 (and `m` to 1); seed 42 (the data order).

Only the scale parameters train; the model weights are frozen. The teacher is the unquantized BF16 model.

**Differences from the paper:**

- **Calibration data.** The FlipQuant release set: 256 windows of 512 tokens of math and code text (131,072 tokens; OpenWebMath and CodeParrot-clean), the same fit set as the FlipQuant release maps, so 8 optimizer steps. The paper uses 1,248 WikiText-2 training samples of 2,048 tokens (39 steps).
- **Activations**, during calibration and evaluation: FourOverSix with one scale per token, as in the FlipQuant paper's protocol (NVFP4 activations in which each 16-element block's scale maps the block maximum to 6 or to 4, whichever has the lower error, and one FP32 scale per token). The paper quantizes activations to NVFP4 with a static per-tensor scale.

Cost on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120), micro-batch 8 (4 forward/backward passes per step), activation checkpointing per decoder block:

| training (8 steps) | end to end (model load, data, training, save) | peak GPU memory |
|---:|---:|---:|
| 50 s | 55 s | 15.9 GiB |

The peak is PyTorch's peak allocated memory on the GPU (`torch.cuda.max_memory_allocated`). The training time includes the BF16 teacher's forward passes.

## Perplexity

Native sm_120 kernels on the NVIDIA RTX PRO 6000 Blackwell Workstation Edition, with FOCUS deployed as standard NVFP4 codes and per-token FourOverSix activations. WikiText-2 test (146 windows of 2048 tokens) and C4 validation (256 windows of 2048 tokens). Every row is evaluated on the same windows:

- BF16: the unquantized model;
- NVFP4: round-to-nearest NVFP4 weights, NVFP4 activations with one scale per token;
- FourOverSix: round-to-nearest FourOverSix weights and per-token FourOverSix activations (the reference);
- FOCUS: this state, with per-token FourOverSix activations.

ΔNLL is the mean paired per-window difference in nats per token against FourOverSix; ± 2 SE over windows; `*` marks |Δ| > 2 SE. The verdict is “better” or “worse” only when |Δ| > 2 SE. The numbers are also in `ppl_summary.json`.

| method | WikiText-2 PPL | C4 PPL | ΔNLL WikiText-2 vs FO6 (± 2 SE) | ΔNLL C4 vs FO6 (± 2 SE) | verdict (WikiText-2 / C4) |
|---|---:|---:|---:|---:|---|
| BF16 | 16.7162 | 19.2463 | -0.1527 ± 0.0112 * | -0.0828 ± 0.0042 * | better / better |
| NVFP4 | 18.8713 | 21.0349 | -0.0315 ± 0.0112 * | +0.0061 ± 0.0028 * | better / worse |
| FourOverSix (reference) | 19.4745 | 20.9072 | — | — | reference |
| FOCUS (this state) | 15.0890 | 19.5345 | -0.2551 ± 0.0145 * | -0.0679 ± 0.0043 * | better / better |

On this model FOCUS's WikiText-2 perplexity is below BF16's, while its C4 perplexity is above it; both are as measured on the same windows.

## How to read the file

With PyTorch alone:

```python
import torch

state = torch.load("focus.pt", map_location="cpu", weights_only=True)
state["config"]   # the calibration settings
state["source"]   # the base model id and revision
layer = state["layers"]["model.layers.0.self_attn.q_proj"]
layer["m"], layer["q"], layer["s2"]   # FP32 tensors (see above)
```

## License

This state is released under the base model's license, the Apache License 2.0.

`LICENSE` is a copy of the base model's license file (Qwen/Qwen3-1.7B at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `focus.pt` | 1,057,104,195 | `5fa877acad86e7b9f64a39936717f8032cab0696d0e657b0525010b4f06a4945` |
| `ppl_summary.json` | 3,865 | `ec585b2d65572e918eb36d08e7c7d78bc3f0d1bb191e70ca20caf6e917e8b245` |
| `LICENSE` | 11,343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |

- `focus.pt`: the FOCUS state (`{config, layers, source}`; above). It is the calibration output, unchanged.
- `ppl_summary.json`: the perplexities, the paired differences against FourOverSix, the evaluation settings and the calibration's settings and cost.
- `LICENSE`: see License.
